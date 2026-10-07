"""CPU-only recipe, launcher and pre-construction argument parsing checks."""

import ast
import contextlib
from dataclasses import dataclass, field
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rynnvla.api.launch import PRESETS, build_command, main
from rynnvla.api.train import check_output_dir, load_config, parse_training_options, resolve_resume, validate_options


ROOT = Path(__file__).resolve().parents[1]
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def recipe(stage):
    return load_config(ROOT / "rynnvla" / "configs" / PRESETS[stage])


def meta_config(stage):
    """Real formal widths, minimal depth/vocabulary; no physical model allocation."""
    small = stage in ("stage1_2b", "stage2_2b")
    return {
        **recipe(stage)["config_overrides"], "model_type": "rynn_brain_vla", "action_dim": 81,
        "text_config": {
            "hidden_size": 2048 if small else 2560,
            "intermediate_size": 6144 if small else 9728, "num_hidden_layers": 1,
            "num_attention_heads": 16 if small else 20,
            "num_key_value_heads": 8 if small else 4, "head_dim": 128,
            "vocab_size": 32, "max_position_embeddings": 4096,
        },
        "vision_config": {
            "hidden_size": 32, "intermediate_size": 64, "depth": 1,
            "num_heads": 4, "out_hidden_size": 2048 if small else 2560, "patch_size": 16,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0], "num_position_embeddings": 16,
        },
    }


@dataclass
class NoInitArguments:
    model_path: str = "base"
    output_dir: str = "outputs"
    max_steps: int = 100
    learning_rate: float = 1e-5
    action_chunk_size: int = 30
    use_latent_actions: bool = True
    latent_action_dim: int = 256
    num_view_slots: int = 6
    config_overrides: str | None = None
    processor_overrides: str | None = None
    lr_scheduler_kwargs: dict = field(default_factory=dict)
    frozen_parameters: list[str] | None = None
    deepspeed: str = "zero1.json"
    bf16: bool = True

    def __post_init__(self):
        raise AssertionError("Parsing must not construct the CUDA-initializing argument object")


class RecipeTests(unittest.TestCase):
    def test_formal_hyperparameters(self):
        first, second = recipe("stage1"), recipe("stage2")
        self.assertEqual(second, recipe("stage2_2b"))
        for config in (first, second):
            self.assertEqual(config["micro_batch_size"] * config["gradient_accumulation_steps"] * 8, 128)
            self.assertIsNone(config["frozen_parameters"])
            self.assertEqual(config["action_head_lr"], 1e-4)
            self.assertNotIn("action_decoder", config["action_head_modules"])
            self.assertIn("state_proj", config["action_head_modules"])
            self.assertTrue(config["use_ema"])
            model = config["config_overrides"]
            for key, value in {
                "expert_backbone": "qwen3_vl", "expert_hidden_size": 768,
                "expert_intermediate_size": 2752, "expert_num_foresight_tokens": 0,
                "expert_time_concat": True, "expert_per_layer_adanorm": True,
                "time_conditioning": "adaln", "num_view_slots": 6,
                "latent_action_dim": 608, "latent_action_chunk_size": 6,
                "latent_action_stride": 5, "use_view_cond_slots": True,
            }.items():
                self.assertEqual(model[key], value)
        self.assertEqual((first["max_steps"], first["action_chunk_size"], first["mm_max_length"]), (60000, 30, 64))
        self.assertEqual((first["learning_rate"], first["warmup_steps"], first["min_lr_rate"]), (1e-5, 5000, 0.05))
        self.assertEqual(first["lr_scheduler_type"], "cosine_with_min_lr")
        self.assertEqual(first["ema_decay"], 0.999)
        self.assertTrue(first["export_ema"])
        self.assertTrue(first["config_overrides"]["use_latent_actions"])
        self.assertFalse(first["processor_overrides"]["use_state"])
        self.assertEqual((second["max_steps"], second["learning_rate"], second["action_chunk_size"]), (30000, 2.5e-5, 10))
        self.assertEqual((second["lr_scheduler_type"], second["warmup_steps"], second["ema_decay"]), ("cosine", 3000, 0.99))
        self.assertFalse(second["config_overrides"]["use_latent_actions"])
        self.assertTrue(second["config_overrides"]["use_latent_head_readout"])
        self.assertEqual(second["processor_overrides"], {"use_state": True, "action_norm_type": "min_max_sym"})
        self.assertIsNone(second["trunk_model_path"])
        self.assertIsNone(second["reset_pretrained_modules"])
        self.assertEqual(json.loads(Path(first["deepspeed"]).read_text())["zero_optimization"]["stage"], 2)
        self.assertEqual(json.loads(Path(second["deepspeed"]).read_text())["zero_optimization"]["stage"], 1)

    def test_presets_use_existing_argument_fields(self):
        tree = ast.parse((ROOT / "rynnvla" / "arguments.py").read_text())
        names = {
            node.target.id for cls in tree.body if isinstance(cls, ast.ClassDef)
            for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        for stage in PRESETS:
            self.assertFalse(set(recipe(stage)) - names)

    def test_non_preset_recipes_parse_and_are_internally_consistent(self):
        """The recipes reachable only through --config get no launcher validation.

        A typo'd field name in one of these is not caught at submit time the way a preset's is:
        it surfaces as an unknown-key error after the GPUs are allocated, or not at all. These
        are also the only shipped recipes that exercise transfer_lr / transfer_modules /
        sampler_shuffle, so a regression in those argument names would otherwise be invisible
        until someone ran a RoboTwin arm.
        """
        tree = ast.parse((ROOT / "rynnvla" / "arguments.py").read_text())
        names = {
            node.target.id for cls in tree.body if isinstance(cls, ast.ClassDef)
            for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        configs = sorted((ROOT / "rynnvla" / "configs").glob("*.json"))
        recipes = [path for path in configs
                   if path.name not in set(PRESETS.values()) and not path.name.startswith("zero")]
        # Guard against this test silently narrowing to nothing if the recipes are renamed.
        self.assertGreaterEqual({p.name for p in recipes},
                                {"stage2_robotwin_2b.json", "stage2_robotwin_4b_2node16gpu.json"})
        for path in recipes:
            with self.subTest(recipe=path.name):
                values = json.loads(path.read_text())
                self.assertFalse(set(values) - names, f"{path.name} has unknown fields")
                transfer_lr = values.get("transfer_lr")
                transfer_modules = values.get("transfer_modules")
                # create_optimizer raises unless both are set and the two prefix lists are
                # disjoint, so an inconsistent recipe cannot start -- but it should not ship.
                self.assertEqual(transfer_lr is None, transfer_modules is None)
                if transfer_modules:
                    self.assertGreater(transfer_lr, 0)
                    self.assertFalse(set(transfer_modules) & set(values["action_head_modules"]))
                self.assertIn(values.get("sampler_shuffle", "auto"), ("auto", "global"))

    @unittest.skipUnless(HAS_TRANSFORMERS, "transformers not installed")
    def test_robotwin_recipes_satisfy_the_robotwin_eval_adapter(self):
        """Train and eval must agree on the action semantics, statically.

        scripts/robotwin_policy.py refuses a checkpoint whose config or processor does not match
        a fixed contract. Running the recipe through that very check means a recipe edit that
        would make a finished 16-GPU run unevaluable fails here instead of after the run.
        """
        from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig
        from rynnvla.models.rynn_brain_vla.processing_rynn_brain_vla import ACTION_DIM

        spec = importlib.util.spec_from_file_location(
            "_robotwin_policy_adapter", ROOT / "scripts" / "robotwin_policy.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)

        class Merged:
            """What api/train.py actually constructs, minus the model.

            ``action_dim`` is NOT a recipe field: train.py builds the config as
            ``processor.get_config_overrides()`` updated with the recipe's ``config_overrides``,
            and the processor is where the 81-dim interleaved rot6d layout comes from. Layering
            the same three sources here, and falling back attribute-by-attribute instead of
            naming the adapter's keys, keeps this honest when the adapter starts pinning
            something new.
            """

            def __init__(self, recipe, defaults):
                overrides = {"action_dim": ACTION_DIM}
                overrides.update(recipe.get("config_overrides") or {})
                self.__dict__["_overrides"] = overrides
                self.__dict__["_defaults"] = defaults

            def __getattr__(self, name):
                overrides = self.__dict__["_overrides"]
                if name in overrides:
                    return overrides[name]
                return getattr(self.__dict__["_defaults"], name)

        def leaf(dim, is_relative):
            entry = {"dim": dim, "is_relative": is_relative, "allow_relative": False}
            if dim == 6:
                entry["representation"] = "rot_6d"
            return entry

        # The dual-arm EEF schema the adapter demands: absolute state, relative action, rot6d
        # rotations, absolute grippers. What the dataset actually derives from the corpus is
        # checked in test_robotwin_stage2.py; this is the shape eval will accept.
        schema = {
            section: {"aloha_agilex": {
                f"{side}_{part}": value
                for side in ("left", "right")
                for part, value in (
                    ("arm", {"type": "eef", "eef_position": leaf(3, section == "action"),
                             "eef_rotation": leaf(6, section == "action")}),
                    ("gripper", leaf(1, False)),
                )
            }}
            for section in ("state", "action")
        }
        defaults = RynnBrainVLAConfig()
        # The adapter hardcodes 81; if the processor's layout constant ever moves, the adapter
        # has to move with it or every RoboTwin export starts failing at eval time.
        self.assertEqual(ACTION_DIM, 81)

        def load(name):
            return json.loads((ROOT / "rynnvla" / "configs" / name).read_text())

        for name in ("stage2_robotwin_2b.json", "stage2_robotwin_4b_2node16gpu.json"):
            with self.subTest(recipe=name):
                values = load(name)
                adapter.validate_checkpoint(
                    Merged(values, defaults),
                    SimpleNamespace(schema=schema, **values["processor_overrides"]))

        # And the LIBERO V5 preset must NOT pass: chunk 10 with min_max_sym is a different action
        # contract, and accepting it would evaluate a RoboTwin run with the wrong normalization
        # rather than failing.
        v5 = load("stage2_v5_4b.json")
        with self.assertRaises(ValueError):
            adapter.validate_checkpoint(
                Merged(v5, defaults),
                SimpleNamespace(schema=schema, **v5["processor_overrides"]))

    def test_relative_data_templates_and_corpus_weights(self):
        latent = json.loads((ROOT / "configs/data_latent_pretrain.example.json").read_text())
        libero = json.loads((ROOT / "configs/data_libero_joint40.example.json").read_text())
        robotwin = json.loads((ROOT / "configs/data_robotwin_mixed.example.json").read_text())
        vlabench = json.loads((ROOT / "configs/data_vlabench.example.json").read_text())
        self.assertEqual(len(libero), 4)
        self.assertAlmostEqual(sum(latent[0]["dataset_weights"].values()), 1.0)
        self.assertEqual(latent[0]["latent_chunk"], 6)
        # The RoboTwin index is user-built, so the template must not pin a checksum nobody can
        # reproduce; build_robotwin_index.py prints the one to put here.
        self.assertNotIn("index_sha256", robotwin[0])
        self.assertNotIn("schema_path", robotwin[0])
        for item in latent + libero + robotwin + vlabench:
            self.assertTrue(item["data_path"].startswith("data/"), item["data_path"])
            self.assertFalse(Path(item["data_path"]).is_absolute())
        for item in robotwin:
            self.assertTrue(item["index_cache"].startswith("data/"))

    @unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "torch/transformers not installed")
    def test_every_head_prefix_matches_real_meta_modules(self):
        import torch
        from rynnvla.api.export_checkpoint import expected_state_shapes

        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA is forbidden")):
            shapes = {stage: expected_state_shapes(meta_config(stage)) for stage in PRESETS}
        for stage, names in shapes.items():
            for prefix in recipe(stage)["action_head_modules"]:
                self.assertTrue(any(name == prefix or name.startswith(prefix + ".") for name in names), prefix)
            self.assertFalse(any(name.startswith("action_decoder.") for name in names))
        common = shapes["stage1"].keys() & shapes["stage2"].keys()
        self.assertTrue(all(shapes["stage1"][key] == shapes["stage2"][key] for key in common))
        self.assertEqual(shapes["stage2"].keys() - shapes["stage1"].keys(), {"latent_readout_proj.weight", "latent_readout_proj.bias"})
        self.assertEqual(shapes["stage1"].keys() - shapes["stage2"].keys(), {"latent_in_proj.weight", "latent_in_proj.bias"})
        common_2b = shapes["stage1_2b"].keys() & shapes["stage2_2b"].keys()
        self.assertTrue(all(shapes["stage1_2b"][key] == shapes["stage2_2b"][key] for key in common_2b))
        self.assertEqual(shapes["stage2_2b"].keys() - shapes["stage1_2b"].keys(), {"latent_readout_proj.weight", "latent_readout_proj.bias"})
        self.assertEqual(shapes["stage1_2b"].keys() - shapes["stage2_2b"].keys(), {"latent_in_proj.weight", "latent_in_proj.bias"})


class LaunchTests(unittest.TestCase):
    def options(self, stage="stage1"):
        return ["--stage", stage, "--model-path", "models/model with spaces", "--data-mixture", "data/mix.json", "--output-dir", "runs/new"]

    def test_single_rank_override_forwarding(self):
        command, dry = build_command(self.options() + ["--nproc-per-node", "1", "--max-steps", "2", "--dry-run"])
        self.assertTrue(dry)
        self.assertIn("--standalone", command)
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "1")
        self.assertEqual(command[-2:], ["--max_steps", "2"])
        self.assertIn("models/model with spaces", command)
        self.assertNotIn("--resume", command)
        self.assertTrue(Path(command[command.index("--config") + 1]).is_absolute())

    def test_2b_stage1_selects_its_recipe_and_preserves_training_settings(self):
        command, _ = build_command(self.options("stage1_2b") + ["--dry-run"])
        self.assertEqual(Path(command[command.index("--config") + 1]).name, "stage1_2b.json")
        self.assertEqual(recipe("stage1_2b"), recipe("stage1"))

    def test_2b_stage2_selects_matching_recipe(self):
        command, _ = build_command(self.options("stage2_2b") + ["--dry-run"])
        self.assertEqual(Path(command[command.index("--config") + 1]).name, "stage2_v5_2b.json")
        self.assertEqual(recipe("stage2_2b"), recipe("stage2"))

    def test_multinode_and_resume_are_explicit(self):
        command, _ = build_command(self.options("stage2") + ["--nnodes", "2", "--node-rank", "1", "--master-addr", "node0", "--master-port", "29512", "--resume"])
        self.assertNotIn("--standalone", command)
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--node_rank") + 1], "1")
        self.assertEqual(command[command.index("--master_addr") + 1], "node0")
        for flags in (["--nnodes", "2"], ["--node-rank", "1"], ["--nproc-per-node", "0"], ["--config", "other.json"]):
            with self.assertRaises(SystemExit):
                build_command(self.options() + flags)

    def test_dry_run_does_not_spawn_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            options = self.options() + ["--output-dir", str(output), "--dry-run"]
            with patch("subprocess.call", side_effect=AssertionError("must not launch")):
                self.assertEqual(main(options), 0)
            self.assertFalse(output.exists())

    def test_existing_output_refused_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            # main() catches (OSError, ValueError) around check_output_dir and FileExistsError is
            # an OSError subclass, so the launcher reports the refusal on stderr and returns 2
            # rather than propagating. The patched subprocess.call raising AssertionError still
            # proves nothing was spawned, because main does not catch AssertionError.
            stderr = io.StringIO()
            with patch("subprocess.call", side_effect=AssertionError("must not launch")):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(self.options() + ["--output-dir", directory]), 2)
            self.assertIn("Output already exists", stderr.getvalue())
            check_output_dir(directory, resume=True)
            with self.assertRaises(ValueError):
                check_output_dir(str(Path(directory) / "missing"), resume=True)


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers not installed")
class ParserTests(unittest.TestCase):
    def parse(self, config, cli):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(config))
            return parse_training_options(["--config", str(path)] + cli, argument_type=NoInitArguments)

    def test_json_then_cli_before_dataclass_initialization(self):
        values, resume = self.parse({"max_steps": 60000, "learning_rate": 1e-5, "bf16": True}, ["--max-steps", "2", "--learning-rate=0.002", "--bf16", "false", "--resume"])
        self.assertEqual(values["max_steps"], 2)
        self.assertEqual(values["learning_rate"], 0.002)
        self.assertFalse(values["bf16"])
        self.assertTrue(resume)

    def test_dictionary_merge_and_geometry_override(self):
        values, _ = self.parse({
            "config_overrides": {"action_chunk_size": 30, "expert_hidden_size": 768},
            "processor_overrides": {"use_state": False, "action_norm_type": "mean_std"},
        }, ["--action-chunk-size", "10", "--config-overrides", '{"expert_train_repeat": 1}', "--processor-overrides", '{"use_state": true}', "--lr-scheduler-kwargs", '{"min_lr_rate": 0.05}'])
        self.assertEqual(values["action_chunk_size"], 10)
        self.assertEqual(values["config_overrides"], {"action_chunk_size": 10, "expert_hidden_size": 768, "expert_train_repeat": 1})
        self.assertEqual(values["processor_overrides"], {"use_state": True, "action_norm_type": "mean_std"})
        self.assertEqual(values["lr_scheduler_kwargs"], {"min_lr_rate": 0.05})
        values, _ = self.parse({}, ["--config-overrides", '{"action_chunk_size": 10}'])
        self.assertEqual(values["action_chunk_size"], 10)

    def test_unknown_fields_and_conflicts_fail_without_construction(self):
        with self.assertRaisesRegex(ValueError, "Unknown TrainingArguments"):
            self.parse({"typo_steps": 2}, [])
        with self.assertRaises(SystemExit):
            self.parse({}, ["--typo-steps", "2"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.parse({}, ["--action-chunk-size", "10", "--config-overrides", '{"action_chunk_size": 20}'])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.parse({"action_chunk_size": 30, "config_overrides": {"action_chunk_size": 10}}, [])

    def test_explicit_list_is_not_filtered(self):
        values, _ = self.parse({}, ["--frozen-parameters", "visual", "language_model", "--config-overrides", "{}"])
        self.assertEqual(values["frozen_parameters"], ["visual", "language_model"])


class ValidationTests(unittest.TestCase):
    def test_resume_compares_mixture_contents_with_training_snapshot(self):
        from rynnvla.constants import RESUME_ARGS_NAME

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-2"
            checkpoint.mkdir()
            for name in ("config.json", "processor_config.json"):
                (checkpoint / name).write_text("{}")
            mixture = [
                {"data_type": "LatentPretrainDataset", "data_path": "data/index.npz", "weight": 0.7},
                {"data_type": "LatentPretrainDataset", "data_path": "data/other.npz", "weight": 0.3},
            ]
            (checkpoint / RESUME_ARGS_NAME).write_text(json.dumps({"data_mixture": mixture, "max_steps": 4}))
            path = root / "mixture.json"
            path.write_text(json.dumps(mixture))
            selector = lambda _: str(checkpoint)

            # The same JSON file passed on the command line must match Trainer's
            # expanded list, including when a copy is used or the list is supplied.
            for name in ("mixture.json", "mixture-copy.json"):
                copy = root / name
                copy.write_text(json.dumps(mixture, sort_keys=True))
                values = {"output_dir": directory, "data_mixture": str(copy), "max_steps": 4}
                self.assertEqual(resolve_resume(values, True, selector), str(checkpoint))
                self.assertEqual(values["data_mixture"], mixture)
            values = {"output_dir": directory, "data_mixture": mixture, "max_steps": 4}
            self.assertEqual(resolve_resume(values, True, selector), str(checkpoint))
            self.assertIsNot(values["data_mixture"], mixture)

            for changed in (
                [{**mixture[0], "weight": 0.5}, mixture[1]],
                [{**mixture[0], "data_path": "different.npz"}, mixture[1]],
                list(reversed(mixture)),
            ):
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "cannot change data_mixture"):
                    resolve_resume({"output_dir": directory, "data_mixture": str(path), "max_steps": 4},
                                   True, selector)
            path.write_text(json.dumps(mixture))
            with self.assertRaisesRegex(ValueError, "cannot change max_steps"):
                resolve_resume({"output_dir": directory, "data_mixture": str(path), "max_steps": 8},
                               True, selector)

    def test_resume_cannot_initialize_a_different_stage(self):
        config = {**recipe("stage1"), "output_dir": "runs/stage1"}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-17"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps(config["config_overrides"]))
            (checkpoint / "processor_config.json").write_text(json.dumps(config["processor_overrides"]))
            selector = lambda _: str(checkpoint)
            self.assertEqual(resolve_resume(config, True, selector), str(checkpoint))
            self.assertIsNone(resolve_resume(config, False, lambda _: self.fail("fresh runs do not search checkpoints")))
            with self.assertRaisesRegex(ValueError, "no complete"):
                resolve_resume(config, True, lambda _: None)
            with self.assertRaisesRegex(ValueError, "cannot change"):
                resolve_resume({**recipe("stage2"), "output_dir": directory}, True, selector)

    def test_save_interval_and_scheduler_checks(self):
        config = {**recipe("stage1"), "model_path": "models/base", "data_mixture": "data/mix.json", "output_dir": "runs/new"}
        validate_options(config)
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_options({**config, "save_keep_every": 5000})
        with self.assertRaisesRegex(ValueError, "cosine_with_min_lr"):
            validate_options({**config, "lr_scheduler_type": "cosine"})


if __name__ == "__main__":
    unittest.main()
