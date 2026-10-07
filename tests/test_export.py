"""CPU-only export I/O tests with tiny tensors; real meta shapes tested in recipes."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rynnvla.api import export_checkpoint as exporter


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def stage1_config():
    preset = json.loads((ROOT / "rynnvla/configs/stage1_4b.json").read_text())
    return {**preset["config_overrides"], "model_type": "rynn_brain_vla", "text_config": {"hidden_size": 2560}}


class ConfigValidationTests(unittest.TestCase):
    def test_stage1_contract_not_training_step(self):
        exporter.validate_config(stage1_config(), {"use_state": False})
        for key, value in {
            "expert_hidden_size": 1024, "expert_intermediate_size": 4096,
            "expert_backbone": "custom", "expert_time_concat": False,
            "expert_num_foresight_tokens": 50, "num_view_slots": 5,
            "latent_action_dim": 128, "latent_action_chunk_size": 5,
            "latent_action_stride": 1, "use_view_cond_slots": False,
            "use_latent_head_readout": True, "use_sf_align": True,
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                exporter.validate_config({**stage1_config(), key: value}, {"use_state": False})
        with self.assertRaises(ValueError):
            exporter.validate_config(stage1_config(), {"use_state": True})
        with self.assertRaisesRegex(ValueError, "4B"):
            exporter.validate_config({**stage1_config(), "text_config": {"hidden_size": 2048}}, {"use_state": False})

    def test_stage2_contract(self):
        preset = json.loads((ROOT / "rynnvla/configs/stage2_v5_4b.json").read_text())
        config = {**stage1_config(), **preset["config_overrides"]}
        exporter.validate_config(config, preset["processor_overrides"], stage="stage2")
        with self.assertRaises(ValueError):
            exporter.validate_config(config, {"use_state": False,
                                              "action_norm_type": "min_max_sym"}, stage="stage2")

    def test_stage2_accepts_exactly_the_norm_types_the_processor_can_round_trip(self):
        """The exporter must not be stricter than the processor, or a trained arm cannot be
        evaluated; nor looser, or it exports a checkpoint whose normalization nobody can read
        back.

        Both sides validate against ``constants.ALLOWED_ACTION_NORM_TYPES``, so the tie is
        structural rather than something this test has to re-check. What it does pin is that
        the exporter accepts the whole set -- including a type that is not the formal recipe's
        -- and rejects everything else, rather than quietly narrowing to ``min_max_sym``.
        """
        from rynnvla.constants import ALLOWED_ACTION_NORM_TYPES

        preset = json.loads((ROOT / "rynnvla/configs/stage2_v5_4b.json").read_text())
        config = {**stage1_config(), **preset["config_overrides"]}
        for norm_type in ALLOWED_ACTION_NORM_TYPES:
            with self.subTest(norm_type=norm_type):
                exporter.validate_config(config, {"use_state": True,
                                                  "action_norm_type": norm_type}, stage="stage2")
        for norm_type in ("", None, "minmax", "q01_q99_sym", "MIN_MAX_SYM"):
            with self.subTest(norm_type=norm_type), self.assertRaises(ValueError):
                exporter.validate_config(config, {"use_state": True,
                                                  "action_norm_type": norm_type}, stage="stage2")

    def test_2b_stage2_contract(self):
        preset = json.loads((ROOT / "rynnvla/configs/stage2_v5_2b.json").read_text())
        config = {**stage1_config(), **preset["config_overrides"], "text_config": {"hidden_size": 2048}}
        exporter.validate_config(config, preset["processor_overrides"], stage="stage2_2b")
        with self.assertRaisesRegex(ValueError, "2B"):
            exporter.validate_config({**config, "text_config": {"hidden_size": 2560}},
                                     preset["processor_overrides"], stage="stage2_2b")

    def test_2b_stage1_export_requires_explicit_matching_backbone(self):
        config = {**stage1_config(), "text_config": {"hidden_size": 2048}}
        exporter.validate_config(config, {"use_state": False}, stage="stage1_2b")
        with self.assertRaisesRegex(ValueError, "2B"):
            exporter.validate_config(stage1_config(), {"use_state": False}, stage="stage1_2b")
        with self.assertRaisesRegex(ValueError, "4B"):
            exporter.validate_config(config, {"use_state": False}, stage="stage1")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ExportTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # An arbitrary step is intentional: no hard-coded step or byte-size contract.
        self.source = self.root / "checkpoint-17"
        self.source.mkdir()
        self.output = self.root / "export"
        self.metadata = {
            "config.json": stage1_config(),
            "processor_config.json": {"use_state": False, "image_processor": {"size": 64}},
            "tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"},
            "tokenizer.json": {"version": "1.0"},
            "preprocessor_config.json": {"image_processor_type": "Qwen3VLImageProcessor"},
            "special_tokens_map.json": {"eos_token": "<|im_end|>"},
        }
        for name, value in self.metadata.items():
            (self.source / name).write_text(json.dumps(value))
        (self.source / "chat_template.jinja").write_text("{{ messages }}")
        (self.source / "trainer_state.json").write_text('{"global_step": 17}')
        (self.source / "optimizer.pt").write_bytes(b"not copied")
        (self.source / "train.log").write_text("not copied")
        self.state = {
            "visual.weight": torch.ones(2, 2),
            "latent_action_head.0.weight": torch.full((2, 2), 3.0),
            "slot_seed_proj.weight": torch.full((2, 2), 7.0),
        }
        self.expected = {key: tuple(value.shape) for key, value in self.state.items()}
        torch.save(self.state, self.source / "ema_model.bin")
        # Small I/O fixture replaces only the expensive real architecture's shape map.
        # The production exporter always derives every key/shape from the meta model.
        self.shape_patch = patch.object(exporter, "expected_state_shapes", return_value=self.expected)
        self.shape_patch.start()
        self.addCleanup(self.shape_patch.stop)
        self.cuda_patch = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA is forbidden"))
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)

    def test_ema_to_pytorch_and_only_loadable_sidecars(self):
        load = self.torch.load
        with patch.object(self.torch, "load", wraps=load) as spy:
            result = exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
            self.assertEqual(spy.call_args.kwargs["weights_only"], True)
            self.assertEqual(spy.call_args.kwargs["map_location"], "cpu")
        self.assertEqual(result, self.output)
        exported = load(self.output / "pytorch_model.bin", weights_only=True)
        self.assertEqual(exported.keys(), self.state.keys())
        for key in self.state:
            self.assertTrue(self.torch.equal(exported[key], self.state[key]))
        self.assertEqual({file.name for file in self.output.iterdir()}, set(self.metadata) | {"chat_template.jinja", "pytorch_model.bin"})
        self.assertTrue((self.source / "optimizer.pt").exists())

    @unittest.skipUnless(HAS_SAFETENSORS, "safetensors not installed")
    def test_safetensors_output_and_live_input(self):
        from safetensors.torch import load_file, save_file
        live = {key: value + 1 for key, value in self.state.items()}
        save_file(live, str(self.source / "model.safetensors"))
        exporter.export_checkpoint(self.source, self.output, weights="model")
        saved = load_file(str(self.output / "model.safetensors"))
        for key in live:
            self.assertTrue(self.torch.equal(saved[key], live[key]))

    def test_live_model_bin_and_pytorch_input(self):
        for name in ("model.bin", "pytorch_model.bin"):
            with self.subTest(name=name):
                source = self.source / name
                self.torch.save(self.state, source)
                output = self.root / name.replace(".", "_")
                exporter.export_checkpoint(self.source, output, weights="model", output_format="pytorch")
                self.assertTrue((output / "pytorch_model.bin").is_file())
                source.unlink()

    def test_pytorch_shards(self):
        names = list(self.state)
        first, second = "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"
        self.torch.save({names[0]: self.state[names[0]]}, self.source / first)
        self.torch.save({key: self.state[key] for key in names[1:]}, self.source / second)
        index = {"weight_map": {key: first if key == names[0] else second for key in names}}
        (self.source / "pytorch_model.bin.index.json").write_text(json.dumps(index))
        exporter.export_checkpoint(self.source, self.output, weights="model", output_format="pytorch")
        self.assertTrue((self.output / "pytorch_model.bin").is_file())

    def test_shard_path_traversal_and_index_mismatch_rejected(self):
        path = self.source / "pytorch_model.bin.index.json"
        path.write_text(json.dumps({"weight_map": {"key": "../outside.bin"}}))
        with self.assertRaisesRegex(ValueError, "shard filename"):
            exporter.load_weights(self.source, "model")
        self.torch.save(self.state, self.source / "shard.bin")
        path.write_text(json.dumps({"weight_map": {"missing_key": "shard.bin"}}))
        with self.assertRaisesRegex(ValueError, "weight_map"):
            exporter.load_weights(self.source, "model")

    def test_missing_ema_does_not_fall_back(self):
        (self.source / "ema_model.bin").rename(self.source / "pytorch_model.bin")
        with self.assertRaisesRegex(ValueError, "ema_model.bin"):
            exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
        self.assertFalse(self.output.exists())

    def test_reject_existing_directory_and_preserve_contents(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
        self.assertEqual(marker.read_text(), "original")

    def test_missing_unexpected_and_misshaped_tensors_rejected(self):
        key = next(iter(self.state))
        bad_states = [
            {k: value for k, value in self.state.items() if k != key},
            {**self.state, "unexpected.weight": self.torch.ones(1)},
            {**self.state, key: self.torch.ones(3, 2)},
        ]
        for state in bad_states:
            self.torch.save(state, self.source / "ema_model.bin")
            with self.assertRaisesRegex(ValueError, "Incomplete/incompatible"):
                exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
            self.assertFalse(self.output.exists())

    def test_metadata_required_and_partial_export_cleaned(self):
        (self.source / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
        (self.source / "tokenizer.json").write_text("{}")
        with patch.object(self.torch, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
        self.assertFalse(self.output.exists())

    def test_non_tensor_pickle_rejected(self):
        self.torch.save({"optimizer": {"lr": 1e-5}}, self.source / "ema_model.bin")
        with self.assertRaisesRegex(ValueError, "Non-tensor"):
            exporter.export_checkpoint(self.source, self.output, output_format="pytorch")

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "transformers not installed")
    def test_export_with_real_meta_shape_validation(self):
        self.shape_patch.stop()
        config = stage1_config()
        config["action_dim"] = 81
        config["text_config"] = {
            "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 1,
            "num_attention_heads": 20, "num_key_value_heads": 4, "head_dim": 128,
            "vocab_size": 32, "max_position_embeddings": 4096,
        }
        config["vision_config"] = {
            "hidden_size": 32, "intermediate_size": 64, "depth": 1, "num_heads": 4,
            "out_hidden_size": 2560, "patch_size": 16, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "deepstack_visual_indexes": [0], "num_position_embeddings": 16,
        }
        (self.source / "config.json").write_text(json.dumps(config))
        expected = exporter.expected_state_shapes(config)
        # CPU views share one scalar's storage: real complete key/shape coverage
        # without allocating full-sized 4B-width parameters or running a forward.
        scalar = self.torch.zeros(())
        state = {key: scalar.expand(shape) for key, shape in expected.items()}
        self.torch.save(state, self.source / "ema_model.bin")
        exporter.export_checkpoint(self.source, self.output, output_format="pytorch")
        saved = self.torch.load(self.output / "pytorch_model.bin", weights_only=True)
        self.assertEqual({key: tuple(value.shape) for key, value in saved.items()}, expected)


if __name__ == "__main__":
    unittest.main()
