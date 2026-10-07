"""CPU contract tests. Simulator doubles below are NOT real LIBERO rollouts."""

import builtins
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation
import torch

from rynnvla.api.eval_libero import LIBERO_DUMMY_ACTION, load_libero, run_episode
from rynnvla.api.predict import load_sample
from rynnvla.constants import RobotType, RotationRepresentation, VIEW_ROLE_TO_ID
from rynnvla.inference_wrappers.rynn_brain_vla import (
    LIBERO_CAMERA_SLOT_MAP,
    RynnBrainVLAInferenceWrapper,
    libero_actions_to_robot_action,
    libero_dataset_actions_to_commands,
    libero_images,
    libero_observation_to_sample,
    libero_state_to_robot_state,
    robot_action_to_libero_actions,
    validate_libero_schema,
)


def _schema():
    def leaf(dim, rotation=False):
        data = {"type": "Rotation" if rotation else "Position", "dim": dim,
                "is_relative": False, "allow_relative": dim != 1,
                "mean": [0.2] * dim, "std": [0.7] * dim,
                "min": [-1.0] * dim, "max": [1.0] * dim,
                "q01": [-0.9] * dim, "q99": [0.8] * dim}
        if rotation:
            data["representation"] = "rot_6d"
        return data

    robot = {"type": "RobotAction", "left_arm": {"type": "Arm",
             "eef_position": leaf(3), "eef_rotation": leaf(6, True)}, "left_gripper": leaf(1)}
    return {"action": {"franka": copy.deepcopy(robot)}, "state": {"franka": copy.deepcopy(robot)}}


def _observation(step=0):
    return {
        "robot0_eef_pos": np.array([0.2 + step, -0.3, 1.4], dtype=np.float32),
        "robot0_eef_quat": ScipyRotation.from_rotvec([0.2, -0.4, 0.1]).as_quat(),
        "robot0_gripper_qpos": np.array([0.027, -0.026], dtype=np.float32),
        "agentview_image": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        "robot0_eye_in_hand_image": np.arange(36, 72, dtype=np.uint8).reshape(3, 4, 3),
    }


def test_adapter_import_needs_no_model_or_simulator():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'libero', 'sapien', 'ray', 'transformers'}:
            raise AssertionError('unwanted dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from rynnvla.inference_wrappers.rynn_brain_vla import libero_actions_to_robot_action
libero_actions_to_robot_action([[0, 0, 0, 0, 0, 0, -1]])
assert not any(name.startswith('rynnvla.models') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_raw_action_roundtrip_interleaved_and_no_command_rescaling():
    actions = np.array([[1.2, -0.4, 0.03, 0.4, -0.8, 0.25, -0.37],
                        [-0.9, 0.2, -0.1, 0, 0, 0, 0.65]], dtype=np.float32)
    robot = libero_actions_to_robot_action(actions)
    expected = ScipyRotation.from_rotvec(actions[:, 3:6]).as_matrix()[:, :, :2].reshape(-1, 6)
    np.testing.assert_allclose(robot.left_arm.eef_rotation.data, expected, atol=1e-7)
    np.testing.assert_allclose(robot_action_to_libero_actions(robot), actions, atol=1e-6)
    np.testing.assert_allclose(robot_action_to_libero_actions(robot.to_dict()), actions, atol=1e-6)


def test_dataset_gripper_is_mapped_to_robosuite_command():
    actions = np.array([
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0],
        [-0.7, 0.8, -0.9, 0.2, -0.3, 0.4, 0.25],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1.0],
    ], dtype=np.float32)
    commands = libero_dataset_actions_to_commands(actions)
    np.testing.assert_array_equal(commands[:, :6], actions[:, :6])
    np.testing.assert_allclose(commands[:, 6], [1.0, 0.5, -1.0])
    np.testing.assert_array_equal(actions[:, 6], [0.0, 0.25, 1.0])


@pytest.mark.parametrize("norm", ["mean_std", "min_max", "min_max_sym", "q01_q99"])
def test_real_processor_81d_roundtrip(norm):
    # No model weights/tokenizer/simulator are needed. Model-stack imports must work
    # in a fully installed checkout; minimal CPU-only adapter environments may skip.
    module = pytest.importorskip(
        "rynnvla.models.rynn_brain_vla.processing_rynn_brain_vla",
        reason="Real processor requires the installed model dependency stack", exc_type=ImportError,
    )
    processor = object.__new__(module.RynnBrainVLAProcessor)
    processor.schema = _schema()
    processor.action_norm_type = norm
    actions = np.array([[0.9, -0.1, 0.3, 0.5, -0.2, 0.7, -0.6]], dtype=np.float32)
    robot = libero_actions_to_robot_action(actions)
    tensor, mask = processor._process_action(robot, processor.schema["action"]["franka"])
    assert tensor.shape == mask.shape == (1, 81)
    assert mask.sum() == 10
    decoded = processor.post_process(tensor, RobotType.FRANKA, state=None)
    np.testing.assert_allclose(robot_action_to_libero_actions(decoded), actions, atol=1e-6)


def test_roles_and_raw_images_match_client_not_vertical_only():
    assert LIBERO_CAMERA_SLOT_MAP == {
        "front": VIEW_ROLE_TO_ID["front_third"], "wrist": VIEW_ROLE_TO_ID["left_wrist"],
    }
    obs = _observation()
    sample = libero_observation_to_sample(obs)
    for name, key in (("front", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
        np.testing.assert_array_equal(sample["images"][name], obs[key][::-1, ::-1])
        assert not np.array_equal(sample["images"][name], obs[key][::-1])
        assert sample["images"][name].flags.c_contiguous
    # Dataset-oriented inputs are not flipped a second time.
    unchanged = libero_images(sample["images"])
    np.testing.assert_array_equal(unchanged["front"], sample["images"]["front"])


def test_world_eef_state_keeps_raw_first_qpos():
    obs = _observation()
    sample = libero_observation_to_sample(obs)
    np.testing.assert_allclose(sample["state"][3:6], [0.2, -0.4, 0.1], atol=1e-7)
    robot = libero_state_to_robot_state(sample["state"])
    np.testing.assert_array_equal(robot.left_arm.eef_position.data[0], obs["robot0_eef_pos"])
    np.testing.assert_array_equal(robot.left_gripper.data[0], obs["robot0_gripper_qpos"][:1])
    assert robot.left_arm.joint_position is None
    assert robot.left_gripper.allow_relative is False
    assert robot.left_arm.eef_rotation.representation == RotationRepresentation.ROT_6D


def test_quaternion_sign_matches_client_axisangle_convention():
    obs = _observation()
    obs["robot0_eef_quat"] *= -1
    quat = obs["robot0_eef_quat"]
    expected = quat[:3] * 2 * np.arccos(quat[3]) / np.sqrt(1 - quat[3] ** 2)
    np.testing.assert_allclose(libero_observation_to_sample(obs)["state"][3:6], expected, atol=1e-6)


@pytest.mark.parametrize("state", [np.zeros(6), np.zeros((1, 8)), [0, 0, 0, 0, 0, np.nan, 0]])
def test_invalid_states_rejected(state):
    with pytest.raises(ValueError):
        libero_state_to_robot_state(state)


def test_relative_schema_is_rejected_not_silently_overridden():
    processor = SimpleNamespace(schema=_schema())
    config = SimpleNamespace(action_dim=81, action_chunk_size=10)
    validate_libero_schema(processor, config)
    processor.schema["action"]["franka"]["left_arm"]["eef_position"]["is_relative"] = True
    with pytest.raises(ValueError, match="non-relative"):
        validate_libero_schema(processor, config)
    assert processor.schema["action"]["franka"]["left_arm"]["eef_position"]["is_relative"] is True


def test_legacy_rot6d_is_not_guessed():
    with pytest.raises(ValueError, match="interleaved"):
        validate_libero_schema(SimpleNamespace(schema=_schema(), rot6d_layout="legacy"),
                              SimpleNamespace(action_dim=81, action_chunk_size=10))


def test_stage1_latent_checkpoint_is_rejected_without_mutation():
    config = SimpleNamespace(action_dim=81, action_chunk_size=10, use_latent_actions=True)
    with pytest.raises(ValueError, match="Stage2 action checkpoint"):
        validate_libero_schema(SimpleNamespace(schema=_schema()), config)
    assert config.use_latent_actions is True


def test_hf_loading_is_local_and_does_not_override_trained_flags(monkeypatch):
    from types import ModuleType

    calls = []
    model = SimpleNamespace(eval=lambda: calls.append("eval") or model)
    processor = object()
    module = ModuleType("rynnvla.models.rynn_brain_vla")
    module.RynnBrainVLAModel = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)) or model,
    )
    module.RynnBrainVLAProcessor = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)) or processor,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    wrapper = RynnBrainVLAInferenceWrapper("export", torch.float32, "sdpa", device="cpu")
    assert wrapper.model is model and wrapper.processor is processor
    assert calls == [
        ("export", {"dtype": torch.float32, "attn_implementation": "sdpa",
                    "device_map": {"": "cpu"}, "local_files_only": True}),
        "eval", ("export", {"local_files_only": True}),
    ]


def test_json_sample_relative_paths_and_orientation_without_disk_writes():
    from PIL import Image

    pixels = _observation()["agentview_image"]
    payload = {"instruction": "pick up the cup", "state": [0, 0, 1, 0, 0, 0, 0.02, -0.02],
               "front": "front.png", "wrist": "wrist.png", "image_convention": "libero_raw"}
    with patch.object(Path, "open", return_value=io.StringIO(json.dumps(payload))), \
            patch.object(Image, "open", side_effect=lambda path: Image.fromarray(pixels)) as image_open:
        result = load_sample(Path("/virtual/sample.json"))
    assert image_open.call_args_list[0].args[0] == Path("/virtual/front.png")
    np.testing.assert_array_equal(result["images"]["front"], pixels[::-1, ::-1])


def test_prediction_reprefills_and_never_adds_current_state(monkeypatch):
    commands = np.array([[0.25, -0.1, 0, 0.1, 0.2, -0.3, -0.35]], dtype=np.float32)
    seen = []

    class Processor:
        schema = _schema()

        def post_process(self, action, robot_type, state):
            assert action.shape == (1, 81)
            assert robot_type == RobotType.FRANKA and state is None
            return libero_actions_to_robot_action(commands)

    wrapper = RynnBrainVLAInferenceWrapper("unused", torch.float32, "sdpa", device="cpu")
    wrapper._model = SimpleNamespace(config=SimpleNamespace(action_dim=81, action_chunk_size=1))
    wrapper._processor = Processor()
    monkeypatch.setattr(wrapper, "process", lambda **kwargs: seen.append(kwargs) or {})
    monkeypatch.setattr(wrapper, "collate", lambda batch: {"states": torch.zeros(1, 1, 81)})
    prefixes = []
    monkeypatch.setattr(wrapper, "prefill", lambda inputs: prefixes.append(object()) or prefixes[-1])
    monkeypatch.setattr(wrapper, "decode", lambda *args, **kwargs: torch.zeros(1, 1, 81))
    sample = libero_observation_to_sample(_observation())
    for _ in range(2):
        result = wrapper.predict_libero("pick cup", **sample)
        expected = commands.copy()
        expected[:, 6] = 1.0 - 2.0 * expected[:, 6]
        np.testing.assert_allclose(result, expected, atol=1e-6)
    assert len(prefixes) == 2 and prefixes[0] is not prefixes[1]
    assert seen[0]["camera_slot_map"] == LIBERO_CAMERA_SLOT_MAP
    assert seen[0]["robot_type"] == "franka"


class FakeEnv:
    """Deterministic bookkeeping double, not a physics simulator."""

    def __init__(self, stop_at=None, success=False, truncated=False, five_tuple=False):
        self.stop_at, self.succeed, self.truncate = stop_at, success, truncated
        self.five_tuple = five_tuple
        self.commands = []
        self.closed = False

    def seed(self, value):
        self.seed_value = value

    def reset(self):
        return _observation()

    def set_init_state(self, value):
        self.initial_state = value
        return _observation()

    def check_success(self):
        return self.succeed and self.stop_at is not None and len(self.commands) >= self.stop_at

    def step(self, command):
        self.commands.append(command)
        stopped = self.stop_at is not None and len(self.commands) >= self.stop_at
        obs = _observation(len(self.commands))
        if self.five_tuple:
            return obs, 0, stopped and not self.truncate, stopped and self.truncate, {}
        return obs, 0, stopped, {"TimeLimit.truncated": stopped and self.truncate}

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self):
        self.states = []

    def predict_libero(self, text, images, state, denoising_steps):
        self.states.append(state.copy())
        return np.tile([0.25, -0.1, 0, 0.1, 0.2, -0.3, -0.35], (4, 1))


def test_replan_budget_and_commands_are_exact():
    env, policy = FakeEnv(), FakePolicy()
    record = run_episode(lambda: env, policy, "initial", "pick cup", max_steps=5,
                         replan_steps=2, warmup_steps=0)
    assert env.closed and env.seed_value == 7
    assert record["stop_reason"] == "max_steps" and record["steps"] == 5
    assert record["success"] is False
    assert len(policy.states) == 3
    np.testing.assert_allclose([state[0] for state in policy.states], [0.2, 2.2, 4.2])
    np.testing.assert_allclose(env.commands, np.tile([0.25, -0.1, 0, 0.1, 0.2, -0.3, -0.35], (5, 1)))


@pytest.mark.parametrize("success,truncated,five_tuple,reason", [
    (True, False, False, "success"), (False, False, False, "terminated"),
    (False, True, False, "truncated"), (False, True, True, "truncated"),
    (True, False, True, "success"),
])
def test_terminal_step_stops_mid_chunk(success, truncated, five_tuple, reason):
    env = FakeEnv(stop_at=1, success=success, truncated=truncated, five_tuple=five_tuple)
    record = run_episode(lambda: env, FakePolicy(), None, "pick cup", max_steps=9, warmup_steps=0)
    assert env.closed and len(env.commands) == 1
    assert record["stop_reason"] == reason and record["success"] == success


def test_warmup_termination_never_queries_policy():
    env, policy = FakeEnv(stop_at=1), FakePolicy()
    record = run_episode(lambda: env, policy, None, "pick cup", warmup_steps=10)
    assert env.closed and policy.states == []
    assert env.commands == [LIBERO_DUMMY_ACTION]
    assert record["steps"] == 0 and record["warmup_steps"] == 1
    assert record["success"] is False


def test_success_without_done_stops_immediately():
    env = FakeEnv()
    env.check_success = lambda: len(env.commands) == 1
    record = run_episode(lambda: env, FakePolicy(), None, "pick cup", warmup_steps=0)
    assert env.closed and len(env.commands) == 1
    assert record["success"] and not record["terminated"]


@pytest.mark.parametrize("phase", ["seed", "reset", "set_init_state", "step", "policy", "frames"])
def test_environment_closes_on_errors(phase, monkeypatch):
    env, policy = FakeEnv(), FakePolicy()

    def fail(*args, **kwargs):
        raise RuntimeError("injected error")

    callback = None
    if phase == "policy":
        monkeypatch.setattr(policy, "predict_libero", fail)
    elif phase == "frames":
        callback = fail
    else:
        monkeypatch.setattr(env, phase, fail)
    with pytest.raises(RuntimeError, match="injected error"):
        run_episode(lambda: env, policy, None, "pick cup", warmup_steps=0, frame_callback=callback)
    assert env.closed


@pytest.mark.parametrize("chunk", [np.zeros((0, 7)), np.full((1, 7), np.nan), np.zeros((1, 8))])
def test_invalid_model_output_is_not_executed(chunk):
    env = FakeEnv()
    policy = SimpleNamespace(predict_libero=lambda **kwargs: chunk)
    with pytest.raises(ValueError, match="raw OSC"):
        run_episode(lambda: env, policy, None, "pick cup", warmup_steps=0)
    assert env.closed and env.commands == []


def test_missing_libero_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def no_libero(name, *args, **kwargs):
        if name.startswith("libero"):
            raise ModuleNotFoundError("libero is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_libero)
    with pytest.raises(RuntimeError, match="optional upstream LIBERO"):
        load_libero()
