"""VLABench inference-adapter conventions (model-free).

The full predict_vlabench (model decode) needs transformers-5.2 + an exported 2B checkpoint + the
VLABench simulator, none of which run here. These tests pin the adapter MATH that is shared by the
eval server and the training dataset, so a convention drift (gripper sign, rotation representation,
camera->role map, absolute-vs-delta) fails fast and model-free:
  - camera->role map equals the train-time VLABenchDataset._CAMERA_SLOTS (no train/eval skew)
  - state gripper inversion (raw 1=closed -> framework 1=open), rotation stored as interleaved rot_6d
  - action rot_6d -> euler_xyz round-trip, (T,7) absolute layout, NO gripper sign flip (unlike LIBERO)
"""

import numpy as np
import pytest
import torch

from rynnvla.constants import RotationRepresentation
from rynnvla.datasets.vla_datasets.vlabench import VLABenchDataset
from rynnvla.inference_wrappers.rynn_brain_vla import (
    VLABENCH_CAMERA_SLOT_MAP,
    robot_action_to_vlabench_actions,
    vlabench_images,
    vlabench_state_to_robot_state,
)
from rynnvla.utils.robot import Arm, Position, RobotAction, Rotation


def test_camera_map_matches_training():
    assert VLABENCH_CAMERA_SLOT_MAP == VLABenchDataset._CAMERA_SLOTS
    assert VLABENCH_CAMERA_SLOT_MAP == {"image": 3, "second_image": 4, "wrist_image": 1}


def test_state_gripper_inversion_and_rotation_repr():
    # raw flag 1 = CLOSED (upstream get_ee_open_state bug) -> framework 0; raw 0 = open -> 1
    closed = vlabench_state_to_robot_state(np.array([.1, .2, .3, 0., 0., 0., 1.0], dtype=np.float32))
    assert float(closed.left_gripper.data) == 0.0
    opened = vlabench_state_to_robot_state(np.array([.1, .2, .3, 0., 0., 0., 0.0], dtype=np.float32))
    assert float(opened.left_gripper.data) == 1.0
    # canonical interleaved rot_6d (6), position 3, gripper non-relative
    assert closed.left_arm.eef_rotation.representation == RotationRepresentation.ROT_6D
    assert closed.left_arm.eef_rotation.data.shape[-1] == 6
    assert closed.left_arm.eef_position.data.shape[-1] == 3
    assert closed.left_gripper.allow_relative is False


def test_action_roundtrip_euler_and_no_gripper_flip():
    euler = np.array([0.3, -0.5, 1.2], dtype=np.float32)
    pos = np.array([0.4, -0.1, 0.55], dtype=np.float32)
    grip = np.array([1.0], dtype=np.float32)  # 1 = open
    rot6d = Rotation(torch.from_numpy(euler.copy()),
                     RotationRepresentation.EULER_XYZ).convert_rotation(RotationRepresentation.ROT_6D)
    action = RobotAction(
        left_arm=Arm(eef_position=Position(torch.from_numpy(pos.copy())), eef_rotation=rot6d),
        left_gripper=Position(torch.from_numpy(grip.copy()), allow_relative=False),
    )
    out = robot_action_to_vlabench_actions(action)
    assert out.shape == (1, 7)
    assert np.allclose(out[0, :3], pos, atol=1e-5)
    assert np.allclose(out[0, 3:6], euler, atol=1e-4)  # rot_6d -> euler_xyz recovers the euler
    assert float(out[0, 6]) == 1.0                      # NO 1-2x flip (LIBERO would emit -1.0)


def test_action_rejects_relative_fields():
    rot6d = Rotation(torch.zeros(1, 6), RotationRepresentation.ROT_6D)
    rel = RobotAction(
        left_arm=Arm(eef_position=Position(torch.zeros(1, 3), is_relative=True), eef_rotation=rot6d),
        left_gripper=Position(torch.zeros(1, 1), allow_relative=False),
    )
    with pytest.raises(ValueError, match="absolute"):
        robot_action_to_vlabench_actions(rel)


def test_images_validate_three_cameras():
    imgs = {k: np.zeros((224, 224, 3), dtype=np.uint8) for k in ("image", "second_image", "wrist_image")}
    assert set(vlabench_images(imgs)) == {"image", "second_image", "wrist_image"}
    with pytest.raises(KeyError):
        vlabench_images({"image": imgs["image"], "second_image": imgs["second_image"]})
