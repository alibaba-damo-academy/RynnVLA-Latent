"""LIBERO-Plus LeRobot dataset adapter for RynnVLA training.

Loads LIBERO-Plus demonstration data in LeRobot v2.1 format (parquet + mp4)
and converts to RynnVLA's RobotAction/RobotState format.

LIBERO uses a single-arm Panda robot:
  - action: 7-dim (3 pos delta + 3 axis-angle delta + 1 gripper)
  - observation.state: 8-dim, one of two conventions (auto-detected from
    meta/info.json state names):
      * joint: 7 joint angles + 1 gripper
      * eef (openpi-style): eef xyz + axis-angle(3) + 2 gripper qpos
  - cameras: front (256x256), wrist (256x256)
"""
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from ...constants import RobotType, RotationRepresentation, VIEW_ROLE_TO_ID
from ...registry import DATASET_REGISTRY

if TYPE_CHECKING:
    # pyarrow is imported lazily inside the reader methods; the annotation below still needs
    # the name to be resolvable for typing.get_type_hints() and static checkers.
    import pyarrow
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset


@DATASET_REGISTRY.register()
class LiberoPlusDataset(BaseVLADataset):
    """LIBERO-Plus dataset in LeRobot v2.1 format."""

    primary_camera_key = "front"

    _CAMERA_NAME_MAP = {
        "image": "front",
        "agentview_rgb": "front",
        "wrist_image": "wrist",
        "eye_in_hand_rgb": "wrist",
    }

    # View-role assignment (constants.VIEW_ROLES). agentview is a FIXED external rig, i.e.
    # the same physical thing the RynnVLA-Base manifests call "global" -> front_third. It is
    # deliberately NOT "head": role 0 is the carrier's own view (head-mounted / on-robot),
    # which pretraining learns from ~6.5M egocentric episodes; feeding LIBERO's static
    # third-person rig into that slot would contradict the pretrained role semantics.
    # The eye-in-hand camera -> left_wrist (single-arm Panda).
    camera_slot_map = {
        "front": VIEW_ROLE_TO_ID["front_third"],
        "wrist": VIEW_ROLE_TO_ID["left_wrist"],
    }

    def __init__(self, *args, **kwargs):
        data_path = args[0] if args else kwargs.pop("data_path")
        action_chunk_size = kwargs.pop("action_chunk_size", 16)
        use_delta_action = kwargs.pop("use_delta_action", False)
        self._rot6d_layout = kwargs.pop("rot6d_layout", os.environ.get("LIBERO_ROT6D_LAYOUT", "interleaved"))
        if self._rot6d_layout not in ("interleaved", "legacy"):
            raise ValueError(f"Unsupported rot6d_layout: {self._rot6d_layout}")
        self._data_path = Path(data_path)

        # Load meta info
        with open(self._data_path / "meta" / "info.json") as f:
            self._info = json.load(f)

        with open(self._data_path / "meta" / "episodes.jsonl") as f:
            self._episodes_meta = [json.loads(line) for line in f]

        with open(self._data_path / "meta" / "tasks.jsonl") as f:
            self._tasks = {t["task_index"]: t["task"] for t in map(json.loads, f)}

        self._fps = self._info["fps"]
        self._chunk_size = self._info["chunks_size"]
        self._camera_keys = [
            k for k in self._info["features"]
            if k.startswith("observation.images.")
        ]

        state_names = self._info["features"]["observation.state"].get("names") or []
        if isinstance(state_names, dict):
            state_names = state_names.get("motors", [])
        self._state_is_eef = "x" in state_names and "y" in state_names and "z" in state_names

        # Pre-load parquet files lazily
        self._parquet_cache: Dict[str, "pyarrow.Table"] = {}

        super().__init__(data_path=data_path, action_chunk_size=action_chunk_size, use_delta_action=use_delta_action, **kwargs)

    @property
    def episode_lengths(self) -> List[int]:
        return [ep["length"] for ep in self._episodes_meta]

    def get_robot_type(self, episode_index: int) -> RobotType:
        return RobotType.FRANKA

    def get_fps(self, episode_index: int) -> float:
        return float(self._fps)

    def _index_lengths(self):
        # One index per CHUNK START, not per frame. The base default enumerates every target frame
        # while __getitem__ snaps onto chunk_stride, so with the Stage-2 recipe's
        # action_chunk_size=10 + chunk_overlap_ratio=0.5 (stride 5) len() overcounted ~5x: each
        # distinct chunk was handed back five times per epoch, num_update_steps_per_epoch was 5x
        # too large, set_epoch reshuffled far less often than intended, and the resume batch-skip
        # math divided by the wrong epoch length. LatentPretrainDataset already fixed this on its
        # own side via episode_lengths, which is why the two stages disagreed.
        return self._chunk_start_counts()

    def _schema_cache_key(self) -> Dict:
        key = super()._schema_cache_key()
        key["rot6d_layout"] = self._rot6d_layout
        key["state_is_eef"] = self._state_is_eef
        key["schema_version"] = 2  # v2: EEF-state gripper keeps qpos[0] only
        return key

    @staticmethod
    def _retry(fn, max_retries=3, delay=2):
        """Retry a function on I/O errors (OSS transient failures)."""
        import time
        for attempt in range(max_retries):
            try:
                return fn()
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise

    def _get_parquet(self, episode_index: int):
        """Load parquet for a given episode."""
        chunk = episode_index // self._chunk_size
        parquet_path = (
            self._data_path
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        key = str(parquet_path)
        if key not in self._parquet_cache:
            import pyarrow.parquet as pq
            self._parquet_cache[key] = self._retry(lambda: pq.read_table(parquet_path))
        return self._parquet_cache[key]

    def _load_frames(self, episode_index: int, frame_indices: List[int]):
        """Load action/state data from parquet for specific frames."""
        import pyarrow as pa
        import pyarrow.compute as pc

        table = self._get_parquet(episode_index)
        mask = pc.is_in(table["frame_index"], pa.array(frame_indices))
        filtered = table.filter(mask)

        actions = filtered["action"].to_pylist()
        states = filtered["observation.state"].to_pylist()
        return actions, states

    def load_action(self, episode_index, frame_index):
        idxs = self._idxs(episode_index, frame_index)
        actions, _ = self._load_frames(episode_index, idxs)

        # LIBERO action: [dx, dy, dz, dax, day, daz, gripper]
        actions_np = np.array(actions, dtype=np.float32)

        # Convert to RobotAction (left arm only for single-arm Panda)
        eef_pos = Position(torch.from_numpy(actions_np[:, :3]).float())
        # axis-angle -> rotation matrix -> rot_6d
        rot_6d = self._axis_angle_to_rot6d(actions_np[:, 3:6])
        eef_rot = Rotation(
            torch.from_numpy(rot_6d).float(),
            representation=RotationRepresentation.ROT_6D,
        )
        gripper = Position(
            torch.from_numpy(actions_np[:, 6:7]).float(),
            allow_relative=False,
        )

        return RobotAction(
            left_arm=Arm(eef_position=eef_pos, eef_rotation=eef_rot),
            left_gripper=gripper,
        )

    def load_state(self, episode_index, frame_index):
        idxs = self._idxs(episode_index, frame_index)
        _, states = self._load_frames(episode_index, idxs)

        states_np = np.array(states, dtype=np.float32)

        if self._state_is_eef:
            # openpi-style: [eef xyz, axis-angle(3), gripper_qpos(2)].
            # ACTION_LAYOUT reserves 1 dim per gripper; qpos[1] ~= -qpos[0], so keep qpos[0].
            eef_pos = Position(torch.from_numpy(states_np[:, :3]).float())
            rot_6d = self._axis_angle_to_rot6d(states_np[:, 3:6])
            eef_rot = Rotation(
                torch.from_numpy(rot_6d).float(),
                representation=RotationRepresentation.ROT_6D,
            )
            gripper = Position(
                torch.from_numpy(states_np[:, 6:7]).float(),
                allow_relative=False,
            )
            return RobotState(
                left_arm=Arm(eef_position=eef_pos, eef_rotation=eef_rot),
                left_gripper=gripper,
            )

        # legacy: [7 joint angles + 1 gripper]
        joint_pos = Position(torch.from_numpy(states_np[:, :7]).float())
        gripper = Position(
            torch.from_numpy(states_np[:, 7:8]).float(),
            allow_relative=False,
        )

        return RobotState(
            left_arm=Arm(joint_position=joint_pos),
            left_gripper=gripper,
        )

    def load_images(self, episode_index, frame_index):
        """Load images from video files."""
        idxs = self._idxs(episode_index, frame_index)
        out = {}

        for cam_key in self._camera_keys:
            # Map LeRobot camera names to canonical names
            raw_name = cam_key.split(".")[-1]
            cam_name = self._CAMERA_NAME_MAP.get(raw_name, raw_name)

            chunk = episode_index // self._chunk_size
            video_path = (
                self._data_path
                / "videos"
                / f"chunk-{chunk:03d}"
                / cam_key
                / f"episode_{episode_index:06d}.mp4"
            )

            # A missing camera must be fatal. Skipping it produced a sample with one (or zero)
            # cameras and no log at all, which changes camera_slot_ids from [3, 1] to [3] and
            # silently skews training against SERVER_CAMERA_SLOT_MAP.
            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing {cam_key} video for episode {episode_index}: {video_path}"
                )

            out[cam_name] = self._decode_video_frames(str(video_path), idxs, episode_index)

        return out

    def _decode_video_frames(self, video_path: str, frame_indices: List[int], episode_index: int):
        """Decode specific frames from a video file."""
        def _do_decode():
            import torchvision.io
            video, _, info = torchvision.io.read_video(
                video_path, pts_unit="sec", output_format="THWC"
            )
            # The video must actually cover the requested indices. Clamping to the last frame
            # instead would silently repeat a stale frame whenever the MP4 is shorter than the
            # length recorded in episodes.jsonl, with no signal that the metadata is wrong.
            expected = self._episodes_meta[episode_index]["length"]
            max_idx = max(frame_indices)
            if video.shape[0] <= max_idx:
                raise ValueError(
                    f"{video_path} decoded {video.shape[0]} frames but episode {episode_index} "
                    f"metadata claims {expected} and index {max_idx} was requested"
                )
            frames = video[frame_indices]
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
            return frames
        # Only transient I/O is retried; decode/format errors propagate rather than being
        # swallowed into a None that the caller would quietly drop.
        return self._retry(_do_decode)

    def load_instruction(self, episode_index, frame_index):
        n = len(self._idxs(episode_index, frame_index))
        task_idx = self._episodes_meta[episode_index].get("tasks", [""])[0]
        if isinstance(task_idx, str):
            instruction = task_idx
        else:
            instruction = self._tasks.get(task_idx, "")
        return [instruction] * n

    def load_episode(self, episode_index, action_only):
        full = slice(0, self._episodes_meta[episode_index]["length"])
        out = {
            "action": self.load_action(episode_index, full),
            "state": self.load_state(episode_index, full),
        }
        if not action_only:
            out["text"] = self.load_instruction(episode_index, full)
            out["images"] = self.load_images(episode_index, full)
        return out

    def _idxs(self, episode_index, frame_index):
        n = self._episodes_meta[episode_index]["length"]
        if isinstance(frame_index, int):
            return [frame_index]
        if isinstance(frame_index, slice):
            return list(range(*frame_index.indices(n)))
        return list(frame_index)

    def _axis_angle_to_rot6d(self, axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle (N, 3) to rotation 6D (N, 6)."""
        rot = ScipyRotation.from_rotvec(axis_angle)
        mat = rot.as_matrix()  # (N, 3, 3)
        if self._rot6d_layout == "legacy":
            return np.concatenate([mat[:, :, 0], mat[:, :, 1]], axis=1)  # (N, 6)
        return mat[:, :, :2].reshape(-1, 6)  # (N, 6)
