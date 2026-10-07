import json
import os
import random
from typing import Dict

import hashlib
import io
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .base import BaseVLADataset
from ...constants import RobotType, RotationRepresentation, VIEW_ROLE_TO_ID
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from ...registry import DATASET_REGISTRY


SUPPORTED_VARIANTS = (
    "aloha-agilex_clean_50", "aloha-agilex_randomized_500",
)
VARIANT_TO_ROBOT = {
    "franka": RobotType.FRANKA,
    "ur5": RobotType.UR5,
    "aloha-agilex": RobotType.ALOHA_AGILEX,
    "arx-x5": RobotType.ARX_X5,
    "piper": RobotType.PIPER,
}

CAMERAS = ("head", "left", "right")
# RoboTwin endpose quaternion columns are scalar-first (w,x,y,z), not xyzw:
# wxyz yields a constant link6 tool offset (1.6deg spread) vs 17deg for xyzw.
QWXYZ = RotationRepresentation.QUAT_WXYZ
QPOS_SLICES = ((slice(0, 6), slice(6, 7)), (slice(7, 13), slice(13, 14)))

def _episode_variant(episode):
    """Read the variant from new indexes or infer it from legacy index paths."""
    if episode.get("variant"):
        return episode["variant"]
    parts = set(os.path.normpath(episode.get("path", "")).split(os.sep))
    return next((variant for variant in SUPPORTED_VARIANTS if variant in parts), None)


@DATASET_REGISTRY.register()
class RoboTwinDataset(BaseVLADataset):
    """One HDF5 file per episode. The variant selects robot_type, and the instruction is
    drawn at random from the episode's seen list on every __getitem__."""

    primary_camera_key = "head"
    camera_slot_map = {
        "head": VIEW_ROLE_TO_ID["head"],
        "left": VIEW_ROLE_TO_ID["left_wrist"],
        "right": VIEW_ROLE_TO_ID["right_wrist"],
    }

    def __init__(self, *args, index_cache, variants=SUPPORTED_VARIANTS,
                 action_space="ee", schema_path=None, index_sha256=None, schema_sha256=None, **kwargs):
        data_path = args[0] if args else kwargs["data_path"]
        self.action_space = action_space
        self.variants = tuple(variants)
        if action_space not in ("ee", "qpos"):
            raise ValueError(f"Unsupported RoboTwin action space: {action_space}")
        if not self.variants or len(set(self.variants)) != len(self.variants) or any(
                value not in SUPPORTED_VARIANTS for value in self.variants):
            raise ValueError(f"Invalid RoboTwin variants: {self.variants}")
        self.index_cache = str(index_cache)
        raw = Path(index_cache).read_bytes()
        self.index_sha256 = hashlib.sha256(raw).hexdigest()
        if index_sha256 is not None and self.index_sha256 != index_sha256:
            raise ValueError("RoboTwin index checksum differs from the pinned training mixture")
        self._episodes = [ep for ep in json.loads(raw) if _episode_variant(ep) in self.variants]
        if not self._episodes:
            raise ValueError(f"No RoboTwin episodes selected from {index_cache}")
        # The pinned index and schema record whichever mount alias was configured when they
        # were built, and a shared filesystem is commonly reached through a symlink -- an
        # object-store mount aliased to a friendlier path, for example. Resolving only the
        # root would then reject every valid indexed path. Normalize both sides lexically,
        # retaining that alias and removing '..' without stat-ing every file in the corpus.
        root = Path(os.path.abspath(data_path))
        if any(not Path(os.path.abspath(ep["path"])).is_relative_to(root) for ep in self._episodes):
            raise ValueError("RoboTwin index paths do not belong to data_path")
        self.schema_path = schema_path
        self.schema_sha256 = schema_sha256
        self._cache: Dict[int, Dict] = {}
        super().__init__(*args, **kwargs)
        # Preserve historical per-frame sampling, including repeated tail chunks.
        # The changed Base implementation is equivalent to the old one ONLY at stride 1.
        if self.chunk_stride != 1:
            raise ValueError("This RoboTwin recipe preserves historical stride-1 sampling; use chunk_overlap_ratio=0.99")

    def get_schema(self, *args, **kwargs):
        if self.schema_path is None:
            return super().get_schema(*args, **kwargs)
        raw = Path(self.schema_path).read_bytes()
        if self.schema_sha256 is not None and hashlib.sha256(raw).hexdigest() != self.schema_sha256:
            raise ValueError("RoboTwin schema checksum differs from the pinned training mixture")
        saved = json.loads(raw)
        expected_metadata = self._schema_cache_key()
        # Older pinned schemas already contain the mean/std/min/max used by
        # this recipe. The additive quantile-cache marker must not invalidate
        # those immutable statistics. Explicit quantile versions still match
        # strictly, as do all dataset/action metadata and the file checksum.
        if "quantiles" not in saved.get("metadata", {}):
            expected_metadata.pop("quantiles", None)
        if saved.get("metadata") != expected_metadata:
            raise ValueError("Pinned RoboTwin schema metadata differs from this dataset/action recipe")
        if saved.get("index_sha256") != self.index_sha256:
            raise ValueError("Pinned RoboTwin schema belongs to a different episode index")
        schema = {key: saved[key] for key in ("action", "state")}
        for kind in ("action", "state"):
            if set(schema[kind]) != {"aloha_agilex"}:
                raise ValueError("RoboTwin schema must contain exactly aloha_agilex")
        return schema

    @property
    def episode_lengths(self):
        return [ep["length"] for ep in self._episodes]

    def get_robot_type(self, episode_index):
        return RobotType(self._episodes[episode_index]["robot_type"])

    def get_fps(self, episode_index: int) -> float:
        # RoboTwin official sim/real captures run at 25Hz.
        return 25.0

    def _schema_cache_key(self):
        """Add the action space to the base key.

        The base key covers data_path / chunk / delta / rotation repr / fps / episode
        and frame counts -- none of which change when ACTION_SPACE flips, even though
        the schema does (joint_position leaves vs eef_position + eef_rotation leaves,
        with completely different stats). Without this an endpose run and a qpos run
        over the same data would collide and the second one would silently train
        against the first one's normalization statistics. Overridden here rather than
        in BaseVLADataset so no other dataset's cache is invalidated.
        """
        key = super()._schema_cache_key()
        key["robotwin_action_space"] = self.action_space
        key["robotwin_variants"] = list(self.variants)
        return key

    # -- abstract interface -----------------------------------------------
    def _retry_read(self, fn, episode_index):
        """Retry a read operation; on I/O error, re-open the file handle and retry."""
        import time
        for attempt in range(3):
            try:
                return fn()
            except (OSError, IOError, KeyError) as e:
                # Invalidate cached handle so _handle() re-opens the file
                b = self._cache.get(os.getpid(), {})
                if "h5" in b:
                    try: b["h5"].close()
                    except: pass
                    b.pop("h5", None)
                    b.pop("path", None)
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise

    def _load_arms(self, episode_index, frame_index):
        """Read (arms, grippers) for a frame range in the configured action space.

        State and action are read from the same source, exactly as before: RoboTwin
        records the commanded value, and the observed value is the previous command.
        """
        idxs = self._idxs(episode_index, frame_index)

        def _read():
            f = self._handle(episode_index)
            arms, grippers = [], []
            if self.action_space == "qpos":
                v = f["joint_action/vector"][idxs]
                for arm_sl, grip_sl in QPOS_SLICES:
                    arms.append(Arm(
                        joint_position=Position(torch.from_numpy(v[:, arm_sl]).float()),
                    ))
                    # allow_relative=False keeps the gripper out of the delta channel
                    # whatever --use_delta_action says; every peer treats it as absolute.
                    grippers.append(Position(
                        torch.from_numpy(v[:, grip_sl]).float(), allow_relative=False
                    ))
            else:
                for side in ("left", "right"):
                    ep = f[f"endpose/{side}_endpose"][idxs]
                    gp = f[f"endpose/{side}_gripper"][idxs].reshape(-1, 1)
                    arms.append(Arm(
                        eef_position=Position(torch.from_numpy(ep[:, :3]).float()),
                        eef_rotation=Rotation(
                            torch.from_numpy(ep[:, 3:7]).float(), representation=QWXYZ
                        ),
                    ))
                    grippers.append(Position(torch.from_numpy(gp).float(), allow_relative=False))
            return arms, grippers

        return self._retry_read(_read, episode_index)

    def load_action(self, episode_index, frame_index):
        arms, grippers = self._load_arms(episode_index, frame_index)
        return RobotAction(
            left_arm=arms[0], right_arm=arms[1],
            left_gripper=grippers[0], right_gripper=grippers[1],
        )

    def load_state(self, episode_index, frame_index):
        arms, grippers = self._load_arms(episode_index, frame_index)
        return RobotState(
            left_arm=arms[0], right_arm=arms[1],
            left_gripper=grippers[0], right_gripper=grippers[1],
        )

    def load_images(self, episode_index, frame_index):
        idxs = self._idxs(episode_index, frame_index)
        path = self._episodes[episode_index]["path"]

        def _decode(f, key, i):
            # Returns RGB, not BGR. RoboTwin's camera.get_rgb() yields RGB from sapien's
            # get_picture("Color"), and pkl2hdf5.images_encoding hands that RGB array to
            # cv2.imencode, which assumes BGR and swaps it on the way in. cv2.imdecode
            # swaps back, so the decoded array is the ORIGINAL RGB. An extra [..., ::-1]
            # here (as robomind.py correctly needs, since robomind stores true BGR) would
            # make every training frame BGR while eval feeds RGB. Verified against the
            # rgb24 mp4 RoboTwin writes itself: unflipped |diff| 1.4-2.5, flipped 5.0-6.2.
            buf = bytes(f[key][i])
            if not buf:
                return None
            try:
                # RoboTwin fed RGB to cv2.imencode (which assumes BGR). Pillow
                # decodes file RGB, so reverse once to recover the original sensor
                # RGB, exactly matching the old cv2.imdecode output. Pixel equality
                # checked on all three cameras in clean and randomized episodes.
                # This avoids adding OpenCV's libGL/X11 dependency to CUDA workers.
                with Image.open(io.BytesIO(buf)) as image:
                    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            except (UnidentifiedImageError, OSError):
                return None

        def _read():
            f = self._handle(episode_index)
            out = {}
            for cam in CAMERAS:
                key = f"observation/{cam}_camera/rgb"
                if key not in f:
                    continue
                frames = []
                for i in idxs:
                    img = _decode(f, key, i)
                    if img is None:
                        # An empty/truncated JPEG blob must not kill a multi-hour run
                        # (one did at step 52015). Adjacent 25Hz frames are near-identical,
                        # so substitute the nearest decodable one and say so loudly.
                        n = int(f[key].shape[0])
                        for off in range(1, n):
                            for j in (i - off, i + off):
                                if 0 <= j < n:
                                    img = _decode(f, key, j)
                                    if img is not None:
                                        break
                            if img is not None:
                                break
                        if img is None:
                            raise OSError(f"no decodable frame near {key}[{i}] in {path}")
                        print(f"[robotwin] WARNING: corrupt frame {key}[{i}] in {path}; "
                              f"substituted frame {j}", flush=True)
                    frames.append(img)
                out[cam] = torch.from_numpy(np.stack(frames))
            return out

        return self._retry_read(_read, episode_index)

    def load_instruction(self, episode_index, frame_index):
        n = len(self._idxs(episode_index, frame_index))
        ins = self._episodes[episode_index]["instructions"]
        return [random.choice(ins) if ins else ""] * n

    def load_episode(self, episode_index, action_only):
        full = slice(0, self._episodes[episode_index]["length"])
        out = {"action": self.load_action(episode_index, full),
               "state":  self.load_state(episode_index, full)}
        if not action_only:
            out["text"]   = self.load_instruction(episode_index, full)
            out["images"] = self.load_images(episode_index, full)
        return out

    # -- internals --------------------------------------------------------
    @staticmethod
    def _retry(fn, max_retries=3, delay=2):
        """Retry a function on I/O errors (OSS transient failures)."""
        import time
        for attempt in range(max_retries):
            try:
                return fn()
            except (OSError, IOError, KeyError) as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise

    def _idxs(self, episode_index, frame_index):
        n = self._episodes[episode_index]["length"]
        if isinstance(frame_index, int):   return [frame_index]
        if isinstance(frame_index, slice): return list(range(*frame_index.indices(n)))
        return list(frame_index)

    def _handle(self, episode_index):
        """One single-slot LRU per PID: close the old handle and drop the cache on an episode switch."""
        ep = self._episodes[episode_index]
        b = self._cache.setdefault(os.getpid(), {})
        if b.get("path") != ep["path"]:
            if "h5" in b:
                b["h5"].close()
            b["h5"] = self._retry(lambda: self._open(ep["path"]))
            b["path"] = ep["path"]
        return b["h5"]

    @staticmethod
    def _open(path):
        return h5py.File(path, "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0)
