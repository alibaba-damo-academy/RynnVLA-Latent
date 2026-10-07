"""RynnVLA inference and the narrow V5 LIBERO raw-command adapter.

Simulator and model packages are deliberately imported only at their call sites.
"""

import logging
import math

import numpy as np
import torch

from .base import BaseVLAInferenceWrapper
from ..registry import INFERENCE_WRAPPER_REGISTRY
from ..constants import RobotType, ROBOT_TYPE_TO_EE_ID, RotationRepresentation, VIEW_ROLE_TO_ID
from ..utils.robot import Arm, Position, RobotAction, RobotState, Rotation

logger = logging.getLogger(__name__)

LIBERO_CAMERA_SLOT_MAP = {
    "front": VIEW_ROLE_TO_ID["front_third"],
    "wrist": VIEW_ROLE_TO_ID["left_wrist"],
}

# VLABench eval camera -> role map. MUST equal VLABenchDataset._CAMERA_SLOTS (the train-time map)
# or use_view_cond_slots seeds a different role at eval than at train -- a silent skew. image is
# the forward rig -> front_third; second_image is a single generic side camera -> side_left (the
# primary side slot, per constants.VIEW_NAME_TO_ROLE["side"]); wrist_image -> left_wrist.
VLABENCH_CAMERA_SLOT_MAP = {
    "image": VIEW_ROLE_TO_ID["front_third"],
    "second_image": VIEW_ROLE_TO_ID["side_left"],
    "wrist_image": VIEW_ROLE_TO_ID["left_wrist"],
}


def libero_state_to_robot_state(state):
    """World xyz + axis-angle + raw gripper qpos (7 or 8 values), not joints."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape not in ((7,), (8,)) or not np.isfinite(state).all():
        raise ValueError("LIBERO state must be 7/8 finite values: world xyz, axis-angle, qpos[0:1/2]")
    return RobotState(
        left_arm=Arm(
            eef_position=Position(torch.from_numpy(state[:3].copy())),
            eef_rotation=Rotation(
                torch.from_numpy(state[3:6].copy()), RotationRepresentation.ROT_VEC,
            ).convert_rotation(RotationRepresentation.ROT_6D),
        ),
        left_gripper=Position(torch.from_numpy(state[6:7].copy()), allow_relative=False),
    )


def libero_actions_to_robot_action(actions):
    """Raw OSC (T, 7) -> dataset RobotAction; no OSC scaling or state subtraction."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("LIBERO actions must have finite, nonempty shape (T, 7)")
    return RobotAction(
        left_arm=Arm(
            eef_position=Position(torch.from_numpy(actions[:, :3].copy())),
            eef_rotation=Rotation(
                torch.from_numpy(actions[:, 3:6].copy()), RotationRepresentation.ROT_VEC,
            ).convert_rotation(RotationRepresentation.ROT_6D),
        ),
        left_gripper=Position(torch.from_numpy(actions[:, 6:7].copy()), allow_relative=False),
    )


def robot_action_to_libero_actions(action):
    """Denormalized interleaved-rot6d RobotAction -> dataset-space actions (T, 7)."""
    if isinstance(action, dict):
        action = RobotAction.from_dict(action)
    arm, grip = action.left_arm, action.left_gripper
    if arm is None or arm.eef_position is None or arm.eef_rotation is None or grip is None:
        raise ValueError("LIBERO requires left-arm EEF position/rotation and left gripper")
    if any(leaf.is_relative for leaf in (arm.eef_position, arm.eef_rotation, grip)):
        raise ValueError("V5 requires raw-command targets, not state-relative RobotAction fields")
    rotation = arm.eef_rotation.convert_rotation(RotationRepresentation.ROT_VEC)
    values = torch.cat((arm.eef_position.data, rotation.data, grip.data), dim=-1)
    commands = values.detach().cpu().float().numpy()
    if (
        commands.ndim != 2
        or commands.shape[1] != 7
        or not len(commands)
        or not np.isfinite(commands).all()
    ):
        raise ValueError("Decoded LIBERO actions must have finite, nonempty shape (T, 7)")
    return commands


def libero_dataset_actions_to_commands(actions):
    """Convert LIBERO dataset gripper values to robosuite's actuator convention.

    The demonstrations store an absolute open amount (1=open, 0=closed), while
    PandaGripper.format_action expects a signed command (-1=open, +1=close).
    Arm position and rotation values are already raw OSC commands and pass through.
    """
    commands = np.asarray(actions, dtype=np.float32).copy()
    if commands.ndim != 2 or commands.shape[1] != 7 or not len(commands) or not np.isfinite(commands).all():
        raise ValueError("LIBERO dataset actions must have finite, nonempty shape (T, 7)")
    commands[:, 6] = 1.0 - 2.0 * commands[:, 6]
    return commands


def libero_images(images, convention="dataset"):
    """Validate RGB uint8 HWC images, rotating raw renders exactly once.

    The actual LIBERO client uses [::-1, ::-1] for BOTH cameras. Dataset/replay
    images already have that orientation; the processor owns resize/normalization.
    """
    if convention not in ("dataset", "libero_raw"):
        raise ValueError("Image convention must be 'dataset' or 'libero_raw'")
    result = {}
    for camera in LIBERO_CAMERA_SLOT_MAP:
        image = np.asarray(images[camera])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError(f"{camera} must be a nonempty RGB uint8 HWC image")
        if convention == "libero_raw":
            image = image[::-1, ::-1]
        result[camera] = np.ascontiguousarray(image)
    return result


def libero_observation_to_sample(observation):
    """Convert OffScreenRenderEnv observations using the verified client convention."""
    quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
        raise ValueError("robot0_eef_quat must be a finite unit xyzw quaternion")
    # Match robosuite/openpi quat2axisangle, including its quaternion sign convention.
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(1.0 - w * w)
    rotvec = np.zeros(3) if math.isclose(denominator, 0.0) else quat[:3] * 2.0 * math.acos(w) / denominator
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper.size != 2:
        raise ValueError("LIBERO Panda observation requires two raw gripper qpos values")
    state = np.concatenate((
        np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3), rotvec, gripper,
    )).astype(np.float32)
    libero_state_to_robot_state(state)  # Validate before any model call.
    images = libero_images({
        "front": observation["agentview_image"],
        "wrist": observation["robot0_eye_in_hand_image"],
    }, convention="libero_raw")
    return {"state": state, "images": images}


def validate_libero_schema(processor, config):
    """Reject incompatible legacy protocols; never override trained checkpoint flags."""
    if config.action_dim != 81:
        raise ValueError("V5 LIBERO requires the 81-dimensional RobotAction layout")
    if getattr(config, "use_latent_actions", False):
        raise ValueError("Use a Stage2 action checkpoint, not a Stage1 latent-only checkpoint")
    if int(config.action_chunk_size) < 1:
        raise ValueError("Checkpoint action_chunk_size must be positive")
    for owner in (processor, config):
        if getattr(owner, "rot6d_layout", "interleaved") != "interleaved":
            raise ValueError("Only V5 interleaved rot6d checkpoints are supported")
    for section in ("state", "action"):
        try:
            schema = processor.schema[section]["franka"]
            arm = schema["left_arm"]
            leaves = (arm["eef_position"], arm["eef_rotation"], schema["left_gripper"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Checkpoint lacks V5 LIBERO EEF {section} schema") from exc
        if set(schema) - {"type", "left_arm", "left_gripper"} or set(arm) - {"type", "eef_position", "eef_rotation"}:
            raise ValueError(f"Unsupported LIBERO {section} fields; expected single-arm EEF schema")
        for leaf, dim in zip(leaves, (3, 6, 1)):
            if leaf.get("dim") != dim or leaf.get("is_relative", False):
                raise ValueError(f"V5 LIBERO {section} requires non-relative xyz/rot6d/gripper fields")
        if leaves[1].get("representation") != "rot_6d":
            raise ValueError("V5 LIBERO schema must use interleaved rot_6d")


def vlabench_state_to_robot_state(state):
    """VLABench eval state (7,) -> RobotState, mirroring the training adapter.

    Layout: ee_pos in the robot-base frame (3) + ee euler_xyz (3) + raw gripper flag (1).
    Matches datasets/vla_datasets/vlabench.py:_build_vlabench_state -- rotation is EULER_XYZ
    converted to the canonical interleaved ROT_6D, and the raw flag (upstream get_ee_open_state
    returns 1 when CLOSED) is inverted to the framework convention 1 = open, so the eval state
    agrees with the trained schema and the action channel.
    """
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("VLABench state must be 7 finite values: ee_pos_base(3), euler_xyz(3), gripper flag(1)")
    gripper = 1.0 - state[6:7]  # raw 1=closed -> framework 1=open
    return RobotState(
        left_arm=Arm(
            eef_position=Position(torch.from_numpy(state[:3].copy())),
            eef_rotation=Rotation(
                torch.from_numpy(state[3:6].copy()), RotationRepresentation.EULER_XYZ,
            ).convert_rotation(RotationRepresentation.ROT_6D),
        ),
        left_gripper=Position(torch.from_numpy(gripper.copy()), allow_relative=False),
    )


def vlabench_images(images):
    """Validate the three VLABench RGB uint8 HWC cameras; no rotation (renders are upright).

    Names match the eval harness client and the training dataset: image (forward), second_image
    (robot-right side), wrist_image (left wrist). The processor owns resize/normalization. Unlike
    LIBERO there is no 180-degree client flip -- that is a LIBERO MuJoCo render artifact.
    """
    result = {}
    for camera in VLABENCH_CAMERA_SLOT_MAP:
        image = np.asarray(images[camera])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError(f"{camera} must be a nonempty RGB uint8 HWC image")
        result[camera] = np.ascontiguousarray(image)
    return result


def robot_action_to_vlabench_actions(action):
    """Denormalized interleaved-rot6d RobotAction -> VLABench controller targets (T, 7).

    Layout matches the harness command and the trained action: ABSOLUTE
    [ee_pos_base(3), ee_euler_xyz_world(3), gripper(1 = open)]. Rotation is converted
    ROT_6D -> EULER_XYZ (the VLABench training representation). Unlike LIBERO there is NO gripper
    sign flip: the client binarizes at 0.5 on the 1=open convention the model already emits.
    """
    if isinstance(action, dict):
        action = RobotAction.from_dict(action)
    arm, grip = action.left_arm, action.left_gripper
    if arm is None or arm.eef_position is None or arm.eef_rotation is None or grip is None:
        raise ValueError("VLABench requires left-arm EEF position/rotation and left gripper")
    if any(leaf.is_relative for leaf in (arm.eef_position, arm.eef_rotation, grip)):
        raise ValueError("VLABench trains absolute targets, not state-relative RobotAction fields")
    rotation = arm.eef_rotation.convert_rotation(RotationRepresentation.EULER_XYZ)
    values = torch.cat((arm.eef_position.data, rotation.data, grip.data), dim=-1)
    commands = values.detach().cpu().float().numpy()
    if commands.ndim != 2 or commands.shape[1] != 7 or not len(commands) or not np.isfinite(commands).all():
        raise ValueError("Decoded VLABench actions must have finite, nonempty shape (T, 7)")
    return commands


def vlabench_observation_to_sample(observation):
    """Convert a VLABench eval observation into predict_vlabench inputs.

    Accepts the harness client message obs dict: {"images": {image, second_image, wrist_image},
    "state": (7,) [ee_pos_base, euler_xyz, raw gripper flag], "prompt": str}. The gripper inversion
    and rotation handling live in vlabench_state_to_robot_state (called inside predict_vlabench).
    """
    state = np.asarray(observation["state"], dtype=np.float32).reshape(7)
    vlabench_state_to_robot_state(state)  # validate before any model call
    images = vlabench_images(observation["images"])
    return {"state": state, "images": images, "text": observation["prompt"]}


def validate_vlabench_schema(processor, config):
    """VLABench stage2 exports share LIBERO's franka single-arm EEF rot_6d contract.

    Mirrors validate_libero_schema (action_dim 81, use_latent_actions False, interleaved rot_6d,
    non-relative eef_position(3)/eef_rotation(6)/left_gripper(1)); kept separate so the validated
    LIBERO path is untouched and VLABench failures read as VLABench.
    """
    if config.action_dim != 81:
        raise ValueError("VLABench requires the 81-dimensional RobotAction layout")
    if getattr(config, "use_latent_actions", False):
        raise ValueError("Use a Stage2 action checkpoint, not a Stage1 latent-only checkpoint")
    if int(config.action_chunk_size) < 1:
        raise ValueError("Checkpoint action_chunk_size must be positive")
    for owner in (processor, config):
        if getattr(owner, "rot6d_layout", "interleaved") != "interleaved":
            raise ValueError("Only VLABench interleaved rot6d checkpoints are supported")
    for section in ("state", "action"):
        try:
            schema = processor.schema[section]["franka"]
            arm = schema["left_arm"]
            leaves = (arm["eef_position"], arm["eef_rotation"], schema["left_gripper"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Checkpoint lacks VLABench EEF {section} schema") from exc
        if set(schema) - {"type", "left_arm", "left_gripper"} or set(arm) - {"type", "eef_position", "eef_rotation"}:
            raise ValueError(f"Unsupported VLABench {section} fields; expected single-arm EEF schema")
        for leaf, dim in zip(leaves, (3, 6, 1)):
            if leaf.get("dim") != dim or leaf.get("is_relative", False):
                raise ValueError(f"VLABench {section} requires non-relative xyz/rot6d/gripper fields")
        if leaves[1].get("representation") != "rot_6d":
            raise ValueError("VLABench schema must use interleaved rot_6d")


@INFERENCE_WRAPPER_REGISTRY.register("rynn_brain_vla")
class RynnBrainVLAInferenceWrapper(BaseVLAInferenceWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled = False
        self._decode_fn = self._decode_step
        self._decode_rtc_fn = self._decode_rtc_step
        self.ee_type_id = None  # Cached from prefill() for use in decode()

    def load_model(self):
        from ..models.rynn_brain_vla import RynnBrainVLAModel

        model = RynnBrainVLAModel.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": str(self.device)},
            local_files_only=self.local_files_only,
        )
        return model.eval()

    def load_processor(self):
        from ..models.rynn_brain_vla import RynnBrainVLAProcessor

        return RynnBrainVLAProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only,
        )

    @torch.no_grad()
    def predict_libero(self, text, images, state, denoising_steps=10):
        """Predict one independent raw-OSC chunk from the current observation.

        Always re-prefill: a closed-loop replan must never reuse an earlier image's
        KV cache. Trained architecture, flags and normalization stay checkpoint-owned.
        """
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty task instruction is required")
        state_dict = libero_state_to_robot_state(state).to_dict()
        images = libero_images(images)
        validate_libero_schema(self.processor, self.model.config)
        inputs = self.process(
            text=text, images=images, state=state_dict, robot_type=RobotType.FRANKA.value,
            camera_slot_map=LIBERO_CAMERA_SLOT_MAP,
        )
        inputs = self.collate([inputs])
        inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        cache = self.prefill(inputs)
        actions = self.decode(inputs, cache, denoising_steps, robot_type=RobotType.FRANKA.value)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 81:
            raise ValueError(f"Expected model actions (1, T, 81), got {tuple(actions.shape)}")
        # The dataset stores OSC commands as non-relative RobotAction fields.
        # Denormalize only; passing state would incorrectly invoke pose composition.
        action = self.processor.post_process(
            actions[0].detach().cpu().float(), robot_type=RobotType.FRANKA, state=None,
        )
        return libero_dataset_actions_to_commands(robot_action_to_libero_actions(action))

    @torch.no_grad()
    def predict_vlabench(self, text, images, state, denoising_steps=10, rtc=None):
        """Predict one absolute EE-pose chunk for VLABench from the current observation.

        The VLABench counterpart of predict_libero: same wrapper pipeline (process / collate /
        prefill / decode / post_process), with the differences isolated in the adapters -- EULER_XYZ
        state rotation with the gripper flag inverted to 1=open, the 3-camera VLABENCH_CAMERA_SLOT_MAP,
        no 180-degree image flip, and an absolute (state=None) post_process whose rot_6d output is
        converted back to euler_xyz with no gripper sign flip. Always re-prefill: a closed-loop replan
        must never reuse an earlier image's KV cache.

        rtc=None (the default) is the historical path and is bit-identical to it. Passing a dict
        routes the denoising loop through decode_rtc instead of decode; see _rtc_state below.
        """
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty task instruction is required")
        state_dict = vlabench_state_to_robot_state(state).to_dict()
        images = vlabench_images(images)
        validate_vlabench_schema(self.processor, self.model.config)
        inputs = self.process(
            text=text, images=images, state=state_dict, robot_type=RobotType.FRANKA.value,
            camera_slot_map=VLABENCH_CAMERA_SLOT_MAP,
        )
        inputs = self.collate([inputs])
        inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        cache = self.prefill(inputs)
        if rtc is None:
            actions = self.decode(inputs, cache, denoising_steps,
                                  robot_type=RobotType.FRANKA.value)
        else:
            actions = self.decode_rtc(
                inputs, cache, denoising_steps, RobotType.FRANKA.value,
                prev_actions=rtc["prev_actions"],
                delay_steps=int(rtc.get("delay_steps", 0)),
                execution_horizon=rtc.get("execution_horizon"),
                beta=float(rtc.get("beta", 10.0)),
            )
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 81:
            raise ValueError(f"Expected model actions (1, T, 81), got {tuple(actions.shape)}")
        # Stash the normalized model-space chunk so the caller can hand the unexecuted tail back
        # as prev_actions on the next query. decode_rtc guides x1_hat toward prev_actions, and
        # x1_hat lives in this space, so the tail must be captured BEFORE post_process denormalizes
        # it into absolute EE poses. Kept on device; one (1, T, 81) tensor per query.
        self.last_vlabench_model_actions = actions.detach()
        # VLABench trains absolute targets: denormalize only (state=None), no pose composition.
        action = self.processor.post_process(
            actions[0].detach().cpu().float(), robot_type=RobotType.FRANKA, state=None,
        )
        return robot_action_to_vlabench_actions(action)

    def process(self, text, images, state, robot_type, camera_slot_map=None):
        # camera_slot_map (camera name -> constants.VIEW_ROLES id) is what makes the two
        # camera-identity mechanisms work at eval: the processor needs it to write the
        # per-view prompt labels (view_label_in_prompt) and to emit camera_slot_ids for
        # the prefix role embedding (use_view_role_embedding). Without it the model would
        # be trained with camera identity and evaluated without -- a silent skew. When
        # both flags are off it is inert: the labels are not written and camera_slot_ids
        # is ignored by the model, so the baseline is unchanged.
        model_inputs = self.processor(
            text=text,
            robot_type=RobotType(robot_type),
            state=RobotState.from_dict(state),
            images=images,
            camera_slot_map=camera_slot_map,
        )
        # Pass robot_type through to collate() so it can generate ee_type_id.
        model_inputs["robot_type"] = robot_type
        return model_inputs

    def collate(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"][0] for instance in batch],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
            padding_side="right",
        )

        position_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["position_ids"][:, 0].transpose(0, 1) for instance in batch],
            batch_first=True,
            padding_value=1,
            padding_side="right",
        ).permute(2, 0, 1)

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                instance.get("attention_mask", torch.ones_like(instance["input_ids"]))[0]
                for instance in batch
            ],
            batch_first=True,
            padding_value=0,
            padding_side="right",
        )

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "states": torch.cat([instance["states"] for instance in batch], dim=0),
        }

        # Flat (sum over batch of num_images,), cat'd exactly like image_grid_thw, which is
        # how the model indexes it (per image, not per sample). Absent when the caller did
        # not supply a camera_slot_map.
        if "camera_slot_ids" in batch[0]:
            model_inputs["camera_slot_ids"] = torch.cat(
                [instance["camera_slot_ids"] for instance in batch], dim=0
            )

        mm_input_names = set(
            self.processor.image_processor.model_input_names + self.processor.video_processor.model_input_names
        )

        for key in mm_input_names:
            data_list = [instance[key] for instance in batch if key in instance]
            if len(data_list) > 0:
                model_inputs[key] = torch.cat(data_list, dim=0)

        # EE type embedding: map robot_type → ee_type_id for embodiment-aware conditioning.
        if "robot_type" in batch[0]:
            ee_type_ids = [
                ROBOT_TYPE_TO_EE_ID.get(RobotType(instance["robot_type"]), 0)  # 0 = DEFAULT fallback
                for instance in batch
            ]
            model_inputs["ee_type_id"] = torch.tensor(ee_type_ids, dtype=torch.long)

        return model_inputs

    def compile_model(self):
        logger.info("Compiling decode step (mode=reduce-overhead)...")
        self._decode_fn = torch.compile(self._decode_step, mode="reduce-overhead")
        logger.info("Compiling RTC decode step (mode=default)...")
        self._decode_rtc_fn = torch.compile(self._decode_rtc_step, mode="default")
        self._compiled = True
        logger.info("torch.compile registered; actual compilation on first call")

    def _decode_step(self, actions, times, past_key_values, cache_position, position_ids, ee_type_id=None):
        return self.model(
            actions=actions,
            times=times,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            ee_type_id=ee_type_id,
        )

    def _decode_rtc_step(self, actions, times, past_key_values, cache_position, position_ids, ee_type_id=None):
        return self.model(
            actions=actions,
            times=times,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            ee_type_id=ee_type_id,
        )

    def _trim_action_dim(self, model_inputs):
        action_dim = self.model.config.action_dim
        for key in ("states", "actions", "action_mask"):
            if key in model_inputs and model_inputs[key].size(-1) != action_dim:
                model_inputs[key] = model_inputs[key][..., :action_dim]

    def _get_action_mask(self, robot_type, chunk_size, batch_size, device):
        if robot_type is None:
            return None
        if isinstance(robot_type, (list, tuple)):
            mask = torch.stack(
                [self.processor.get_action_mask(RobotType(rt), chunk_size) for rt in robot_type]
            )
        else:
            mask = self.processor.get_action_mask(RobotType(robot_type), chunk_size)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        action_dim = self.model.config.action_dim
        if mask.size(-1) != action_dim:
            mask = mask[..., :action_dim]
        return mask.to(device)

    def prefill(self, model_inputs):
        from ..models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLACache

        self._trim_action_dim(model_inputs)
        # Cache ee_type_id only for this prefix, never a previous request.
        self.ee_type_id = model_inputs.get("ee_type_id")
        past_key_values = RynnBrainVLACache()
        self.model(
            **model_inputs,
            past_key_values=past_key_values,
        )
        return {"past_key_values": past_key_values}

    def decode(self, model_inputs, cache, num_steps, robot_type=None):
        # Exit inference_mode (if set by caller) so that CUDA graph pool
        # buffers are regular tensors, compatible with the enable_grad
        # context used by decode_rtc on the same device.
        with torch.inference_mode(False), torch.no_grad():
            return self._decode_inner(model_inputs, cache, num_steps, robot_type)

    def _decode_inner(self, model_inputs, cache, num_steps, robot_type=None):
        batch_size = model_inputs["input_ids"].size(0)
        prefix_cache = cache["past_key_values"]
        cache_length = prefix_cache.get_seq_length()
        device = model_inputs["input_ids"].device
        action_chunk_size = self.model.config.action_chunk_size

        cache_position = torch.arange(action_chunk_size, device=device) + cache_length

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        action_mask = self._get_action_mask(robot_type, action_chunk_size, batch_size, device)

        actions_shape = (batch_size, action_chunk_size, self.model.config.action_dim)
        x_t = self.model.sample_noise(actions_shape)
        if action_mask is not None:
            x_t[~action_mask] = 0.0
        times = torch.tensor(1.0, dtype=torch.float32, device=device).expand(batch_size)

        position_ids = torch.arange(1, action_chunk_size + 1).unsqueeze(0).unsqueeze(1)
        position_ids = position_ids.repeat(3, batch_size, 1)
        last_position = prefix_cache.prefix_last_position
        if last_position is None:
            from ..models.rynn_brain_vla.modeling_rynn_brain_vla import _global_last_position

            last_position = _global_last_position(
                model_inputs["position_ids"], model_inputs.get("attention_mask")
            )
        position_ids = position_ids.to(last_position) + last_position

        while times[0] >= -dt / 2:
            outputs = self._decode_fn(
                actions=x_t,
                times=times,
                past_key_values=cache["past_key_values"],
                cache_position=cache_position,
                position_ids=position_ids,
                ee_type_id=self.ee_type_id,
            )
            x_t += dt * outputs.actions
            if action_mask is not None:
                x_t[~action_mask] = 0.0
            times = times + dt

        return x_t

    @staticmethod
    def _rtc_soft_mask(H, d, s, device):
        """Compute soft guidance mask W (RTC paper Eq 5).

        W[i] = 1                              if i < d       (frozen)
        W[i] = c_i * (e^c_i - 1) / (e - 1)   if d <= i < H-s (transition)
        W[i] = 0                              if i >= H-s    (fresh)
        where c_i = (H - s - i) / (H - s - d + 1)
        """
        W = torch.zeros(H, device=device)
        W[:d] = 1.0
        e_val = math.e
        for i in range(d, H - s):
            c_i = (H - s - i) / (H - s - d + 1)
            W[i] = c_i * (math.exp(c_i) - 1) / (e_val - 1)
        return W

    def decode_rtc(self, model_inputs, cache, num_steps, robot_type,
                   prev_actions, delay_steps, execution_horizon=None, beta=10.0):
        with torch.inference_mode(False), torch.no_grad():
            return self._decode_rtc_inner(
                model_inputs, cache, num_steps, robot_type, prev_actions, delay_steps,
                execution_horizon=execution_horizon, beta=beta,
            )

    def _decode_rtc_inner(self, model_inputs, cache, num_steps, robot_type,
                          prev_actions, delay_steps, execution_horizon=None, beta=10.0):
        """GuidedInference from RTC, following LeRobot/PI0 reference.

        Args:
            prev_actions: (B, H_prev, action_dim) — remaining actions from
                the previous chunk, in normalized model space.
            delay_steps: int d — estimated inference delay in control steps.
            execution_horizon: int s — execution horizon. Defaults to d.
            beta: float — max guidance weight (default 10.0).
        """
        batch_size = model_inputs["input_ids"].size(0)
        prefix_cache = cache["past_key_values"]
        cache_length = prefix_cache.get_seq_length()
        device = model_inputs["input_ids"].device
        action_chunk_size = self.model.config.action_chunk_size
        action_dim = self.model.config.action_dim
        H = action_chunk_size
        d = delay_steps
        s = execution_horizon if execution_horizon is not None else d
        s = max(s, d)

        cache_position = torch.arange(H, device=device) + cache_length

        n = num_steps
        dt_val = -1.0 / n

        action_mask = self._get_action_mask(robot_type, H, batch_size, device)

        if prev_actions.size(-1) != action_dim:
            prev_actions = prev_actions[..., :action_dim]

        H_prev = prev_actions.size(1)
        if H_prev < H:
            pad = torch.zeros(batch_size, H - H_prev, prev_actions.size(2),
                              device=device, dtype=prev_actions.dtype)
            prev_padded = torch.cat([prev_actions, pad], dim=1)
        else:
            prev_padded = prev_actions[:, :H]
        prev_padded = prev_padded.clone()

        W = self._rtc_soft_mask(H, d, s, device)
        W_expanded = W.view(1, H, 1).expand(batch_size, H, action_dim)
        if action_mask is not None:
            W_expanded = W_expanded * action_mask.float()

        actions_shape = (batch_size, H, action_dim)
        x_t = self.model.sample_noise(actions_shape)
        if action_mask is not None:
            x_t[~action_mask] = 0.0
        times = torch.empty(batch_size, dtype=torch.float32, device=device)

        position_ids = torch.arange(1, H + 1).unsqueeze(0).unsqueeze(1)
        position_ids = position_ids.repeat(3, batch_size, 1)
        last_position = prefix_cache.prefix_last_position
        if last_position is None:
            from ..models.rynn_brain_vla.modeling_rynn_brain_vla import _global_last_position

            last_position = _global_last_position(
                model_inputs["position_ids"], model_inputs.get("attention_mask")
            )
        position_ids = position_ids.to(last_position) + last_position

        prefix_cache.freeze()
        try:
            for step in range(n):
                t = 1.0 - step / n
                tau = step / n
                times.fill_(t)

                x_t = x_t.detach().requires_grad_(True)
                with torch.enable_grad():
                    outputs = self._decode_rtc_fn(
                        actions=x_t,
                        times=times,
                        past_key_values=prefix_cache,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        ee_type_id=self.ee_type_id,
                    )
                    v = outputs.actions
                    x1_hat = x_t - t * v
                    err = (prev_padded - x1_hat) * W_expanded
                    correction = torch.autograd.grad(
                        x1_hat, x_t, grad_outputs=err.detach(), retain_graph=False
                    )[0]

                one_minus_tau = t
                sq_one_minus_tau = one_minus_tau ** 2
                inv_r2 = (sq_one_minus_tau + tau ** 2) / (sq_one_minus_tau + 1e-8)
                c = min(one_minus_tau / (tau + 1e-8), beta)
                w = min(c * inv_r2, beta)

                v_guided = v.detach() - w * correction.detach()
                x_t = x_t.detach() + dt_val * v_guided
                if action_mask is not None:
                    x_t[~action_mask] = 0.0

            return x_t
        finally:
            prefix_cache.unfreeze()

    def post_process(self, action, state, robot_type):
        action = self.processor.post_process(
            action=action,
            state=RobotState.from_dict(state),
            robot_type=RobotType(robot_type),
        )
        return action.to_dict()
