"""Native RoboTwin protocol adapter for Stage-2 EE-delta checkpoints.

Client imports stay light and the model imports happen only inside get_model(), so
a RoboTwin runner can import this module without paying for torch or the model
package up front. The simulator side is not bundled: RoboTwin's own eval entry
point loads this module and calls get_model / reset_model / eval below.
"""

import os

import numpy as np


CAMERA_SLOTS = {"head": 0, "left": 1, "right": 2}


def validate_checkpoint(config, processor):
    expected = {"action_dim": 81, "action_chunk_size": 30, "use_latent_actions": False,
                "use_latent_head_readout": True, "use_view_cond_slots": True,
                "latent_action_dim": 608, "bypass_latent_output_projection": False}
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"Incompatible RoboTwin checkpoint: {key} must be {value}")
    if processor.action_norm_type != "mean_std" or not processor.use_state:
        raise ValueError("Expected RoboTwin mean/std normalization with state conditioning")
    for section in ("state", "action"):
        schema = processor.schema[section]["aloha_agilex"]
        for side in ("left", "right"):
            arm = schema[f"{side}_arm"]
            if set(arm) - {"type", "eef_position", "eef_rotation"}:
                raise ValueError("Expected dual-arm EEF schema, not joint actions")
            for field, dim in (("eef_position", 3), ("eef_rotation", 6)):
                leaf = arm[field]
                if leaf["dim"] != dim or leaf["is_relative"] != (section == "action"):
                    raise ValueError(f"Invalid {section}/{side}/{field} schema")
            if arm["eef_rotation"]["representation"] != "rot_6d":
                raise ValueError("Expected canonical interleaved rot6d")
            grip = schema[f"{side}_gripper"]
            if grip["dim"] != 1 or grip["is_relative"] or grip["allow_relative"]:
                raise ValueError("RoboTwin grippers must be absolute open amounts")


def observation_to_inputs(observation):
    import torch
    from rynnvla.constants import RotationRepresentation as RR
    from rynnvla.utils.robot import Arm, Position, RobotState, Rotation

    fields = {}
    for side in ("left", "right"):
        pose = np.asarray(observation["endpose"][f"{side}_endpose"], dtype=np.float32)
        grip = float(observation["endpose"][f"{side}_gripper"])
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isfinite(grip):
            raise ValueError(f"{side} endpose must be finite xyz + wxyz quaternion")
        if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-3):
            raise ValueError(f"{side} endpose quaternion is not unit length")
        fields[f"{side}_arm"] = Arm(
            eef_position=Position(torch.from_numpy(pose[None, :3].copy())),
            eef_rotation=Rotation(torch.from_numpy(pose[None, 3:].copy()), RR.QUAT_WXYZ)
            .convert_rotation(RR.ROT_6D),
        )
        fields[f"{side}_gripper"] = Position(torch.tensor([[grip]]), allow_relative=False)
    images = {}
    for camera in CAMERA_SLOTS:
        image = np.asarray(observation["observation"][f"{camera}_camera"]["rgb"])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{camera} must be an RGB uint8 HWC image")
        images[camera] = np.ascontiguousarray(image)
    return RobotState(**fields), images


def decode_commands(processor, normalized_actions, state):
    import torch
    from rynnvla.constants import RobotType, RotationRepresentation as RR

    normalized_actions = torch.as_tensor(normalized_actions).detach().cpu().float()
    if normalized_actions.ndim != 2 or normalized_actions.shape[1] != 81:
        raise ValueError("Expected (T, 81) normalized actions")
    # Training subtracts the chunk's current state: position target = state + delta;
    # rotation target = R_state @ R_delta. Compose exactly once, for the whole chunk.
    action = processor.post_process(normalized_actions, RobotType.ALOHA_AGILEX, state)
    columns = []
    for side in ("left", "right"):
        arm = getattr(action, f"{side}_arm")
        grip = getattr(action, f"{side}_gripper")
        if arm.eef_position.is_relative or arm.eef_rotation.is_relative or grip.is_relative:
            raise ValueError("Simulator actions must be absolute")
        columns.extend((arm.eef_position.data, arm.eef_rotation.convert_rotation(RR.QUAT_WXYZ).data,
                        grip.data.clamp(0, 1)))
    commands = torch.cat(columns, dim=-1).numpy().astype(np.float32)
    if commands.shape != (len(normalized_actions), 16) or not np.isfinite(commands).all():
        raise ValueError("Expected finite (T, 16) dual-arm xyz/wxyz/gripper commands")
    return commands


class RynnVLARoboTwinPolicy:
    def __init__(self, args):
        import torch
        from rynnvla.inference_wrappers.rynn_brain_vla import RynnBrainVLAInferenceWrapper

        if args.get("robot_type", "aloha_agilex") != "aloha_agilex":
            raise ValueError("This adapter only supports RoboTwin aloha_agilex")
        if int(args.get("action_chunk_size", 30)) != 30:
            raise ValueError("Do not override the trained chunk size")
        self.torch = torch
        self.device = args.get("device", "cuda:0")
        self.num_steps = int(args.get("num_steps", 10))
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.wrapper = RynnBrainVLAInferenceWrapper(
            args["model_path"], dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device=self.device, local_files_only=True,
        )
        validate_checkpoint(self.wrapper.model.config, self.wrapper.processor)
        print("[rynnvla-robotwin] loaded 81D / latent608 readout / chunk30 / mean_std / EE delta", flush=True)

    def reset_model(self, seed=None):
        self.wrapper.ee_type_id = None
        if seed is not None:
            self.torch.manual_seed(int(seed))
            self.torch.cuda.manual_seed_all(int(seed))

    def update_cache(self, request):
        # get_action always prefills the observation it receives, avoiding stale views.
        return None

    def get_action(self, request):
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty instruction is required")
        state, images = observation_to_inputs(request["observation"])
        with self.torch.inference_mode():
            inputs = self.wrapper.process(prompt, images, state.to_dict(), "aloha_agilex", CAMERA_SLOTS)
            inputs = self.wrapper.collate([inputs])
            inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v
                      for k, v in inputs.items()}
            cache = self.wrapper.prefill(inputs)
            actions = self.wrapper.decode(inputs, cache, self.num_steps, robot_type="aloha_agilex")
        if tuple(actions.shape) != (1, 30, 81):
            raise ValueError(f"Unexpected action shape: {tuple(actions.shape)}")
        return decode_commands(self.wrapper.processor, actions[0], state)


def get_model(usr_args):
    return RynnVLARoboTwinPolicy(usr_args)


def reset_model(model):
    if hasattr(model, "call"):
        model.call(func_name="reset_model")
    else:
        model.reset_model()


def eval(TASK_ENV, model, observation):
    # Same native controller, rendering cadence and success termination as RoboTwin's own protocol.
    if os.environ.get("ROBOTWIN_ACTION_SPACE", "ee") != "ee":
        raise ValueError("RoboTwin evaluation must execute end-effector actions")
    horizon = int(os.environ.get("RYNNVLA_ROBOTWIN_INFER_HORIZON", "30"))
    if horizon != 30:
        raise ValueError("Primary protocol executes the complete trained chunk30")
    request = {"observation": observation, "prompt": TASK_ENV.get_instruction()}
    actions = (model.call(func_name="get_action", obs=request) if hasattr(model, "call")
               else model.get_action(request))
    for action in actions[:horizon]:
        TASK_ENV.take_action(action, action_type="ee")
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV.get_obs()
        if TASK_ENV.eval_success:
            break
    TASK_ENV.get_obs()
