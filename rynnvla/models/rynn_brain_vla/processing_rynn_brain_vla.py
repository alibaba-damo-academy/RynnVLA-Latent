# Portions of this file are derived from HuggingFace Transformers
# (https://github.com/huggingface/transformers), Copyright The HuggingFace Inc. team,
# licensed under the Apache License, Version 2.0. The license text is in LICENSE; the
# attribution is recorded in NOTICE.
# Upstream reference: src/transformers/models/qwen3_vl/processing_qwen3_vl.py

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature

from ...constants import (ALLOWED_ACTION_NORM_TYPES, VIEW_ROLE_LABEL, RobotType,
                          RotationRepresentation)
from ...utils.robot import Arm, Position, Rotation, RobotAction, RobotState


ACTION_LAYOUT: Dict[Union[str, Tuple[str, str]], Tuple[int, int]] = {
    ("left_arm",  "joint_position"): (0,   7),
    ("right_arm", "joint_position"): (7,  14),
    ("left_arm",  "eef_position"):   (14, 17),
    ("left_arm",  "eef_rotation"):   (17, 23),
    ("right_arm", "eef_position"):   (23, 26),
    ("right_arm", "eef_rotation"):   (26, 32),
    "left_gripper":                  (32, 33),
    "right_gripper":                 (33, 34),
    "left_hand":                     (34, 54),
    "right_hand":                    (54, 74),
    "torso":                         (74, 78),   # 4 DoF for Astribot humanoid waist
    "head":                          (78, 81),   # 3 DoF max (Astribot uses 2)
}
ACTION_DIM = 81
_TOP_LEVEL_NAMES = (
    "left_gripper", "right_gripper", "left_hand", "right_hand", "torso", "head",
)


def _orthogonalize_rot_6d(tensor: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt orthogonalization for 6D rotation vectors (Zhou et al. 2019).

    This codebase stores rot_6d in **interleaved** layout produced by
    ``_matrix_to_rotation_6d``: [col0[0], col1[0], col0[1], col1[1], col0[2], col1[2]].
    """
    a1 = tensor[..., 0:6:2]
    a2 = tensor[..., 1:7:2]
    e1 = F.normalize(a1, dim=-1)
    e2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(e2, dim=-1)
    out = torch.stack([e1, e2], dim=-1).flatten(-2)
    return out


def _any_leaf_data(action: Union[RobotAction, RobotState]) -> torch.Tensor:
    """Find any populated leaf tensor for dtype/device reference."""
    for _, field_value in action._fields():
        if isinstance(field_value, (Position, Rotation)):
            return field_value.data
        if isinstance(field_value, Arm):
            for _, sub_value in field_value._fields():
                return sub_value.data
    raise ValueError("RobotAction has no populated leaf field")


def get_rope_index(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> torch.Tensor:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids


class RynnBrainVLAProcessor(Qwen3VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        schema=None,
        use_state=True,
        mm_max_length: int = 10240,
        action_norm_type: str = "mean_std",
        view_label_in_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

        self.schema = schema
        self.use_state = use_state
        # Per-image token budget, matching the training token limit (DataArguments.mm_max_length).
        # Translates to max_pixels = mm_max_length * (patch_size * merge_size) ** 2 for
        # the parent Qwen2-VL image processor's smart_resize.
        self.mm_max_length = int(mm_max_length)
        # How the action chunk is scaled before the flow-matching target. Must be an __init__
        # parameter: those land in processor_config.json and survive from_pretrained, while an
        # attribute assigned after construction is dropped on save - which would train on one
        # normalization and serve with another, silently.
        # The accepted values and what each one means live in constants.ALLOWED_ACTION_NORM_TYPES,
        # shared with the checkpoint exporter so the two cannot drift apart.
        if action_norm_type not in ALLOWED_ACTION_NORM_TYPES:
            raise ValueError(
                f"Unsupported action_norm_type: {action_norm_type!r} "
                f"(expected one of {', '.join(ALLOWED_ACTION_NORM_TYPES)})."
            )
        self.action_norm_type = action_norm_type

        if schema is not None:
            self._validate_rotation_repr(schema)

        # B1 camera identity: name each observation image in the prompt with its
        # constants.VIEW_ROLES role ("front camera:", "left wrist camera:", ...) instead of
        # emitting bare, interchangeable <image> tokens. Off by default so every existing
        # checkpoint's prompt -- and therefore its token ids -- stays byte-identical.
        # Like action_norm_type this must be a ctor arg: it lands in processor_config.json
        # and survives from_pretrained, whereas an attribute set after construction is
        # dropped on save and would serve with a different prompt than it trained on.
        self.view_label_in_prompt = bool(view_label_in_prompt)

        self.state_token = "<|state_pad|>"
        self.tokenizer.add_tokens([self.state_token], special_tokens=True)
        self.state_token_id = self.tokenizer.convert_tokens_to_ids(self.state_token)

    def _validate_rotation_repr(self, schema: Dict) -> None:
        for section in ("action", "state"):
            if section not in schema:
                continue
            for robot_type, robot_schema in schema[section].items():
                self._check_rotation_leaves(robot_schema, path=f"{section}.{robot_type}")

    def _check_rotation_leaves(self, node: Dict, path: str) -> None:
        if node.get("type") == "Rotation":
            if node.get("representation") != "rot_6d":
                raise ValueError(
                    f"Schema rotation repr mismatch at '{path}': "
                    f"expected 'rot_6d', got '{node.get('representation')}'"
                )
            return
        for key, value in node.items():
            if isinstance(value, dict):
                self._check_rotation_leaves(value, path=f"{path}.{key}")

    def _process_action(
        self,
        action: Union[RobotAction, RobotState],
        schema_subtree: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten ``action`` into a fixed-layout tensor using ``schema_subtree``.

        The schema's stats drive normalization; the schema's field set drives
        which slots are filled. Returns ``(tensor, mask)`` with shape
        ``(chunk_size, ACTION_DIM)``; ``mask`` is True at filled positions.
        """
        action = action.convert_rotation(RotationRepresentation.ROT_6D)
        action = action.normalize(schema_subtree, norm_type=self.action_norm_type)

        chunk_size = len(action)
        ref = _any_leaf_data(action)
        out = torch.zeros(chunk_size, ACTION_DIM, dtype=ref.dtype, device=ref.device)
        mask = torch.zeros(chunk_size, ACTION_DIM, dtype=torch.bool, device=ref.device)

        # Pre-fill identity rotation into both eef_rotation slots so unused
        # slots stay valid in the configured representation.
        identity_rot = torch.tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=ref.dtype, device=ref.device
        )
        for arm_name in ("left_arm", "right_arm"):
            s, _ = ACTION_LAYOUT[(arm_name, "eef_rotation")]
            out[:, s:s + identity_rot.numel()] = identity_rot

        def _fill(slot: Tuple[int, int], value: torch.Tensor) -> None:
            start, end = slot
            d = value.size(1)
            assert d <= end - start, f"dim {d} exceeds slot size {end - start}"
            out[:, start:start + d] = value
            mask[:, start:start + d] = True

        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(action, arm_name)
            if arm is None:
                continue
            # Fill whichever representations the dataset provides — both eef
            # and joint may coexist (e.g. Astribot uses pose for the wrist
            # and joint angles for redundancy / joint-limit awareness).
            if arm.eef_position is not None:
                _fill(ACTION_LAYOUT[(arm_name, "eef_position")], arm.eef_position.data)
                _fill(ACTION_LAYOUT[(arm_name, "eef_rotation")], arm.eef_rotation.data)
            if arm.joint_position is not None:
                _fill(ACTION_LAYOUT[(arm_name, "joint_position")], arm.joint_position.data)

        for name in _TOP_LEVEL_NAMES:
            value = getattr(action, name, None)
            if value is not None:
                _fill(ACTION_LAYOUT[name], value.data)

        return out, mask

    @staticmethod
    def _to_pil(image: Any) -> Image.Image:
        """Coerce any common image input to a raw PIL.Image (no resize)."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255)
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def __call__(
        self,
        text: str,
        images: Dict[str, Any],
        robot_type: RobotType,
        action: Optional[RobotAction] = None,
        state: Optional[RobotState] = None,
        visual_instruction: Optional[Any] = None,
        latent_targets: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
        camera_slot_map: Optional[Dict[str, int]] = None,
        return_tensors: str = "pt",
    ):
        image_list = []

        contents = [{"type": "text", "text": "INSTRUCTION:\n"}]
        if visual_instruction is not None:
            image_list.append(self._to_pil(visual_instruction))
            contents.append({"type": "image"})
        contents.append({"type": "text", "text": text})

        contents.append({"type": "text", "text": "\n\nOBSERVATION:\n"})
        # One ordering for the prompt, the pixel list and camera_slot_ids: they must never
        # drift apart. Order stays alphabetical-by-key (NOT role order) so the token layout
        # of existing runs is unchanged; camera identity travels in the role id / label, not
        # in the position.
        camera_keys = sorted([k for k in images])
        for key in camera_keys:
            image_list.append(self._to_pil(images[key]))
            if self.view_label_in_prompt and camera_slot_map is not None:
                role = camera_slot_map.get(key)
                label = VIEW_ROLE_LABEL.get(role) if role is not None else None
                if label is not None:
                    contents.append({"type": "text", "text": f"\n{label}:\n"})
            contents.append({"type": "image"})

        if self.use_state:
            contents.append({"type": "text", "text": f"\n\nSTATE:\n{self.state_token}"})

        contents.append({"type": "text", "text": f"\n\nWhat action should the robot take?"})
        conversation = [{"role": "user", "content": contents}]

        text = self.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Per-image pixel budget, aligned with the training token limit:
        #   max_pixels = mm_max_length * factor**2, where factor = patch_size * merge_size
        # For RynnBrain-2B (patch_size=16, merge_size=2, factor=32) and mm_max_length=10240,
        # this gives max_pixels = 10240 * 1024 = 10,485,760 (matches VLM training).
        # Images whose HxW exceeds this are smart-resized (aspect-preserving, snapped to factor).
        # The parent image processor's saved min_pixels (= shortest_edge, 4*factor**2 = 4096 for
        # Qwen3-VL) remains in effect as the lower bound.
        factor = int(self.image_processor.patch_size) * int(self.image_processor.merge_size)
        max_pixels = self.mm_max_length * factor * factor
        shortest_edge = int(self.image_processor.size.get("shortest_edge", 4 * factor * factor))
        size_override = {"shortest_edge": shortest_edge, "longest_edge": max_pixels}

        model_inputs = super().__call__(
            text=text,
            images=image_list,
            return_tensors="pt",
            size=size_override,
        )

        model_inputs["position_ids"] = get_rope_index(
            input_ids=model_inputs["input_ids"],
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            attention_mask=model_inputs.get("attention_mask", None),
            spatial_merge_size=self.image_processor.merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
        )

        if action is not None:
            action_schema = self.schema["action"][robot_type.value]
            action_tensor, action_mask = self._process_action(action, action_schema)
            model_inputs["actions"] = action_tensor.unsqueeze(0)
            model_inputs["action_mask"] = action_mask.unsqueeze(0)

        if state is not None:
            state_schema = self.schema["state"][robot_type.value]
            state_tensor, _ = self._process_action(state, state_schema)
            model_inputs["states"] = state_tensor.unsqueeze(0)

        # Multi-view latent targets (hierarchical VLA, latent-action mode).
        if latent_targets is not None:
            lt = latent_targets if torch.is_tensor(latent_targets) else torch.as_tensor(latent_targets)
            model_inputs["latent_targets"] = lt.unsqueeze(0)          # (1, K, chunk, dim)
        if slot_mask is not None:
            sm = slot_mask if torch.is_tensor(slot_mask) else torch.as_tensor(slot_mask)
            model_inputs["slot_mask"] = sm.to(torch.bool).unsqueeze(0)  # (1, K)

        # Per-image camera-role tag, aligned with the image order packed above: a leading -1
        # for the visual_instruction image (not a camera view) if present, then the dataset's
        # constants.VIEW_ROLES role for each sorted camera key.
        #
        # (2026-08-28) This used to emit range(len(images)) -- the packing position -- and throw
        # camera_slot_map's values away. That made the tag meaningless across samples: the same
        # physical camera lands on a different id depending on which other cameras the episode
        # happens to carry, and datasets such as Table30 / RoboMIND2.0 vary their camera set
        # from episode to episode. The old comment justified this by noting that role ids fall
        # outside _pool_view_seeds' 0 <= slot < num_slots window when num_slots is the sample's
        # own camera count; that window is the thing being fixed (num_slots becomes the fixed
        # NUM_VIEW_SLOTS), and in the direct path seeds are not used at all.
        #
        # -1 is also the fallback for a key the dataset did not map, which makes it a no-op
        # downstream (no seed, no role embedding) rather than a silently wrong role.
        # Stored flat (num_images,) so the collator cat's it like image_grid_thw. Emitted only
        # for datasets that declare camera_slot_map.
        if camera_slot_map is not None:
            slot_ids = []
            if visual_instruction is not None:
                slot_ids.append(-1)
            slot_ids.extend(int(camera_slot_map.get(key, -1)) for key in camera_keys)
            model_inputs["camera_slot_ids"] = torch.tensor(slot_ids, dtype=torch.long)  # (num_images,)

        model_inputs = BatchFeature(
            model_inputs,
            tensor_type=return_tensors,
        )

        return model_inputs

    def get_action_mask(self, robot_type: RobotType, chunk_size: int) -> torch.Tensor:
        """Build the action_mask for a given robot type without needing action data."""
        action_schema = self.schema["action"][robot_type.value]
        mask = torch.zeros(chunk_size, ACTION_DIM, dtype=torch.bool)

        for arm_name in ("left_arm", "right_arm"):
            if arm_name not in action_schema:
                continue
            arm_schema = action_schema[arm_name]
            if "eef_position" in arm_schema:
                s, e = ACTION_LAYOUT[(arm_name, "eef_position")]
                mask[:, s:s + arm_schema["eef_position"]["dim"]] = True
                s, e = ACTION_LAYOUT[(arm_name, "eef_rotation")]
                mask[:, s:s + 6] = True
            if "joint_position" in arm_schema:
                s, e = ACTION_LAYOUT[(arm_name, "joint_position")]
                mask[:, s:s + arm_schema["joint_position"]["dim"]] = True

        for name in _TOP_LEVEL_NAMES:
            if name not in action_schema:
                continue
            s, _ = ACTION_LAYOUT[name]
            mask[:, s:s + action_schema[name]["dim"]] = True

        return mask

    def get_config_overrides(self):
        return {
            "action_dim": ACTION_DIM,
            "state_token_id": self.state_token_id,
        }

    def process_prev_actions(
        self,
        prev_actions: RobotAction,
        state: RobotState,
        robot_type: RobotType,
    ) -> torch.Tensor:
        """Convert absolute prev_actions to normalized model-space tensor.

        Sets ``allow_relative`` on each field from the schema, then computes
        relative actions (prev_actions - state) and normalizes to ``(N, 81)``.
        """
        action_schema = self.schema["action"][robot_type.value]
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(prev_actions, arm_name)
            if arm is None or arm_name not in action_schema:
                continue
            arm_schema = action_schema[arm_name]
            for field in ("joint_position", "eef_position", "eef_rotation"):
                leaf = getattr(arm, field, None)
                if leaf is not None and field in arm_schema:
                    leaf.allow_relative = arm_schema[field].get("is_relative", False)
        for name in _TOP_LEVEL_NAMES:
            leaf = getattr(prev_actions, name, None)
            if leaf is not None and name in action_schema:
                leaf.allow_relative = action_schema[name].get("is_relative", False)
        relative = prev_actions - state
        tensor, _ = self._process_action(relative, action_schema)
        return tensor

    def post_process(
        self,
        action: torch.Tensor,
        robot_type: RobotType,
        state: Optional[RobotState] = None,
    ) -> RobotAction:
        """Reverse of :meth:`_process_action`.

        The action's structure (which arms / joint-vs-eef / per-field dims and
        ``is_relative`` flags) is read from the schema rather than inferred
        from a state tensor. The optional ``state`` is only used to pick the
        final eef rotation representation; without it the rotation stays in
        ``self.eef_rotation_repr``.
        """
        assert action.ndim == 2

        eef_repr = RotationRepresentation.ROT_6D
        action_schema = self.schema["action"][robot_type.value]
        kwargs: Dict = {}

        for arm_name in ("left_arm", "right_arm"):
            if arm_name not in action_schema:
                continue
            arm_schema = action_schema[arm_name]
            arm_kwargs: Dict = {}

            # Both eef and joint may coexist for an arm (mirror of _process_action).
            if "eef_position" in arm_schema:
                pos_leaf = arm_schema["eef_position"]
                ps, _ = ACTION_LAYOUT[(arm_name, "eef_position")]
                arm_kwargs["eef_position"] = Position(
                    data=action[:, ps:ps + pos_leaf["dim"]],
                    is_relative=pos_leaf["is_relative"],
                    allow_relative=pos_leaf["allow_relative"],
                )
                rot_leaf = arm_schema["eef_rotation"]
                rs, _ = ACTION_LAYOUT[(arm_name, "eef_rotation")]
                arm_kwargs["eef_rotation"] = Rotation(
                    data=action[:, rs:rs + eef_repr.dim],
                    representation=eef_repr,
                    is_relative=rot_leaf["is_relative"],
                    allow_relative=rot_leaf["allow_relative"],
                )
            if "joint_position" in arm_schema:
                joint_leaf = arm_schema["joint_position"]
                s, _ = ACTION_LAYOUT[(arm_name, "joint_position")]
                arm_kwargs["joint_position"] = Position(
                    data=action[:, s:s + joint_leaf["dim"]],
                    is_relative=joint_leaf["is_relative"],
                    allow_relative=joint_leaf["allow_relative"],
                )

            kwargs[arm_name] = Arm(**arm_kwargs)

        for name, (s, _) in ACTION_LAYOUT.items():
            if not isinstance(name, str):
                continue
            if name not in action_schema:
                continue
            leaf = action_schema[name]
            kwargs[name] = Position(
                data=action[:, s:s + leaf["dim"]],
                is_relative=leaf["is_relative"],
                allow_relative=leaf["allow_relative"],
            )

        formatted_action = RobotAction(**kwargs)
        formatted_action = formatted_action.denormalize(action_schema, norm_type=self.action_norm_type)

        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(formatted_action, arm_name)
            if arm is not None and arm.eef_rotation is not None:
                arm.eef_rotation.data = _orthogonalize_rot_6d(arm.eef_rotation.data)

        if state is not None:
            formatted_action = formatted_action + state

            target_repr: Optional[RotationRepresentation] = None
            for arm_name in ("left_arm", "right_arm"):
                state_arm = getattr(state, arm_name, None)
                if state_arm is not None and state_arm.eef_rotation is not None:
                    target_repr = state_arm.eef_rotation.representation
                    break
            if target_repr is not None:
                formatted_action = formatted_action.convert_rotation(target_repr)

        return formatted_action
