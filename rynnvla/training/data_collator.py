from typing import Any, Dict, List

import torch
from transformers import ProcessorMixin

from ..utils import context_parallel


class DataCollator(object):
    def __init__(
        self,
        processor: ProcessorMixin,
        sequence_packing: bool,
    ):
        self.processor = processor
        self.sequence_packing = sequence_packing

    @staticmethod
    def _collate_teacher_inputs(instances, batch):
        present = ["teacher_images" in instance for instance in instances]
        if not any(present):
            return
        if not all(present):
            raise ValueError("teacher_images must be present for every sample in a batch")

        teacher_images = []
        teacher_grid_indices = []
        primary_teacher_indices = []
        teacher_offset = 0
        grid_offset = 0
        for sample_idx, instance in enumerate(instances):
            images = instance["teacher_images"]
            mapping = instance.get("teacher_image_grid_indices")
            primary = instance.get("primary_teacher_index")
            grids = instance.get("image_grid_thw")
            if mapping is None or primary is None or grids is None:
                raise ValueError(
                    f"sample {sample_idx} teacher inputs require teacher_image_grid_indices, "
                    "primary_teacher_index, and image_grid_thw"
                )
            mapping = mapping.reshape(-1).to(torch.long)
            primary = int(primary.item())
            if mapping.numel() != images.size(0):
                raise ValueError(
                    f"sample {sample_idx} has {images.size(0)} teacher images but "
                    f"{mapping.numel()} grid indices"
                )
            if mapping.numel() and (mapping.min() < 0 or mapping.max() >= grids.size(0)):
                raise ValueError(
                    f"sample {sample_idx} teacher grid indices {mapping.tolist()} are out of "
                    f"range for {grids.size(0)} grids"
                )
            if not 0 <= primary < images.size(0):
                raise ValueError(
                    f"sample {sample_idx} primary_teacher_index={primary} is out of range "
                    f"for {images.size(0)} teacher images"
                )
            teacher_images.append(images)
            teacher_grid_indices.append(mapping + grid_offset)
            primary_teacher_indices.append(primary + teacher_offset)
            teacher_offset += images.size(0)
            grid_offset += grids.size(0)

        batch["teacher_images"] = torch.cat(teacher_images, dim=0)
        batch["teacher_image_grid_indices"] = torch.cat(teacher_grid_indices, dim=0)
        batch["primary_teacher_indices"] = torch.tensor(primary_teacher_indices, dtype=torch.long)

    @staticmethod
    def _collate_multiview(instances, batch):
        """Multi-view latent fields with variable per-sample camera count N: pad
        slot_mask (1, N) and latent_targets (1, N, chunk, dim) to the batch max N.
        Padded slots are False in the mask, so they are key-masked and loss-masked."""
        if "slot_mask" in instances[0]:
            masks = [instance["slot_mask"][0] for instance in instances]
            n_max = max(m.size(0) for m in masks)
            padded = torch.zeros(len(masks), n_max, dtype=torch.bool)
            for i, m in enumerate(masks):
                padded[i, :m.size(0)] = m
            batch["slot_mask"] = padded
        if "latent_targets" in instances[0]:
            lts = [instance["latent_targets"][0] for instance in instances]
            n_max = max(t.size(0) for t in lts)
            padded = torch.zeros(len(lts), n_max, *lts[0].shape[1:], dtype=lts[0].dtype)
            for i, t in enumerate(lts):
                padded[i, :t.size(0)] = t
            batch["latent_targets"] = padded
        if "camera_slot_ids" in instances[0]:
            batch["camera_slot_ids"] = torch.cat([instance["camera_slot_ids"] for instance in instances], dim=0)

    def _collate_mm_inputs(self, instances):
        mm_input_names = set(
            self.processor.image_processor.model_input_names + self.processor.video_processor.model_input_names
        )

        mm_inputs = {}
        for key in mm_input_names:
            data_list = [instance[key] for instance in instances if key in instance]
            if len(data_list) > 0:
                mm_inputs[key] = torch.cat(data_list, dim=0)

        return mm_inputs

    def _collate_fn_packing(self, instances):
        input_ids_list, position_ids_list, labels_list = [], [], []

        cu_seq_lens = [0]
        max_length = 0

        for instance in instances:
            input_ids = instance.get("input_ids")
            position_ids = instance.get(
                "position_ids", torch.arange(instance["input_ids"].size(-1)).unsqueeze(0)
            )
            labels = instance.get("labels", None)

            if "labels" in instance:
                labels = instance["labels"].clone()
            else:
                labels = torch.full_like(input_ids, fill_value=-100, dtype=torch.long)
            labels[..., 0] = -100

            input_ids, _, position_ids, labels = context_parallel.pad_sequence(
                input_ids,
                position_ids=position_ids,
                labels=labels,
            )

            input_ids_list.append(input_ids)
            position_ids_list.append(position_ids)
            labels_list.append(labels)

            seq_len = input_ids.size(-1)
            cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
            max_length = max(max_length, seq_len)

        cu_seq_lens = torch.as_tensor(cu_seq_lens, dtype=torch.int32)

        batch = {
            "input_ids": torch.cat(input_ids_list, dim=-1),
            "position_ids": torch.cat(position_ids_list, dim=-1),
            "labels": torch.cat(labels_list, dim=-1),
            **self._collate_mm_inputs(instances),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length,
        }

        if "actions" in instances[0]:
            batch["actions"] = torch.cat([instance["actions"] for instance in instances], dim=0)

        if "action_mask" in instances[0]:
            batch["action_mask"] = torch.cat([instance["action_mask"] for instance in instances], dim=0)

        if "states" in instances[0]:
            batch["states"] = torch.cat([instance["states"] for instance in instances], dim=0)

        if "data_index" in instances[0]:
            batch["data_indices"] = [instance["data_index"] for instance in instances]

        # In-context history conditioning: past K steps of (state, action) pairs.
        if "history_states" in instances[0]:
            batch["history_states"] = torch.cat([instance["history_states"] for instance in instances], dim=0)
        if "history_actions" in instances[0]:
            batch["history_actions"] = torch.cat([instance["history_actions"] for instance in instances], dim=0)

        # EE (end-effector) type embedding: per-sample robot category ID.
        if "ee_type_id" in instances[0]:
            batch["ee_type_id"] = torch.cat([instance["ee_type_id"] for instance in instances], dim=0)

        # Auxiliary depth prediction: current + future depth targets and masks.
        if "depth_target" in instances[0]:
            batch["depth_target"] = torch.cat([instance["depth_target"] for instance in instances], dim=0)
        if "depth_mask" in instances[0]:
            batch["depth_mask"] = torch.cat([instance["depth_mask"] for instance in instances], dim=0)
        if "future_depth_target" in instances[0]:
            batch["future_depth_target"] = torch.cat([instance["future_depth_target"] for instance in instances], dim=0)
        if "future_depth_mask" in instances[0]:
            batch["future_depth_mask"] = torch.cat([instance["future_depth_mask"] for instance in instances], dim=0)

        self._collate_teacher_inputs(instances, batch)

        # Multi-view latent-action targets + per-image slot tags (variable N padded).
        self._collate_multiview(instances, batch)

        return batch

    def _collate_fn_padding(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"][0] for instance in instances],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
            padding_side="left",
        )

        if "attention_mask" in instances[0]:
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [instance["attention_mask"][0] for instance in instances],
                batch_first=True,
                padding_value=0,
                padding_side="left",
            )
        else:
            attention_mask = input_ids != self.processor.tokenizer.pad_token_id

        if "position_ids" in instances[0]:
            if instances[0]["position_ids"].ndim == 3:
                position_ids = torch.nn.utils.rnn.pad_sequence(
                    [instance["position_ids"][:, 0].transpose(0, 1) for instance in instances],
                    batch_first=True,
                    padding_value=1,
                    padding_side="left",
                ).permute(2, 0, 1)
            else:
                position_ids = torch.nn.utils.rnn.pad_sequence(
                    [instance["position_ids"][0] for instance in instances],
                    batch_first=True,
                    padding_value=1,
                    padding_side="left",
                )
        else:
            assert attention_mask.ndim == 2
            position_ids = attention_mask.cumsum(-1) - 1

        if "labels" in instances[0]:
            labels = torch.nn.utils.rnn.pad_sequence(
                [instance["labels"][0] for instance in instances],
                batch_first=True,
                padding_value=-100,
                padding_side="left",
            )
        else:
            labels = torch.full_like(input_ids, fill_value=-100, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            **self._collate_mm_inputs(instances),
        }

        if "actions" in instances[0]:
            batch["actions"] = torch.cat([instance["actions"] for instance in instances], dim=0)

        if "action_mask" in instances[0]:
            batch["action_mask"] = torch.cat([instance["action_mask"] for instance in instances], dim=0)

        if "states" in instances[0]:
            batch["states"] = torch.cat([instance["states"] for instance in instances], dim=0)

        if "data_index" in instances[0]:
            batch["data_indices"] = [instance["data_index"] for instance in instances]

        # In-context history conditioning: past K steps of (state, action) pairs.
        if "history_states" in instances[0]:
            batch["history_states"] = torch.cat([instance["history_states"] for instance in instances], dim=0)
        if "history_actions" in instances[0]:
            batch["history_actions"] = torch.cat([instance["history_actions"] for instance in instances], dim=0)

        # EE (end-effector) type embedding: per-sample robot category ID.
        if "ee_type_id" in instances[0]:
            batch["ee_type_id"] = torch.cat([instance["ee_type_id"] for instance in instances], dim=0)

        # Auxiliary depth prediction: current + future depth targets and masks.
        if "depth_target" in instances[0]:
            batch["depth_target"] = torch.cat([instance["depth_target"] for instance in instances], dim=0)
        if "depth_mask" in instances[0]:
            batch["depth_mask"] = torch.cat([instance["depth_mask"] for instance in instances], dim=0)
        if "future_depth_target" in instances[0]:
            batch["future_depth_target"] = torch.cat([instance["future_depth_target"] for instance in instances], dim=0)
        if "future_depth_mask" in instances[0]:
            batch["future_depth_mask"] = torch.cat([instance["future_depth_mask"] for instance in instances], dim=0)

        self._collate_teacher_inputs(instances, batch)

        # Multi-view latent-action targets + per-image slot tags (variable N padded).
        self._collate_multiview(instances, batch)

        return batch

    def __call__(self, instances: List[Dict[str, Any]]):
        if self.sequence_packing:
            batch = self._collate_fn_packing(instances)
        else:
            batch = self._collate_fn_padding(instances)
        batch["use_cache"] = False
        return batch
