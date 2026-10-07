from ..registry import DATASET_REGISTRY
from .vla_datasets import BaseVLADataset
from ..constants import RotationRepresentation
from .concat import ConcatDataset


def _build_dataset(
    data_type: str,
    data_path: str,
    model_max_length: int,
    mm_max_length: int,
    fps: int,
    max_frames: int,
    action_chunk_size: int,
    use_delta_action: bool,
    eef_rotation_repr: RotationRepresentation,
    action_only: bool,
    target_fps=None,
    **kwargs,
):
    dataset_class = DATASET_REGISTRY[data_type]
    if issubclass(dataset_class, BaseVLADataset):
        return dataset_class(
            data_path=data_path,
            action_chunk_size=action_chunk_size,
            use_delta_action=use_delta_action,
            eef_rotation_repr=eef_rotation_repr,
            action_only=action_only,
            target_fps=target_fps,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset type: {data_type}")


def build_dataset(args):
    defaults = {
        # VLM processing configs
        "model_max_length": args.model_max_length,
        "mm_max_length": args.mm_max_length,
        "fps": args.fps,
        "max_frames": args.max_frames,
        # VLA processing configs
        "action_chunk_size": args.action_chunk_size,
        "use_delta_action": args.use_delta_action,
        "eef_rotation_repr": args.eef_rotation_repr,
        "action_only": args.action_only,
        "target_fps": args.target_fps,
        "num_view_slots": getattr(args, "num_view_slots", 1),
        # latent_action_dim is a MODEL_DATA_FIELDS entry, so api/train.py already syncs it into
        # config_overrides and the model is built at the recipe's width. It was missing here, so
        # LatentPretrainDataset fell back to its own hardcoded 256 and the two sides silently
        # disagreed -- the run then died in load_latent_stats' width assertion rather than at
        # argument validation. BaseVLADataset absorbs unknown kwargs, so non-latent datasets
        # are unaffected.
        "latent_action_dim": args.latent_action_dim,
        "use_visual_augmentation": getattr(args, "use_visual_augmentation", False),
        "chunk_overlap_ratio": getattr(args, "chunk_overlap_ratio", 0.0),
        "emit_teacher_images": getattr(args, "emit_teacher_images", False),
    }

    if args.data_mixture is None:
        data_mixture = [
            {"data_type": args.data_type, "data_path": args.data_path},
        ]
    else:
        data_mixture = args.data_mixture

    datasets = []
    weights = []
    for data_source in data_mixture:
        assert "data_type" in data_source and "data_path" in data_source
        source_cfg = dict(data_source)
        weights.append(float(source_cfg.pop("weight", 1.0)))
        datasets.append(_build_dataset(**{**defaults, **source_cfg}))

    if len(datasets) == 1:
        return datasets[0]

    use_weights = len(set(weights)) > 1
    return ConcatDataset(datasets, weights=weights if use_weights else None)
