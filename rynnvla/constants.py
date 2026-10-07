import os
from enum import Enum


_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    ),
    "rynnvla"
)
CACHE_DIR = os.environ.get("RYNNVLA_CACHE_DIR", _DEFAULT_CACHE_DIR)


# ==========================================================================
# Resume-argument validation.
#
# Settings that silently reshape training if changed on a resume. The scheduler lambda is built
# by create_scheduler from the CURRENT args before the checkpoint's state_dict is loaded over it,
# so max_steps / warmup_steps / lr_scheduler_type / min_lr_rate / the two rates quietly redraw the
# LR curve. gradient_accumulation_steps and data_mixture change the epoch length that the resume
# batch-skip math divides by. sequence_packing changes the collator, frozen_parameters changes
# requires_grad (hence the EMA shadow key set and the optimizer groups), and deepspeed changes the
# ZeRO stage the optimizer checkpoint was sharded for.
#
# Lives here rather than in training/trainer.py because api/train.py:resolve_resume reads it too,
# and that function is unit-tested standalone -- importing it from the trainer would drag in
# deepspeed.
# ==========================================================================
RESUME_ARGS_NAME = "resume_args.json"
RESUME_CRITICAL_FIELDS = (
    "max_steps",
    "warmup_steps",
    "lr_scheduler_type",
    "min_lr_rate",
    "learning_rate",
    "action_head_lr",
    "action_head_modules",
    "transfer_lr",
    "transfer_modules",
    "gradient_accumulation_steps",
    "micro_batch_size",
    "sampler_shuffle",
    "seed",
    "dataloader_num_workers",
    "data_mixture",
    "sequence_packing",
    "frozen_parameters",
    "deepspeed",
)


class RobotType(Enum):
    DEFAULT = "default"
    FRANKA = "franka"
    UR5 = "ur5"
    ALOHA_AGILEX = "aloha_agilex"
    AGILEX_COBOT_MAGIC = "agilex_cobot_magic"
    AGILEX_PIPER = "agilex_piper"
    ARX_X5 = "arx_x5"
    PIPER = "piper"
    ARX_LIFT2 = "arx_lift2"
    LEROBOT = "lerobot"
    AGIBOT_G2 = "agibot_g2"
    AGIBOT_G1 = "agibot_g1"
    SPLIT_ALOHA = "split_aloha"
    GALAXEA_R1_LITE = "galaxea_r1_lite"
    ARK = "ark"
    TIANYI = "tianyi"
    TIENKUNG = "tienkung"
    UR5_DEX = "ur5_dex"
    TIANJI_WUJI = "tianji_wuji"
    ASTRIBOT = "astribot"


# Robot type → EE (end-effector) type ID mapping.
# Classifies robots into 5 EE categories for embodiment-aware conditioning:
#   0: DEFAULT (default/unknown)
#   1: Single-arm with gripper (Franka, UR5, Piper, etc.)
#   2: Dual-arm with grippers (ALOHA, AgileX Cobot, etc.)
#   3: Dexterous hands (UR5-Dex, Tianji-Wuji)
#   4: Mobile robots / other (AGIBOT G1/G2)
ROBOT_TYPE_TO_EE_ID = {
    RobotType.DEFAULT: 0,
    RobotType.FRANKA: 1,
    RobotType.UR5: 1,
    RobotType.PIPER: 1,
    RobotType.ARX_X5: 1,
    RobotType.ARX_LIFT2: 1,
    RobotType.LEROBOT: 1,
    RobotType.GALAXEA_R1_LITE: 1,
    RobotType.ARK: 1,
    RobotType.TIANYI: 1,
    RobotType.ALOHA_AGILEX: 2,
    RobotType.AGILEX_COBOT_MAGIC: 2,
    RobotType.AGILEX_PIPER: 2,
    RobotType.SPLIT_ALOHA: 2,
    RobotType.TIENKUNG: 2,
    RobotType.ASTRIBOT: 2,
    RobotType.UR5_DEX: 3,
    RobotType.TIANJI_WUJI: 3,
    RobotType.AGIBOT_G2: 4,
    RobotType.AGIBOT_G1: 4,
}


# ==========================================================================
# Canonical view-role taxonomy (multi-view slot-based latent action).
#
# Global, FIXED contract: slot i <-> a semantic camera role. Every dataset's
# per-camera `camera_slot_map` (camera_name -> slot_id) must map into this
# enumeration so slots carry consistent semantics across embodiments. The VLA
# outputs one latent action per slot; a sample only activates the slots for the
# cameras it actually has. Keep this stable — changing ids invalidates trained
# slot embeddings and any cached per-view latents.
# ==========================================================================
VIEW_ROLES = {
    0: "head",          # carrier own primary view: head-mounted (human) or head/torso (robot)
    1: "left_wrist",
    2: "right_wrist",
    3: "front_third",   # fixed third-person / overview ("global" in the RynnVLA-Base manifests)
    4: "side_left",     # fixed third-person side, left of the workspace
    5: "side_right",    # fixed third-person side, right of the workspace
}
VIEW_ROLE_TO_ID = {name: idx for idx, name in VIEW_ROLES.items()}
NUM_VIEW_SLOTS = len(VIEW_ROLES)  # K = 6

# The side pair is split, not merged, for the same reason the wrist pair is: a role has to
# identify ONE physical camera, otherwise two cameras in the same episode claim one slot and
# the slot axis stops being a function of the camera. This is not hypothetical -- at least one
# source ships camera_left_external and camera_right_external together in the same episode,
# which a single `side` role would collide. The old dense sample-order packing hid it (two
# same-role cameras simply took consecutive slots); moving LatentPretrainDataset onto this axis
# is what surfaced it.

# Manifest camera name -> canonical role. The RynnVLA-Base dataset JSONs
# (data/RynnVLA-Base/*.json) key their `views` dict by an already-normalised camera name;
# a full scan of the pretraining corpus yields exactly these five names and at most five
# cameras per episode, so this map is total over the corpus and the old `top` role never occurs.
#
# This map is injective, so no episode can collide here. That is a property of THIS table,
# not of the role axis: a many-to-one table can collide, and
# latent_pretrain._VIEW_ROLE used to (see the side_left/side_right note above). Any new
# camera_name -> role table must be checked for injectivity per source.
#
# The single generic `side` camera takes the primary (left) side slot, following the same
# convention LiberoPlusDataset uses when it sends a single-arm eye-in-hand camera to
# `left_wrist` rather than inventing an ambiguous "the one wrist" role.
VIEW_NAME_TO_ROLE = {
    "head": "head",
    "wrist_left": "left_wrist",
    "wrist_right": "right_wrist",
    "global": "front_third",
    "side": "side_left",
}

# Short human-readable camera label, used when the prompt carries an explicit camera tag
# (processor `view_label_in_prompt`). Kept separate from the role names so the wording can
# change without touching the slot contract.
# Camera name -> role id for the INFERENCE servers, which name cameras themselves rather
# than reading a dataset class. It must stay identical to the train-time `camera_slot_map`
# of the corresponding dataset, otherwise a model trained with camera identity
# (view_label_in_prompt / use_view_role_embedding) is shown a different role at eval than
# at train -- a silent train/eval skew, not a crash. tests/test_camera_roles.py pins this.
#   head / left / right  <- RoboTwinDataset.camera_slot_map
#   front / wrist        <- LiberoPlusDataset.camera_slot_map (agentview / eye-in-hand)
SERVER_CAMERA_SLOT_MAP = {
    "head": VIEW_ROLE_TO_ID["head"],
    "left": VIEW_ROLE_TO_ID["left_wrist"],
    "right": VIEW_ROLE_TO_ID["right_wrist"],
    "front": VIEW_ROLE_TO_ID["front_third"],
    "wrist": VIEW_ROLE_TO_ID["left_wrist"],
}

VIEW_ROLE_LABEL = {
    0: "head camera",
    1: "left wrist camera",
    2: "right wrist camera",
    3: "front camera",
    4: "left side camera",
    5: "right side camera",
}


# ==========================================================================
#  Action normalization types
# ==========================================================================
# The single list both the processor and the checkpoint exporter validate against.
#
# `action_norm_type` is persisted in processor_config.json and applied symmetrically by
# utils/robot.py:_norm_coeffs on normalize and denormalize, so any type here makes an exported
# checkpoint self-consistent -- including one that is not the formal recipe's. The two sides
# have to agree exactly:
#   * exporter stricter than processor -> an arm trains for 30k steps and then cannot be
#     exported for evaluation;
#   * exporter looser -> it publishes a checkpoint whose normalization nothing can read back.
# Keeping one tuple is what stops that drift; utils/robot.py raises on anything outside it.
#
# "min_max_sym" is the [-1, 1] form the LIBERO GR00T-style head uses and needs no statistics
# beyond the min/max the dataset already stores; plain "min_max" lands on [0, 2].
# "q01_q99" maps the central 98% onto [-1, 1] using per-leaf reservoir quantiles
# (vla_datasets/base.py:_finalize_leaf). It does NOT clip: values outside [q01, q99] land
# outside [-1, 1], and heavy-tailed dims get a larger normalized range than under
# min_max_sym. Pick it for parity with a recipe that used it, not because it bounds the target.
ALLOWED_ACTION_NORM_TYPES = ("mean_std", "min_max", "min_max_sym", "q01_q99")


class RotationRepresentation(Enum):
    EULER_XYZ = ("euler_xyz", 3)
    EULER_ZYX = ("euler_zyx", 3)
    QUAT_XYZW = ("quat_xyzw", 4)
    QUAT_WXYZ = ("quat_wxyz", 4)
    ROT_6D = ("rot_6d", 6)
    ROT_VEC = ("rot_vec", 3)

    def __new__(cls, label, dim):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.dim = dim
        return obj
