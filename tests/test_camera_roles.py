"""Camera-identity (view role) wiring.

Three mechanisms tag which physical camera an image came from:
  B1 ``view_label_in_prompt``    -- a text label before each image in the VLM prompt
  B2 ``use_view_role_embedding`` -- a zero-init embedding added to each image's tokens
  V4 ``use_view_cond_slots``     -- one expert-suffix context token per camera ROLE
Both are keyed by ``constants.VIEW_ROLES`` ids that reach the processor through a
dataset's / server's ``camera_slot_map``. The tests below pin the two properties that
make the A/B against the existing baseline meaningful:
  1. train-time and eval-time maps agree (no silent train/eval skew), and
  2. with both flags off nothing changes at all.
"""

import pytest
import torch

from rynnvla.constants import (
    NUM_VIEW_SLOTS,
    SERVER_CAMERA_SLOT_MAP,
    VIEW_NAME_TO_ROLE,
    VIEW_ROLE_LABEL,
    VIEW_ROLES,
    VIEW_ROLE_TO_ID,
)
from rynnvla.datasets.vla_datasets.libero_plus import LiberoPlusDataset
from rynnvla.datasets.vla_datasets.robotwin import RoboTwinDataset
from rynnvla.datasets.vla_datasets.vlabench import VLABenchDataset


def test_role_tables_are_mutually_consistent():
    assert NUM_VIEW_SLOTS == len(VIEW_ROLES)
    assert set(VIEW_ROLE_TO_ID.values()) == set(range(NUM_VIEW_SLOTS))
    # every manifest camera name resolves to a real role
    assert set(VIEW_NAME_TO_ROLE.values()) <= set(VIEW_ROLES.values())
    # every role has a prompt label, so B1 never silently drops a camera
    assert set(VIEW_ROLE_LABEL) == set(VIEW_ROLES)


@pytest.mark.parametrize(
    "dataset_cls",
    [LiberoPlusDataset, RoboTwinDataset],
    ids=lambda c: c.__name__,
)
def test_server_map_matches_dataset_map(dataset_cls):
    """The eval-time camera->role map must equal the train-time one.

    If these drift, a model trained with camera identity is shown a different role at
    eval than at train. That is a silent accuracy loss, not a crash, so it is pinned.
    """
    for name, role in dataset_cls.camera_slot_map.items():
        assert name in SERVER_CAMERA_SLOT_MAP, f"{dataset_cls.__name__} camera {name!r} unserved"
        assert SERVER_CAMERA_SLOT_MAP[name] == role, name


def test_libero_agentview_is_not_the_head_role():
    """LIBERO's agentview is a fixed external rig, i.e. the manifests' "global".

    Role 0 is the carrier's own view (head-mounted / on-robot), which pretraining learns
    from millions of egocentric episodes; routing a static third-person rig into it would
    contradict those semantics.
    """
    assert LiberoPlusDataset.camera_slot_map["front"] == VIEW_ROLE_TO_ID["front_third"]
    assert LiberoPlusDataset.camera_slot_map["wrist"] == VIEW_ROLE_TO_ID["left_wrist"]
    assert VIEW_NAME_TO_ROLE["global"] == "front_third"


def test_role_ids_are_not_packing_positions():
    """Regression: camera_slot_ids used to be range(len(images)).

    Under that scheme LIBERO's eye-in-hand camera would be id 1 here but id 0 on a
    wrist-only episode, so the tag meant nothing across samples.
    """
    ids = [LiberoPlusDataset.camera_slot_map[k] for k in sorted(LiberoPlusDataset.camera_slot_map)]
    assert ids != list(range(len(ids)))
    assert ids == [3, 1]


def test_vlabench_camera_roles_are_intended_and_injective():
    """VLABench's three cameras must land on distinct, intended view roles.

    ``camera_slot_map`` is built per-instance (it depends on the selected cameras), so the
    class-level ``_CAMERA_SLOTS`` is what pins the mapping. ``second_image`` is a single
    generic side rig -> side_left, following constants.VIEW_NAME_TO_ROLE["side"] and the same
    convention LiberoPlusDataset uses when it sends a single eye-in-hand camera to left_wrist.
    Injectivity matters (see the note above constants.VIEW_NAME_TO_ROLE) -- two cameras claiming
    one role stop the slot axis being a function of the camera.

    VLABench eval does not read ``SERVER_CAMERA_SLOT_MAP``: the policy server goes through
    ``inference_wrappers.rynn_brain_vla``, which carries its own VLABench camera -> role table.
    ``test_vlabench_inference_adapter.py::test_camera_map_matches_training`` pins that table
    equal to the one asserted here, which is the train/eval-skew half of the contract. That is
    why VLABenchDataset still cannot join test_server_map_matches_dataset_map's parametrize
    list: that test pins the LIBERO/RoboTwin server's table, whose camera names are different.
    """
    slots = VLABenchDataset._CAMERA_SLOTS
    assert slots["image"] == VIEW_ROLE_TO_ID["front_third"]
    assert slots["second_image"] == VIEW_ROLE_TO_ID["side_left"]
    assert slots["wrist_image"] == VIEW_ROLE_TO_ID["left_wrist"]
    assert set(slots.values()) <= set(VIEW_ROLE_TO_ID.values())
    assert len(set(slots.values())) == len(slots)


def test_robotwin_camera_roles_are_intended_and_injective():
    """RoboTwin carries a head camera plus two wrist cameras on distinct roles.

    head -> the robot's own primary view (role 0); left/right -> the two wrist cameras
    (left_wrist/right_wrist). The output names deliberately match SERVER_CAMERA_SLOT_MAP, so the
    eval server tags cameras identically to train without a new constants entry -- which is why
    RoboTwinDataset CAN join test_server_map_matches_dataset_map while VLABench (unserved names)
    cannot.
    """
    slots = RoboTwinDataset.camera_slot_map
    assert slots["head"] == VIEW_ROLE_TO_ID["head"]
    assert slots["left"] == VIEW_ROLE_TO_ID["left_wrist"]
    assert slots["right"] == VIEW_ROLE_TO_ID["right_wrist"]
    assert set(slots.values()) <= set(VIEW_ROLE_TO_ID.values())
    assert len(set(slots.values())) == len(slots)


def _tiny_role_embedding_shim(merge=2, role_emb=None):
    from rynnvla.models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLAModel

    class _Shim:
        config = type("C", (), {"vision_config": type("V", (), {"spatial_merge_size": merge})()})()
        _image_token_spans = RynnBrainVLAModel._image_token_spans
        _add_view_role_embedding = RynnBrainVLAModel._add_view_role_embedding

    shim = _Shim()
    shim.view_role_emb = role_emb
    return shim


def _fake_prefix(n_images=2, tokens_per_image=4, hidden=8, length=24):
    torch.manual_seed(0)
    grid = torch.tensor([[1, 4, 4]] * n_images)  # prod // merge**2 == 4
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 3:3 + tokens_per_image * n_images] = True
    return torch.randn(1, length, hidden), mask, grid


def test_role_embedding_is_a_no_op_when_disabled():
    embeds, mask, grid = _fake_prefix()
    shim = _tiny_role_embedding_shim(role_emb=None)
    out = shim._add_view_role_embedding(embeds, mask, grid, torch.tensor([3, 1]))
    assert torch.equal(out, embeds)


def test_role_embedding_is_a_no_op_at_zero_init():
    """Enabling the flag must not perturb step 0, so an A/B differs only by training."""
    embeds, mask, grid = _fake_prefix()
    emb = torch.nn.Embedding(NUM_VIEW_SLOTS, embeds.size(-1))
    torch.nn.init.zeros_(emb.weight)
    shim = _tiny_role_embedding_shim(role_emb=emb)
    assert torch.equal(shim._add_view_role_embedding(embeds, mask, grid, torch.tensor([3, 1])), embeds)


def test_role_embedding_lands_only_on_its_own_image_tokens():
    embeds, mask, grid = _fake_prefix()
    emb = torch.nn.Embedding(NUM_VIEW_SLOTS, embeds.size(-1))
    torch.nn.init.normal_(emb.weight, std=1.0)
    shim = _tiny_role_embedding_shim(role_emb=emb)
    delta = shim._add_view_role_embedding(embeds, mask, grid, torch.tensor([3, 1])) - embeds

    assert delta[~mask].abs().max().item() == 0.0, "text/state tokens must be untouched"
    img_pos = mask[0].nonzero(as_tuple=True)[0]
    assert torch.allclose(delta[0, img_pos[:4]], emb.weight[3].expand(4, -1), atol=1e-6)
    assert torch.allclose(delta[0, img_pos[4:]], emb.weight[1].expand(4, -1), atol=1e-6)


def test_unmapped_camera_is_inert_not_wrong():
    """-1 (camera the dataset did not map) must add nothing rather than pick role 0."""
    embeds, mask, grid = _fake_prefix()
    emb = torch.nn.Embedding(NUM_VIEW_SLOTS, embeds.size(-1))
    torch.nn.init.normal_(emb.weight, std=1.0)
    shim = _tiny_role_embedding_shim(role_emb=emb)
    delta = shim._add_view_role_embedding(embeds, mask, grid, torch.tensor([-1, 1])) - embeds
    img_pos = mask[0].nonzero(as_tuple=True)[0]
    assert delta[0, img_pos[:4]].abs().max().item() == 0.0
    assert torch.allclose(delta[0, img_pos[4:]], emb.weight[1].expand(4, -1), atol=1e-6)


def test_grid_mismatch_disables_the_feature_instead_of_misaligning():
    embeds, mask, grid = _fake_prefix()
    emb = torch.nn.Embedding(NUM_VIEW_SLOTS, embeds.size(-1))
    torch.nn.init.normal_(emb.weight, std=1.0)
    shim = _tiny_role_embedding_shim(role_emb=emb)
    bad_grid = grid[:1]  # claims 1 image, mask holds 2
    assert torch.equal(shim._add_view_role_embedding(embeds, mask, bad_grid, torch.tensor([3, 1])), embeds)


def test_zero_init_survives_the_meta_device_build():
    """The __init__ zeroing is NOT what makes view_role_emb start at zero.

    Training builds the model on the meta device (models/__init__.py:_init_empty_params),
    where every nn.init.* in __init__ is a silent no-op; the real values are drawn later by
    _init_missing_weights, which dispatches on module TYPE and hands any nn.Embedding a
    normal_ draw. Only the ``_is_zero_init`` marker survives that path. Without it, turning
    the flag on perturbs every image token from step 0 and the A/B against Arm L would
    measure that perturbation instead of camera identity.

    slot_seed_proj was the cautionary example: its "starts at zero" comment is violated in
    every checkpoint shipped before 2026-08-21 (xl_lingbot: slot_seed_proj.weight absmax
    1.0e-1). It now carries the same marker; both are asserted below -- but only in the
    ADDITIVE form, see test_v4_does_not_zero_init_the_camera_projection.
    """
    import torch.nn as nn

    from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig
    from rynnvla.models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLAModel

    cfg = RynnBrainVLAConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         vocab_size=64,
                         rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]}),
        vision_config=dict(hidden_size=32, intermediate_size=64, depth=1, num_heads=2,
                           out_hidden_size=32, patch_size=14, spatial_merge_size=2,
                           temporal_patch_size=2, in_channels=3, num_position_embeddings=16),
        action_dim=7, action_chunk_size=2, action_head_type="expert",
        expert_hidden_size=32, expert_intermediate_size=64, expert_num_attention_heads=2,
        expert_num_foresight_tokens=0, state_token_id=9, use_view_role_embedding=True,
    )
    model = RynnBrainVLAModel(cfg)

    # slot_seed_proj rides the same mechanism: it is the K-slot camera-identity injection
    # for the latent path, and its zero start is what makes all view slots begin symmetric.
    for name, mod in (("view_role_emb", model.view_role_emb),
                      ("slot_seed_proj", model.slot_seed_proj)):
        assert getattr(mod, "_is_zero_init", False), f"{name}: marker missing"

        # Re-run exactly what _init_missing_weights does for a missing weight.
        for p in mod.parameters(recurse=False):
            nn.init.normal_(p, std=1.0)
        model._init_weights(mod)
        for p in mod.parameters(recurse=False):
            assert p.abs().max().item() == 0.0, (
                f"{name}: _init_weights must re-zero the marked module; otherwise the "
                "meta-device build leaves it randomly initialized and the feature is no "
                "longer a no-op at step 0"
            )


def test_adanorm_film_is_identity_after_meta_device_build():
    """AdaRMSNorm's gamma/beta must survive the meta-device build as zeros.

    Same trap as view_role_emb / slot_seed_proj: the model is built on the meta device, so the
    ``nn.init.zeros_`` in AdaRMSNorm.__init__ is a silent no-op, and the real values come from
    _init_missing_weights, which maps a missing "...layernorm.gamma.weight" to the owning
    nn.Linear and calls _init_weights on it. Without the ``_is_zero_init`` marker that hands
    gamma/beta a normal_(std=initializer_range) draw, so FiLM is NOT the identity at step 0 and
    the expert is not the clean residual network its docstring claims -- with cond of order
    sqrt(hidden) the resulting scale jitter is roughly +/-0.5 on a stream whose RMS is 1.

    AdaRMSNorm.weight itself needs no marker: HF's _init_weights has a name-based branch that
    fills any "*RMSNorm" module's weight with 1.0. Both expert backbones are covered, since the
    hand-written stack (replace_lnorm_with_adanorm) and the stock stack
    (replace_stock_norms_with_adanorm) both construct AdaRMSNorm.
    """
    import torch
    import torch.nn as nn

    from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig
    from rynnvla.models.rynn_brain_vla.expert_rynn_brain_vla import AdaRMSNorm
    from rynnvla.models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLAModel

    def config(backbone):
        return RynnBrainVLAConfig(
            text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                             vocab_size=64,
                             rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]}),
            vision_config=dict(hidden_size=32, intermediate_size=64, depth=1, num_heads=2,
                               out_hidden_size=32, patch_size=14, spatial_merge_size=2,
                               temporal_patch_size=2, in_channels=3, num_position_embeddings=16),
            action_dim=7, action_chunk_size=2, action_head_type="expert",
            expert_backbone=backbone,
            expert_hidden_size=32, expert_intermediate_size=64, expert_num_attention_heads=2,
            expert_num_foresight_tokens=0, state_token_id=9,
        )

    for backbone in ("custom", "qwen3_vl"):
        model = RynnBrainVLAModel(config(backbone))
        norms = [m for m in model.action_expert.modules() if isinstance(m, AdaRMSNorm)]
        assert norms, f"{backbone}: expected AdaRMSNorm modules in the expert"

        for norm in norms:
            for child_name in ("gamma", "beta"):
                child = getattr(norm, child_name)
                assert getattr(child, "_is_zero_init", False), (
                    f"{backbone}: {child_name} is missing the _is_zero_init marker"
                )
                # Re-run exactly what _init_missing_weights does for a missing weight.
                nn.init.normal_(child.weight, std=1.0)
                nn.init.normal_(child.bias, std=1.0)
                model._init_weights(child)
                assert child.weight.abs().max().item() == 0.0, (
                    f"{backbone}: {child_name}.weight must be re-zeroed, otherwise FiLM is not "
                    "the identity at step 0"
                )
                assert child.bias.abs().max().item() == 0.0, f"{backbone}: {child_name}.bias"

            # The RMSNorm gain rides HF's name-based branch, not the marker.
            nn.init.zeros_(norm.weight)
            model._init_weights(norm)
            assert torch.all(norm.weight == 1.0), (
                f"{backbone}: AdaRMSNorm.weight must be 1.0 after _init_weights"
            )

            # With gamma/beta zero the module must be exactly plain RMSNorm, for an arbitrarily
            # large conditioning vector.
            x = torch.randn(2, 5, 32)
            cond = torch.randn(2, 32) * 100.0
            var = x.pow(2).mean(-1, keepdim=True)
            plain = norm.weight * (x * torch.rsqrt(var + norm.variance_epsilon))
            assert torch.allclose(norm(x, cond), plain, atol=1e-6), (
                f"{backbone}: FiLM must be the identity at step 0"
            )


# --------------------------------------------------------------------------------------
# Arm V4: per-camera-role context tokens in the expert suffix.
#
# B1/B2 tag the camera on the PREFIX side; V4 is the expert-side counterpart, and the only
# one of the three whose weights (slot_seed_proj) a latent stage-1 run can hand to a direct
# fine-tune: both stages feed it the same input (a normalised pooled camera feature) and
# read it in the same place (one context token per role, positions fixed by the role table).
#
# Unlike B2, V4 is NOT a step-0 no-op -- it adds attention KEYS, so the flag-on run differs
# from the flag-off run even at zero-init. What zero-init buys is that the five tokens start
# identical (no camera identity), so the identity is learned rather than injected as noise.
# --------------------------------------------------------------------------------------

_V4_TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=200,
                rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]})
_V4_VIS = dict(hidden_size=64, intermediate_size=128, depth=2, num_heads=4, out_hidden_size=64,
               patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
               num_position_embeddings=64, deepstack_visual_indexes=[0])
_V4_SLOTS = torch.tensor([3, 1])          # LIBERO: front_third, left_wrist
# Derived from _V4_SLOTS and the constant, never hardcoded: this doubles as slot_mask for the
# latent path, so a literal of the wrong width silently desynchronises the K action blocks
# from their validity mask instead of failing on the axis it is meant to pin.
_V4_VALID = [[i in set(_V4_SLOTS.tolist()) for i in range(NUM_VIEW_SLOTS)]]
_LATENT_K, _LATENT_CHUNK, _LATENT_DIM = NUM_VIEW_SLOTS, 3, 8


def _v4_model(foresight=0, latent=False, **kw):
    from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig
    from rynnvla.models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLAModel

    extra = dict(use_latent_actions=True, latent_action_dim=_LATENT_DIM,
                 latent_action_chunk_size=_LATENT_CHUNK, num_view_slots=_LATENT_K) if latent else {}
    torch.manual_seed(0)
    cfg = RynnBrainVLAConfig(
        text_config=_V4_TEXT, vision_config=_V4_VIS, action_dim=7, action_chunk_size=4,
        action_head_type="expert", expert_hidden_size=64, expert_intermediate_size=128,
        expert_num_attention_heads=4, expert_num_foresight_tokens=foresight,
        expert_backbone="qwen3_vl",
        # expert_time_concat is what the xl_lingbot arms run with, but it is guarded off for
        # the multiview path (modeling :488-500), so the latent build has to drop it.
        expert_time_concat=not latent,
        state_token_id=99, image_token_id=100, **extra, **kw)
    return RynnBrainVLAModel(cfg).eval()


def _v4_batch(n_images=2):
    torch.manual_seed(7)
    patch, merge = 14, 2
    ntok = (2 * 4 * 4) // (merge * merge)
    grid = torch.tensor([[2, 4, 4]] * n_images)
    ids = [5, 6]
    for _ in range(n_images):
        ids += [100] * ntok + [7]
    ids += [8, 99, 9]
    input_ids = torch.tensor([ids])
    return dict(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.randn(int(grid.prod(-1).sum()), 3 * 2 * patch * patch),
        image_grid_thw=grid, states=torch.randn(1, 1, 7),
        actions=torch.randn(1, 4, 7), action_mask=torch.ones(1, 4, 7, dtype=torch.bool),
        times=torch.rand(1),
    )


def _v4_actions(out):
    return out.actions if getattr(out, "actions", None) is not None else out[0]


def _v4_prefix(model, batch, slots=_V4_SLOTS):
    return model._embed_prefix(
        input_ids=batch["input_ids"], inputs_embeds=None, position_ids=None,
        attention_mask=batch["attention_mask"], past_key_values=None,
        pixel_values=batch["pixel_values"], pixel_values_videos=None,
        image_grid_thw=batch["image_grid_thw"], video_grid_thw=None,
        states=batch["states"], actions=None, history_states=None, history_actions=None,
        camera_slot_ids=slots)


def _break_symmetry(model, std=0.05):
    """slot_seed_proj is zero-init, so at step 0 all five tokens are the SAME zero vector.

    Camera identity only exists once it has weights, so every identity assertion below has
    to perturb it first -- testing at init would pass trivially and prove nothing.
    """
    with torch.no_grad():
        model.slot_seed_proj.weight.normal_(std=std)
        model.slot_seed_proj.bias.normal_(std=std)


def test_v4_is_off_by_default():
    from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig

    assert RynnBrainVLAConfig().use_view_cond_slots is False
    model, batch = _v4_model(), _v4_batch()
    embeds, _, vmask, _ = _v4_prefix(model, batch)
    assert model._build_view_cond_tokens(embeds, vmask, batch["image_grid_thw"], _V4_SLOTS) == (None, None)


def test_v4_emits_one_token_per_role_and_marks_the_empty_ones():
    """The slot axis is the fixed role table, not the sample's camera count.

    LIBERO ships 2 cameras, so NUM_VIEW_SLOTS-2 role slots are empty; they must be flagged
    explicitly (``valid``) rather than inferred from "the pooled feature is zero", which is
    not guaranteed and fails silently.
    """
    model, batch = _v4_model(use_view_cond_slots=True), _v4_batch()
    embeds, _, vmask, _ = _v4_prefix(model, batch)
    tokens, valid = model._build_view_cond_tokens(embeds, vmask, batch["image_grid_thw"], _V4_SLOTS)
    assert tokens.shape == (1, NUM_VIEW_SLOTS, 64)
    assert valid.tolist() == _V4_VALID
    # The two filled roles carry real, DIFFERENT content from step 0 (they are pooled from
    # two different cameras). Not zero -- see test_v4_does_not_zero_init_the_camera_projection.
    filled = tokens[0][torch.tensor(_V4_VALID[0])]
    assert filled.abs().max().item() > 0.0
    assert (filled[0] - filled[1]).abs().max().item() > 0.0


@pytest.mark.parametrize("foresight", [0, 2])
@pytest.mark.parametrize("latent", [False, True], ids=["direct", "latent"])
def test_v4_carries_camera_identity(foresight, latent):
    """Swapping which physical camera owns which role MUST change the output.

    This is the regression that pins the design. The obvious cheap implementation --
    ``cond += sum_k slot_seed_proj(norm(seed_k))`` -- is permutation INVARIANT, because a
    shared linear map obeys sum_k proj(s_k) == proj(sum_k s_k) + (K-1)b: it would train,
    log a plausible loss, and carry exactly zero camera identity. Identity has to be
    carried by POSITION, i.e. one token per role.
    """
    model = _v4_model(foresight=foresight, latent=latent, use_view_cond_slots=True)
    batch = _v4_batch()
    _break_symmetry(model)
    with torch.no_grad():
        if latent:
            kw = dict(latent_noise=torch.randn(1, _LATENT_K, _LATENT_CHUNK, _LATENT_DIM),
                      slot_mask=torch.tensor(_V4_VALID))
            inputs = {k: v for k, v in batch.items() if k not in ("actions", "action_mask")}
            base = _v4_actions(model(**inputs, camera_slot_ids=_V4_SLOTS, **kw))
            swapped = _v4_actions(model(**inputs, camera_slot_ids=torch.tensor([1, 3]), **kw))
        else:
            base = _v4_actions(model(**batch, camera_slot_ids=_V4_SLOTS))
            swapped = _v4_actions(model(**batch, camera_slot_ids=torch.tensor([1, 3])))
    assert (swapped - base).abs().max().item() > 1e-5


@pytest.mark.parametrize("foresight", [0, 2])
def test_v4_empty_roles_are_key_masked(foresight):
    """Roles no camera landed in must be unreadable, not "attended as a zero token".

    Poisoning the empty slots and getting a bit-identical output is the only way to tell
    the two apart: an unmasked zero token still contributes value rows after LayerNorm.
    """
    model = _v4_model(foresight=foresight, use_view_cond_slots=True)
    batch = _v4_batch()
    _break_symmetry(model)
    with torch.no_grad():
        clean = _v4_actions(model(**batch, camera_slot_ids=_V4_SLOTS))

    original = model._build_view_cond_tokens

    def poisoned(*args, **kwargs):
        tokens, valid = original(*args, **kwargs)
        tokens = tokens.clone()
        tokens[:, ~valid[0]] += 7.0
        return tokens, valid

    model._build_view_cond_tokens = poisoned
    with torch.no_grad():
        after = _v4_actions(model(**batch, camera_slot_ids=_V4_SLOTS))
    assert torch.equal(after, clean)


@pytest.mark.parametrize("foresight", [0, 2])
@pytest.mark.parametrize("v4", [False, True], ids=["off", "on"])
def test_v4_direct_train_and_decode_agree(foresight, v4):
    """The joint forward and the cached decode implement the suffix layout TWICE.

    Any disagreement is a silent eval-only regression (the class of bug that produced the
    decode RoPE-offset fix documented in _expert_decode), so V4's insert point, its RoPE
    positions and its readout slice are checked against each other, not just each alone.
    """
    from rynnvla.models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLACache

    model = _v4_model(foresight=foresight, use_view_cond_slots=v4)
    batch = _v4_batch()
    if v4:
        _break_symmetry(model)
    with torch.no_grad():
        trained = _v4_actions(model(**batch, camera_slot_ids=_V4_SLOTS))

        prefill_inputs = {k: v for k, v in batch.items()
                          if k not in ("actions", "action_mask", "times")}
        cache = RynnBrainVLACache()
        model(**prefill_inputs, camera_slot_ids=_V4_SLOTS, past_key_values=cache)

        _, position_ids, _, _ = _v4_prefix(model, batch)
        chunk = batch["actions"].size(1)
        decode_pos = torch.arange(1, chunk + 1).view(1, 1, chunk).repeat(3, 1, 1) \
            + position_ids[..., -1:]
        decoded = _v4_actions(model(
            actions=batch["actions"], times=batch["times"], past_key_values=cache,
            position_ids=decode_pos,
            cache_position=torch.arange(chunk) + batch["input_ids"].size(-1),
        ))
    assert (decoded - trained).abs().max().item() < 2e-4


@pytest.mark.parametrize("foresight", [0, 2])
@pytest.mark.parametrize("v4", [False, True], ids=["off", "on"])
def test_v4_latent_joint_and_cached_step_agree(foresight, v4):
    """Same check for the latent path, whose cached step is a separate implementation."""
    model = _v4_model(foresight=foresight, latent=True, use_view_cond_slots=v4)
    batch = _v4_batch()
    if v4:
        _break_symmetry(model)
    torch.manual_seed(3)
    x_t, times = torch.randn(1, _LATENT_K, _LATENT_CHUNK, _LATENT_DIM), torch.rand(1)
    slot_mask = torch.tensor(_V4_VALID)

    inputs = {k: v for k, v in batch.items() if k not in ("actions", "action_mask", "times")}
    with torch.no_grad():
        joint = _v4_actions(model(**inputs, camera_slot_ids=_V4_SLOTS, latent_noise=x_t,
                                  times=times, slot_mask=slot_mask))
        prefix_state, ee_bias = model.prefill_prefix_from_inputs(
            camera_slot_ids=_V4_SLOTS, **inputs)
        cached = model.expert_step_multiview(prefix_state, x_t, times,
                                             slot_mask=slot_mask, ee_bias=ee_bias)
    assert joint.shape == (1, _LATENT_K, _LATENT_CHUNK, _LATENT_DIM)
    assert (cached - joint).abs().max().item() < 2e-4

    if v4:
        tokens, valid = prefix_state["view_cond"]
        assert tokens.shape == (1, _LATENT_K, 64) and valid.tolist() == _V4_VALID
    else:
        assert prefix_state["view_cond"] == (None, None)

    # fix4's action step reuses the same context block; it must survive the wider block.
    with torch.no_grad():
        velocity = model.expert_action_step_multiview(
            prefix_state, joint, torch.randn(1, 4, 7), times,
            slot_mask=slot_mask, ee_bias=ee_bias)
    assert velocity.shape == (1, 4, 7)


def test_v4_refuses_the_amortized_path_instead_of_silently_dropping_cameras():
    """expert_train_repeat>1 has no V4 wiring; training there while inference builds the
    tokens would be a train/eval mismatch, so it raises."""
    model = _v4_model(use_view_cond_slots=True, expert_train_repeat=4)
    batch = _v4_batch()
    model.train()
    with pytest.raises(ValueError, match="use_view_cond_slots"):
        model(**batch, camera_slot_ids=_V4_SLOTS)


def test_v4_per_sample_roles_survive_flat_collation():
    """camera_slot_ids is collated FLAT (data_collator.py:37-38), not (B, n_images).

    Slot k of sample b therefore lives at index b*n_images + k, and _pool_view_seeds indexes
    it with the GLOBAL image counter from _image_token_spans. An off-by-one there would
    still run, still train, and quietly hand every sample after the first the wrong camera
    identities -- so a batched run is pinned against the equivalent single-sample runs.
    """
    model = _v4_model(use_view_cond_slots=True)
    _break_symmetry(model)
    single = _v4_batch()
    roles_a, roles_b = torch.tensor([3, 1]), torch.tensor([1, 3])
    with torch.no_grad():
        solo_a = _v4_actions(model(**single, camera_slot_ids=roles_a))
        solo_b = _v4_actions(model(**single, camera_slot_ids=roles_b))

    pair = {k: torch.cat([v, v], dim=0) for k, v in single.items()}
    with torch.no_grad():
        batched = _v4_actions(model(**pair, camera_slot_ids=torch.cat([roles_a, roles_b])))

    assert (batched[0] - solo_a[0]).abs().max().item() < 1e-5
    assert (batched[1] - solo_b[0]).abs().max().item() < 1e-5
    assert (batched[0] - batched[1]).abs().max().item() > 1e-5, "both samples got the same roles"


def test_v4_refuses_to_run_without_role_ids():
    """No camera_slot_ids -> _pool_view_seeds would slot by sample order (0,1) instead of
    by role (3,1). Train and eval would then disagree about which token is which camera,
    with no error and no crash -- so it raises instead."""
    model, batch = _v4_model(use_view_cond_slots=True), _v4_batch()
    embeds, _, vmask, _ = _v4_prefix(model, batch)
    with pytest.raises(ValueError, match="camera_slot_ids"):
        model._build_view_cond_tokens(embeds, vmask, batch["image_grid_thw"], None)


def test_v4_does_not_zero_init_the_camera_projection():
    """Zero-init is right for an additive bias and WRONG for a token.

    In the latent additive form slot_seed_proj's output is added to an already-nonzero
    action block, so zero simply means "slots start symmetric". In V4 the output IS the
    token, and a zero token is degenerate rather than neutral: the expert's first norm sees
    x = 0, whose Jacobian scales like 1/sqrt(eps). Measured on the real 2B + a LIBERO batch,
    step 0:

        zero-init    camera token rms 0.000   slot_seed_proj grad 1.28e5   TOTAL 1.285e5
        normal(.02)  camera token rms 0.576   slot_seed_proj grad 2.67     TOTAL 1.57e2
        (xl_lingbot, for scale:                                            TOTAL 1.51e2)

    With max_grad_norm=1.0 the zero-init variant scales every OTHER parameter's step down by
    ~850x, so the arm would measure "camera tokens + a wrecked effective LR" instead of
    camera tokens. slot_seed_norm's gradient is exactly 0 there as well (it sits before a
    zero matrix), i.e. the whole block is dead on arrival.
    """
    additive = _v4_model()                                   # latent-style additive use
    token = _v4_model(use_view_cond_slots=True)               # V4 context-token use
    assert getattr(additive.slot_seed_proj, "_is_zero_init", False) is True
    assert getattr(token.slot_seed_proj, "_is_zero_init", False) is False
    assert additive.slot_seed_proj.weight.abs().max().item() == 0.0
    assert token.slot_seed_proj.weight.abs().max().item() > 0.0
