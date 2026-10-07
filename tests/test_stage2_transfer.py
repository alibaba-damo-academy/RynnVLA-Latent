"""Stage-1 -> Stage-2 transfer invariants, on small CPU models.

These are the properties a Stage-2 run silently depends on and that nothing raises about:

* ``bypass_latent_output_projection`` drops the Stage-1 latent head's output layer, so the
  inherited weights land in a differently-shaped module. Which keys go missing, which must be
  re-initialised, and that the flag survives a ``save_pretrained`` / ``from_pretrained``
  roundtrip (HF's ``from_dict`` drops config keys it does not know about, so an undeclared flag
  would come back as the default and rebuild the wrong head).
* ``_reset_pretrained_modules`` consumes the global RNG. If it did not restore it, every
  downstream init in the same process would differ depending on whether a reset ran -- a
  reproducibility break with no visible symptom.
* every parameter is in exactly one LR group, so a module cannot silently train at the base LR
  when the recipe meant to give it ``action_head_lr``.
* the LR-group recipe is locked on resume: changing which modules are "fresh" mid-run changes
  what every optimizer state slot means.
* the direct forward and the cached two-pass forward agree, and gradients reach every part of
  the trunk the recipe expects to train.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from rynnvla import models
from rynnvla.api.train import resolve_resume
from rynnvla.models.rynn_brain_vla.configuration_rynn_brain_vla import RynnBrainVLAConfig
from rynnvla.models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLAModel, RynnBrainVLACache
from rynnvla.training.trainer import Trainer


def make_model(stage1=False, bypass=False):
    cfg = RynnBrainVLAConfig(
        text_config=dict(vocab_size=200, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                                      'mrope_section': [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                           out_hidden_size=64, deepstack_visual_indexes=[0], patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                           num_position_embeddings=64),
        action_dim=81, action_chunk_size=10, latent_action_dim=608, num_view_slots=6,
        state_token_id=99, image_token_id=100, action_head_type='expert',
        expert_backbone='qwen3_vl', expert_hidden_size=64, expert_intermediate_size=128,
        expert_num_foresight_tokens=0, expert_time_concat=True, expert_per_layer_adanorm=True,
        use_view_cond_slots=True, use_latent_actions=stage1, use_latent_head_readout=not stage1,
        bypass_latent_output_projection=bypass,
    )
    return RynnBrainVLAModel(cfg).float()


@pytest.fixture(autouse=True)
def local_runtime(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for name in ('get_expert_model_parallel_rank', 'get_expert_data_parallel_rank', 'get_data_parallel_rank'):
        monkeypatch.setattr(models.mpu, name, lambda: 0)
    monkeypatch.setattr(models.mpu, 'get_expert_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(models.mpu, 'get_expert_model_parallel_group', lambda: None)


def test_default_keeps_old_head_and_bypass_requires_stage2():
    model = make_model()
    assert model.latent_action_head[2].weight.shape == (608, 64)
    assert model.latent_readout_proj.weight.shape == (81, 608)
    assert RynnBrainVLAConfig().bypass_latent_output_projection is False
    with pytest.raises(ValueError, match='requires a Stage-2'):
        make_model(stage1=True, bypass=True)


def test_bypass_transfers_expert_and_first_head_layer_exactly(tmp_path):
    source = make_model(stage1=True)
    source.save_pretrained(tmp_path)
    target = make_model(bypass=True)
    target_sd = target.state_dict()
    missing = models._load_pretrained_weights(target, target_sd, str(tmp_path))
    assert set(missing) == {'latent_readout_proj.weight', 'latent_readout_proj.bias'}
    assert not any(k.startswith('latent_action_head.2.') for k in target_sd)
    assert target.latent_readout_proj.weight.shape == (81, 64)
    source_sd = source.state_dict()
    for name, tensor in target_sd.items():
        if name not in missing:
            assert torch.equal(tensor, source_sd[name]), name
    models._init_missing_weights(target, missing)
    target.save_pretrained(tmp_path/'roundtrip')
    restored = RynnBrainVLAModel.from_pretrained(tmp_path/'roundtrip', attn_implementation='eager')
    assert restored.config.bypass_latent_output_projection is True
    assert restored.latent_readout_proj.weight.shape == (81, 64)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def test_random_head_reset_preserves_trunk_adapters_readout_and_rng():
    model = make_model()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    prefixes = ['action_expert', 'latent_action_head', 'slot_seed_norm', 'slot_seed_proj']
    models._reset_pretrained_modules(model, prefixes,
        ['latent_readout_proj.weight', 'latent_readout_proj.bias'], seed=42)
    assert torch.equal(rng, torch.random.get_rng_state())
    after = {k:v.clone() for k,v in model.state_dict().items()}
    for name, tensor in after.items():
        if not any(name.startswith(prefix+'.') for prefix in prefixes):
            assert torch.equal(tensor, before[name]), name
    for prefix in ['action_expert', 'latent_action_head', 'slot_seed_proj']:
        assert any(not torch.equal(t, before[n]) for n,t in after.items() if n.startswith(prefix+'.'))
    models._reset_pretrained_modules(model, prefixes, [], seed=42)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, after[name]), name


def test_low_lr_only_changes_inherited_modules():
    model = make_model()
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    fresh = ['state_proj', 'action_in_proj', 'action_out_proj', 'latent_readout_proj']
    trainer.args = SimpleNamespace(action_head_lr=1e-4, learning_rate=2.5e-5,
        action_head_modules=fresh, weight_decay=0., deepspeed_config={})
    trainer.get_optimizer_cls_and_kwargs = lambda args, model: (torch.optim.AdamW, {'lr':args.learning_rate})
    trainer.create_optimizer()
    rates = {id(p):g['lr'] for g in trainer.optimizer.param_groups for p in g['params']}
    assert len(rates) == len(list(model.parameters()))
    for name,p in model.named_parameters():
        expected = 1e-4 if name.split('.')[0] in fresh else 2.5e-5
        assert rates[id(p)] == expected, name


def test_lr_group_definition_is_locked_on_resume(tmp_path):
    checkpoint = tmp_path/'checkpoint-5000'
    checkpoint.mkdir()
    for name in ['config.json', 'processor_config.json']:
        (checkpoint/name).write_text('{}')
    modules = ['state_proj', 'action_in_proj', 'action_out_proj', 'latent_readout_proj']
    (checkpoint/'resume_args.json').write_text(json.dumps({'action_head_modules':modules}))
    values = dict(output_dir=str(tmp_path), action_head_modules=modules)
    assert resolve_resume(values, True, lambda _:str(checkpoint)) == str(checkpoint)
    with pytest.raises(ValueError, match='action_head_modules'):
        resolve_resume({**values, 'action_head_modules':['action_expert']}, True, lambda _:str(checkpoint))


def test_new_head_backward_and_cached_inference_agree():
    model = make_model(bypass=True).eval()
    ids = torch.tensor([[5, 6]+[100]*8+[7]+[100]*8+[8, 99, 9]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(ids.shape[1]).view(1,1,-1).expand(3,1,-1),
                 pixel_values=torch.randn(64, 3*2*14*14),
                 image_grid_thw=torch.tensor([[2,4,4]]*2), camera_slot_ids=torch.tensor([3,1]),
                 states=torch.randn(1,1,81))
    actions, times = torch.randn(1,10,81), torch.tensor([0.8])
    with torch.no_grad():
        direct = model(**batch, actions=actions, times=times).actions
        cache = RynnBrainVLACache()
        model(**batch, past_key_values=cache)
        cached = model(actions=actions, times=times, past_key_values=cache).actions
    assert direct.shape == (1,10,81)
    torch.testing.assert_close(direct, cached, atol=1e-6, rtol=1e-5)
    model.train()
    loss = model(**batch, actions=actions, times=times).loss
    assert torch.isfinite(loss)
    loss.backward()
    for prefix in ['visual', 'language_model', 'action_expert', 'latent_action_head.0', 'latent_readout_proj']:
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for n,p in model.named_parameters()
                   if n.startswith(prefix+'.')), prefix
