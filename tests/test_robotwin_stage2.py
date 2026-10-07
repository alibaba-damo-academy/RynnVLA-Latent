"""RoboTwin transfer: action semantics, LR isolation, recipe and resume checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from transformers import get_scheduler

from rynnvla.api.train import resolve_resume
from rynnvla.constants import RotationRepresentation
from rynnvla.datasets.vla_datasets.robotwin import RoboTwinDataset, SUPPORTED_VARIANTS
from rynnvla.training.trainer import Trainer
from test_stage2_transfer import make_model


def optimizer(model, transfer_lr=2.5e-5, **overrides):
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    values = dict(learning_rate=1e-5, action_head_lr=1e-4, transfer_lr=transfer_lr,
                  action_head_modules=['state_proj', 'action_in_proj', 'action_out_proj', 'latent_readout_proj'],
                  transfer_modules=['action_expert', 'latent_action_head', 'slot_seed_norm', 'slot_seed_proj'],
                  weight_decay=.1, deepspeed_config={})
    trainer.args = SimpleNamespace(**{**values, **overrides})
    trainer.get_optimizer_cls_and_kwargs = lambda args, model: (torch.optim.AdamW, {'lr': args.learning_rate})
    trainer.create_optimizer()
    return trainer.optimizer


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.mark.parametrize('rate', [1e-4, 2.5e-5])
def test_three_rates_cover_parameters_once(rate):
    model = make_model()
    opt = optimizer(model, rate)
    assigned = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))
    rates = {id(p): group['lr'] for group in opt.param_groups for p in group['params']}
    for name, parameter in model.named_parameters():
        module = name.split('.')[0]
        expected = 1e-5 if module in ('visual', 'language_model') else (
            rate if module in ('action_expert', 'latent_action_head', 'slot_seed_norm', 'slot_seed_proj') else 1e-4)
        assert rates[id(parameter)] == expected, name


@pytest.mark.parametrize('overrides', [
    {'transfer_modules': ['state_proj']}, {'transfer_modules': ['misspelled']},
    {'transfer_lr': -1}, {'transfer_lr': float('nan')}, {'transfer_lr': None},
    {'transfer_modules': []},
])
def test_bad_transfer_groups_fail(overrides):
    with pytest.raises(ValueError):
        optimizer(make_model(), **overrides)


def test_optimizer_scheduler_state_resumes_exactly():
    model = make_model()
    opt = optimizer(model)
    def schedule(optim):
        return get_scheduler('cosine_with_min_lr', optim, num_warmup_steps=2, num_training_steps=8,
                             scheduler_specific_kwargs={'min_lr_rate': .05})
    scheduler = schedule(opt)
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.ones_like(p) * .01
        opt.step()
        scheduler.step()
    cloned = copy.deepcopy(model)
    restored = optimizer(cloned)
    restored_scheduler = schedule(restored)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for _ in range(5):
        for m, o, s in [(model, opt, scheduler), (cloned, restored, restored_scheduler)]:
            for p in m.parameters():
                p.grad = torch.ones_like(p) * .01
            o.step()
            s.step()
        assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
    for left, right in zip(model.parameters(), cloned.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for g in opt.param_groups:
        peak = 2.5e-5 if g['name'].startswith('transfer_') else (
            1e-4 if g['name'].startswith('action_head_') else 1e-5)
        assert g['lr'] == pytest.approx(peak * .05)


@pytest.mark.parametrize('field,value', [('transfer_lr', 1e-4), ('transfer_modules', ['action_expert'])])
def test_resume_rejects_transfer_recipe_changes(tmp_path, field, value):
    checkpoint = tmp_path / 'checkpoint-5000'
    checkpoint.mkdir()
    for name in ('config.json', 'processor_config.json'):
        (checkpoint / name).write_text('{}')
    values = dict(output_dir=str(tmp_path), transfer_lr=2.5e-5,
                  transfer_modules=['action_expert', 'latent_action_head'])
    (checkpoint / 'resume_args.json').write_text(json.dumps(values))
    assert resolve_resume(values, True, lambda _: str(checkpoint)) == str(checkpoint)
    with pytest.raises(ValueError, match=field):
        resolve_resume({**values, field: value}, True, lambda _: str(checkpoint))


@pytest.fixture
def episode_dataset(tmp_path):
    import cv2
    episodes = []
    for variant in SUPPORTED_VARIANTS:
        path = tmp_path / 'task' / variant / 'data' / 'episode0.hdf5'
        path.parent.mkdir(parents=True)
        with h5py.File(path, 'w') as f:
            n = 35
            for side in ('left', 'right'):
                pose = np.zeros((n, 7), dtype=np.float32)
                pose[:, 0] = np.arange(n) * .01
                angle = np.arange(n) * .02
                pose[:, 3] = np.cos(angle / 2)
                pose[:, 6] = np.sin(angle / 2)
                f.create_dataset(f'endpose/{side}_endpose', data=pose)
                f.create_dataset(f'endpose/{side}_gripper', data=np.linspace(0, 1, n, dtype=np.float32))
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :, 0] = 200
            _, jpg = cv2.imencode('.jpg', frame)
            for camera in ('head', 'left', 'right'):
                images = f.create_dataset(f'observation/{camera}_camera/rgb', (n,),
                                          dtype=h5py.vlen_dtype(np.dtype('uint8')))
                for i in range(n):
                    images[i] = jpg
        episodes.append(dict(path=str(path), length=n, instructions=['move'], robot_type='aloha_agilex'))
    index = tmp_path / 'index.json'
    index.write_text(json.dumps(episodes))
    return RoboTwinDataset(data_path=str(tmp_path), index_cache=str(index), action_chunk_size=30,
                           use_delta_action=True, eef_rotation_repr=RotationRepresentation.ROT_6D,
                           chunk_overlap_ratio=.99)


def test_data_delta_gripper_rotation_camera_and_tail(episode_dataset):
    dataset = episode_dataset
    assert len(dataset) == 70
    sample = dataset[3]
    assert sample['camera_slot_map'] == {'head': 0, 'left': 1, 'right': 2}
    assert all(image.shape == (16, 16, 3) for image in sample['images'].values())
    pixel = sample['images']['head'][0, 0].to(torch.int32)
    assert pixel[0] > pixel[2] + 100
    action, state = sample['action'], sample['state']
    absolute = action + state
    expected = dataset.load_action(0, slice(3, 33)).convert_rotation(RotationRepresentation.ROT_6D)
    for side in ('left', 'right'):
        arm = getattr(action, side + '_arm')
        torch.testing.assert_close(arm.eef_position.data[0], torch.zeros(3), atol=1e-7, rtol=0)
        torch.testing.assert_close(arm.eef_rotation.data[0], torch.tensor([1., 0., 0., 1., 0., 0.], dtype=arm.eef_rotation.data.dtype))
        grip = getattr(action, side + '_gripper')
        assert not grip.is_relative
        torch.testing.assert_close(grip.data, getattr(expected, side + '_gripper').data)
        for field in ('eef_position', 'eef_rotation'):
            torch.testing.assert_close(getattr(getattr(absolute, side + '_arm'), field).data,
                                       getattr(getattr(expected, side + '_arm'), field).data)
    torch.testing.assert_close(dataset[5]['action'].left_arm.eef_position.data,
                               dataset[34]['action'].left_arm.eef_position.data)


def test_index_paths_under_a_symlinked_data_path(episode_dataset, tmp_path):
    # Object-store mounts are commonly reached through a symlinked alias, and the index and
    # data_path may both name the alias rather than the real directory. The schema cache key
    # and the episode paths have to agree on whichever spelling was given, or a checkpoint
    # trained through one alias would be rejected (or worse, accepted) under the other.
    alias = tmp_path.parent / (tmp_path.name + '_public_mount')
    alias.symlink_to(tmp_path, target_is_directory=True)
    episodes = copy.deepcopy(episode_dataset._episodes)
    for episode in episodes:
        episode['path'] = str(alias / Path(episode['path']).relative_to(tmp_path))
    index = tmp_path / 'alias_index.json'
    index.write_text(json.dumps(episodes))
    dataset = RoboTwinDataset(data_path=str(alias), index_cache=str(index), action_chunk_size=30,
                              use_delta_action=True, eef_rotation_repr=RotationRepresentation.ROT_6D,
                              chunk_overlap_ratio=.99)
    assert dataset._schema_cache_key()['data_path'] == str(alias)
    assert len(dataset) == len(episode_dataset)
    for idx in (3, 38):
        actual, expected = dataset[idx], episode_dataset[idx]
        for camera in ('head', 'left', 'right'):
            torch.testing.assert_close(actual['images'][camera], expected['images'][camera])
        torch.testing.assert_close(actual['action'].left_arm.eef_position.data,
                                   expected['action'].left_arm.eef_position.data)


@pytest.mark.parametrize('escape', ['../outside', '../dataset-sibling'])
def test_index_rejects_paths_outside_dataset_root(episode_dataset, tmp_path, escape):
    episodes = copy.deepcopy(episode_dataset._episodes)
    episodes[0]['path'] = str(tmp_path / escape / 'aloha-agilex_clean_50/data/episode0.hdf5')
    index = tmp_path / 'outside_index.json'
    index.write_text(json.dumps(episodes))
    with pytest.raises(ValueError, match='do not belong'):
        RoboTwinDataset(data_path=str(tmp_path), index_cache=str(index), action_chunk_size=30,
                        use_delta_action=True, chunk_overlap_ratio=.99)


def test_schema_rejects_wrong_recipe_and_index(episode_dataset, tmp_path):
    dataset = episode_dataset
    path = tmp_path / 'schema.json'
    schema = dict(metadata=dataset._schema_cache_key(), index_sha256=dataset.index_sha256,
                  action={'aloha_agilex': {}}, state={'aloha_agilex': {}})
    path.write_text(json.dumps(schema))
    dataset.schema_path = str(path)
    assert dataset.get_schema()['action'] == {'aloha_agilex': {}}
    dataset.use_delta_action = False
    with pytest.raises(ValueError, match='metadata'):
        dataset.get_schema()
    dataset.use_delta_action = True
    dataset.index_sha256 = 'changed'
    with pytest.raises(ValueError, match='index'):
        dataset.get_schema()


def test_three_camera_chunk30_forward_backward_and_save(tmp_path):
    model = make_model()
    model.config.action_chunk_size = 30
    ids = torch.tensor([[5] + [100] * 8 + [6] + [100] * 8 + [7] + [100] * 8 + [8, 99, 9]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  position_ids=torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1),
                  pixel_values=torch.randn(96, 3 * 2 * 14 * 14),
                  image_grid_thw=torch.tensor([[2, 4, 4]] * 3), camera_slot_ids=torch.tensor([0, 1, 2]),
                  states=torch.randn(1, 1, 81), actions=torch.randn(1, 30, 81))
    result = model(**inputs)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for prefix in ('visual', 'language_model', 'action_expert', 'latent_action_head', 'slot_seed_proj',
                   'state_proj', 'action_in_proj', 'latent_readout_proj'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if n.startswith(prefix + '.')), prefix
    model.save_pretrained(tmp_path)
    restored = type(model).from_pretrained(tmp_path, attn_implementation='eager')
    assert restored.config.action_chunk_size == 30 and restored.latent_readout_proj.in_features == 608
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
