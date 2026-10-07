"""``scripts/build_dataset_weights.py`` must describe the sample space training actually walks.

The script re-derives per-dataset window counts from the latent index and writes
``dataset_weights = windows ** alpha`` into a new mixture. Two failure modes are silent and
expensive, and each has a test here:

* the window formula drifting from ``latent_pretrain``'s own ``_start_stride`` / ``_ep_starts``.
  The weights would then describe a different sample space than the one the dataset enumerates,
  so the realised mixture ratio is not the requested one and nothing reports it.
* the weighted virtual index changing length. ``max_steps`` is derived from ``len(dataset)`` at
  submit time and sits in ``RESUME_CRITICAL_FIELDS``, so a weighting that moved the total would
  either make an alpha arm incomparable to its baseline at equal steps, or reject every
  existing checkpoint of that arm on resume.

Also pinned: alpha=1.0 is a true no-op (writes no ``dataset_weights`` at all, so the arm is not
routed through the weighted index for no effect), and the output mixture cannot silently replace
one a queued job already read.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from rynnvla.constants import NUM_VIEW_SLOTS
from rynnvla.datasets.vla_datasets import latent_pretrain as lp

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "_build_dataset_weights_under_test", REPO_ROOT / "scripts/build_dataset_weights.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bdw = _load_script()

CHUNK, STEP_SECONDS, FPS, PAIR_STRIDE, LATENT_DIM = 6, 0.25, 20.0, 4, 4
# Shaped so the two datasets disagree in both directions at alpha<1: "Big" is downsampled and
# "Small" is upsampled, which is the only regime where a weighting bug is visible.
SPEC = {
    "Big": [40, 40, 20, 10],
    "Small": [10, 6],
}


def _build_index(tmp_path, spec=SPEC, latent_chunk=CHUNK):
    """A real index, built by the real build_index -- not a hand-rolled npz."""
    lines = []
    for name, nlats in spec.items():
        for i, nlat in enumerate(nlats):
            lines.append(json.dumps({
                "dataset": name, "episode_id": f"ep{i}", "caption": f"{name} ep{i}",
                "gap": 4, "latent_dim": LATENT_DIM,
                "views": [{"view": "head", "video_path": str(tmp_path / f"{name}{i}.mp4"),
                           "latent_path": str(tmp_path / f"{name}{i}.npz"),
                           "num_latents": nlat, "fps": FPS,
                           "pair_stride": PAIR_STRIDE, "start_frame": 0}],
            }))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = tmp_path / "index.npz"
    lp.build_index(str(manifest), str(index), latent_chunk, STEP_SECONDS)
    return index


def _mixture(tmp_path, index, **overrides):
    entry = {"data_type": "LatentPretrainDataset", "data_path": str(index),
             "action_chunk_size": 30, "use_delta_action": False,
             "latent_chunk": CHUNK, "latent_step_seconds": STEP_SECONDS,
             "latent_action_dim": LATENT_DIM}
    entry.update(overrides)
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


def _run(*argv):
    return bdw.main([str(a) for a in argv])


def _weights(path):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return entries[0].get("dataset_weights")


# ── the formula is the dataset's formula ─────────────────────────────────────

def _parser():
    import argparse
    return argparse.ArgumentParser(prog="test")


def test_window_counts_match_the_dataset_the_weights_are_for(tmp_path):
    """The docstring's promise, checked against the class rather than a re-typed formula.

    Recomputing windows independently here would only prove the test agrees with the script;
    reading ``_ep_starts`` off a constructed dataset proves both agree with the thing that
    enumerates samples at train time.
    """
    index = _build_index(tmp_path)
    ds = lp.LatentPretrainDataset(
        data_path=str(index), action_chunk_size=30, use_delta_action=False, processor=None,
        num_view_slots=NUM_VIEW_SLOTS, latent_chunk=CHUNK,
        latent_step_seconds=STEP_SECONDS, latent_action_dim=LATENT_DIM)
    expected = np.bincount(ds._ep_ds.astype(np.int64),
                           weights=ds._ep_starts.astype(np.float64),
                           minlength=len(ds._ds_names))
    names, got = bdw._window_counts(index, CHUNK, STEP_SECONDS, _parser())
    assert names == list(ds._ds_names)
    np.testing.assert_array_equal(got, expected)
    assert int(got.sum()) == len(ds)


def test_alpha_one_is_a_byte_level_no_op(tmp_path):
    """alpha=1 must produce the mixture an unweighted run would use, not an equivalent one.

    Writing ``windows ** 1.0`` would be numerically the natural proportions but would route the
    arm through ``_build_weighted_index`` and its largest-remainder rounding for no effect -- and
    a no-op that changes the code path is not a control.
    """
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    out = tmp_path / "a10.json"
    assert _run("--mixture", mixture, "--out", out, "--alpha", "1.0") == 0
    assert _weights(out) is None
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(mixture.read_text(encoding="utf-8"))


def test_weights_are_windows_to_the_alpha(tmp_path):
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    out = tmp_path / "a07.json"
    assert _run("--mixture", mixture, "--out", out, "--alpha", "0.7") == 0
    names, windows = bdw._window_counts(index, CHUNK, STEP_SECONDS, _parser())
    weights = _weights(out)
    assert set(weights) == set(names)
    for i, name in enumerate(names):
        assert weights[name] == pytest.approx(float(windows[i]) ** 0.7, rel=1e-12)
    # The fixture's shape: flattening moves share from Big to Small.
    natural = windows / windows.sum()
    effective = np.array([weights[n] for n in names])
    effective = effective / effective.sum()
    big = names.index("Big")
    assert effective[big] < natural[big]


def test_weighting_does_not_move_len_so_max_steps_stays_put(tmp_path):
    """The claim in the script docstring, measured on the dataset itself."""
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    out = tmp_path / "a07.json"
    _run("--mixture", mixture, "--out", out, "--alpha", "0.7")

    def build(entries):
        entry = entries[0]
        return lp.LatentPretrainDataset(
            data_path=entry["data_path"], action_chunk_size=30, use_delta_action=False,
            processor=None, num_view_slots=NUM_VIEW_SLOTS, latent_chunk=CHUNK,
            latent_step_seconds=STEP_SECONDS, latent_action_dim=LATENT_DIM,
            dataset_weights=entry.get("dataset_weights"))

    natural_len = len(build(json.loads(mixture.read_text(encoding="utf-8"))))
    weighted_len = len(build(json.loads(out.read_text(encoding="utf-8"))))
    assert weighted_len == natural_len
    global_batch = 8
    assert math.ceil(weighted_len / global_batch) == math.ceil(natural_len / global_batch)


# ── refusing to be wrong quietly ─────────────────────────────────────────────

def test_refuses_to_overwrite_an_existing_mixture(tmp_path):
    """data_mixture is resume-critical; a silently replaced file makes an arm irreproducible."""
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    out = tmp_path / "a07.json"
    out.write_text("[]", encoding="utf-8")
    before = out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        _run("--mixture", mixture, "--out", out, "--alpha", "0.7")
    assert out.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("argv", [
    ["--alpha", "0"],
    ["--alpha", "-0.5"],
    ["--alpha", "nan"],
])
def test_rejects_a_nonsensical_alpha(tmp_path, argv):
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    with pytest.raises(SystemExit):
        _run("--mixture", mixture, "--out", tmp_path / "w.json", *argv)
    assert not (tmp_path / "w.json").exists()


def test_out_must_differ_from_mixture_and_be_json(tmp_path):
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    with pytest.raises(SystemExit):
        _run("--mixture", mixture, "--out", mixture, "--alpha", "0.7")
    with pytest.raises(SystemExit):
        _run("--mixture", mixture, "--out", tmp_path / "w.txt", "--alpha", "0.7")


def test_rejects_a_mixture_without_exactly_one_latent_entry(tmp_path):
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    entries = json.loads(mixture.read_text(encoding="utf-8"))
    doubled = tmp_path / "two.json"
    doubled.write_text(json.dumps(entries + entries), encoding="utf-8")
    with pytest.raises(SystemExit):
        _run("--mixture", doubled, "--out", tmp_path / "w.json", "--alpha", "0.7")
    empty = tmp_path / "none.json"
    empty.write_text(json.dumps([{"data_type": "RoboTwinDataset"}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        _run("--mixture", empty, "--out", tmp_path / "w.json", "--alpha", "0.7")


def test_rejects_an_index_built_for_a_different_recipe(tmp_path):
    """Weights computed at chunk=4 would describe a sample space chunk=6 training never walks."""
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index, latent_chunk=CHUNK + 1)
    with pytest.raises(SystemExit):
        _run("--mixture", mixture, "--out", tmp_path / "w.json", "--alpha", "0.7")
    assert not (tmp_path / "w.json").exists()


def test_index_version_is_checked(tmp_path):
    index = _build_index(tmp_path)
    with np.load(index) as z:
        payload = {key: z[key] for key in z.files}
    payload["version"] = np.array([999], dtype=payload["version"].dtype)
    bogus = tmp_path / "bogus.npz"
    np.savez(bogus, **payload)
    mixture = _mixture(tmp_path, bogus)
    with pytest.raises(SystemExit):
        _run("--mixture", mixture, "--out", tmp_path / "w.json", "--alpha", "0.7")


# ── path handling ────────────────────────────────────────────────────────────

def test_relative_paths_resolve_against_the_cwd(tmp_path, monkeypatch):
    """The bundled sample mixture uses repo-relative paths; this script has to read it."""
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    entries = json.loads(mixture.read_text(encoding="utf-8"))
    entries[0]["data_path"] = index.name          # relative to the mixture's own directory
    monkeypatch.chdir(tmp_path)
    out = Path("nested/a07.json")
    assert _run("--mixture", "mixture.json", "--out", out, "--alpha", "0.7") == 0
    assert (tmp_path / out).is_file()
    assert _weights(tmp_path / out) is not None


def test_explicit_index_overrides_the_mixture_entry(tmp_path):
    index = _build_index(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = _build_index(other_dir, spec={"Solo": [40, 40]})
    mixture = _mixture(tmp_path, index)
    out = tmp_path / "w.json"
    assert _run("--mixture", mixture, "--out", out, "--alpha", "0.7", "--index", other) == 0
    assert set(_weights(out)) == {"Solo"}


def test_other_mixture_entries_survive_untouched(tmp_path):
    index = _build_index(tmp_path)
    mixture = _mixture(tmp_path, index)
    entries = json.loads(mixture.read_text(encoding="utf-8"))
    sibling = {"data_type": "RoboTwinDataset", "data_path": "data/robotwin"}
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps([sibling, entries[0], sibling]), encoding="utf-8")
    out = tmp_path / "w.json"
    assert _run("--mixture", mixed, "--out", out, "--alpha", "0.7") == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written) == 3
    assert written[0] == sibling and written[2] == sibling
    assert "dataset_weights" in written[1] and "dataset_weights" not in written[0]
