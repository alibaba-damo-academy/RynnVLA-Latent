from fractions import Fraction
from types import SimpleNamespace
import io
import json
import os
from pathlib import Path
import tarfile

import av
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def read_frame():
    from rynnvla.datasets.vla_datasets.latent_pretrain import _read_frame

    return _read_frame


@pytest.fixture(scope="module", params=[("mp4", "libx264", 0), ("mp4", "libx264", 450),
                                       ("webm", "libvpx-vp9", 0)])
def video(request, tmp_path_factory):
    suffix, codec, offset = request.param
    path = tmp_path_factory.mktemp("latent_video") / f"sample.{suffix}"
    rng = np.random.default_rng(42)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "yuv420p"
        stream.thread_count = 1
        stream.gop_size = 12
        for index in range(72):
            pixels = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = offset + index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if offset:
            assert stream.start_time > 0
        reference = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    assert len(reference) == 72
    return str(path), reference


def test_matches_sequential_decode_in_random_order(read_frame, video):
    path, reference = video
    for index in (71, 0, 31, 5, 63, 15, 1, 31):
        actual = read_frame(path, index)
        assert actual.dtype == torch.uint8
        assert actual.shape == reference[index].shape
        np.testing.assert_array_equal(actual.numpy(), reference[index])


def test_out_of_range_is_not_a_success(read_frame, video):
    path, reference = video
    with pytest.raises(IndexError):
        read_frame(path, len(reference) + 100)


def test_missing_video_raises(read_frame, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_frame(str(tmp_path / "missing.mp4"), 0)


def test_closes_container_on_seek_error(read_frame, monkeypatch):
    stream = SimpleNamespace(average_rate=30, time_base=Fraction(1, 30), start_time=0)
    closed = []

    def fail_seek(*args, **kwargs):
        raise OSError("seek failed")

    container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), seek=fail_seek,
                                close=lambda: closed.append(True))
    monkeypatch.setattr(av, "open", lambda path: container)
    with pytest.raises(OSError, match="seek failed"):
        read_frame("unused.mp4", 5)
    assert stream.thread_count == 4
    assert closed == [True]


def test_does_not_change_torch_thread_count(read_frame, video):
    before = torch.get_num_threads()
    read_frame(video[0], 5)
    assert torch.get_num_threads() == before


def test_indexed_tar_matches_original_video(read_frame, video, tmp_path, monkeypatch):
    path, reference = video
    name = 'ab0123456789.mp4'
    shard = tmp_path / 'ab.tar'
    with tarfile.open(shard, 'w') as archive:
        archive.add(path, arcname=name)
    with tarfile.open(shard) as archive:
        member = archive.getmember(name)
    index = {'n': name, 'o': member.offset_data, 's': member.size}
    # A malformed trailing duplicate must not override the valid first entry.
    (tmp_path / 'ab.jsonl').write_text(json.dumps(index) + '\n' + json.dumps({**index, 'o': 10**12}) + '\n')
    monkeypatch.setenv('ROVIDX_TAR_ROOT', str(tmp_path))
    monkeypatch.setenv('ROVIDX_INDEX_ROOT', str(tmp_path))
    for number in (71, 0, 31, 5, 63):
        actual = read_frame('rovidx_tar://' + name, number)
        np.testing.assert_array_equal(actual.numpy(), reference[number])
    from rynnvla.datasets.vla_datasets.latent_sources import TarVideo
    with TarVideo('rovidx_tar://' + name) as reader:
        reader.seek(-4, os.SEEK_END)
        assert reader.read(1000) == Path(path).read_bytes()[-4:]
        assert reader.read(1) == b''
    with pytest.raises(FileNotFoundError):
        read_frame('rovidx_tar://abmissing.mp4', 0)
    with pytest.raises(IndexError):
        read_frame('rovidx_tar://' + name, -1)


@pytest.fixture
def hdf5_video(tmp_path):
    import h5py
    from PIL import Image

    path = tmp_path / 'trajectory.hdf5'
    expected = {}
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('camera_observations/color_images')
        for camera, channel in [('camera_front', 0), ('camera_left', 1), ('camera_right', 2)]:
            dataset = group.create_dataset(camera, (80,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            expected[camera] = []
            for number in range(80):
                pixels = np.zeros((24, 32, 3), dtype=np.uint8)
                pixels[:, :, channel] = 30 + number
                encoded = io.BytesIO()
                Image.fromarray(pixels).save(encoded, format='JPEG')
                dataset[number] = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
                expected[camera].append(np.array(Image.open(io.BytesIO(encoded.getvalue())).convert('RGB')))
    return path, expected


def test_hdf5_cameras_and_frame_bounds(read_frame, hdf5_video):
    path, expected = hdf5_video
    for view, camera in [('head', 'camera_front'), ('global', 'camera_front'),
                         ('side', 'camera_front'), ('wrist_left', 'camera_left'),
                         ('wrist_right', 'camera_right')]:
        for number in (0, 79, 12):
            actual = read_frame(str(path), number, view=view)
            np.testing.assert_array_equal(actual.numpy(), expected[camera][number])
    with pytest.raises(ValueError, match='requires a source view'):
        read_frame(str(path), 0)
    with pytest.raises(IndexError):
        read_frame(str(path), 80, view='head')


def test_hdf5_dataset_keeps_camera_and_latent_frame_alignment(hdf5_video, tmp_path):
    from rynnvla.datasets.vla_datasets.latent_pretrain import LatentPretrainDataset, build_index

    video, expected = hdf5_video
    latent = tmp_path / 'latent.npz'
    np.savez(latent, latent_action=np.repeat(np.arange(20, dtype=np.float32)[:, None], 608, axis=1))
    manifest = tmp_path / 'manifest.json'
    views = [{'view': name, 'video_path': str(video), 'latent_path': str(latent),
              'num_latents': 20, 'fps': 16, 'pair_stride': 4}
             for name in ('head', 'wrist_left', 'wrist_right', 'global', 'side')]
    manifest.write_text(json.dumps([{'dataset': 'RoboMIND2.0', 'episode_id': 'one', 'caption': 'move', 'views': views}]))
    index = tmp_path / 'index.npz'
    build_index(str(manifest), str(index))
    dataset = LatentPretrainDataset(data_path=str(index), processor=None, action_chunk_size=30,
                                   use_delta_action=False, latent_action_dim=608, num_view_slots=6)
    sample = dataset._build_sample(1)  # latent 3 -> source frame 12; targets 3..8.
    expected_cameras = {0: 'camera_front', 1: 'camera_left', 2: 'camera_right',
                        3: 'camera_front', 4: 'camera_front'}
    direct = dataset.load_images(0, 3)
    for role, camera in expected_cameras.items():
        np.testing.assert_array_equal(sample['images'][f'view{role}'].numpy(), expected[camera][12])
        np.testing.assert_array_equal(direct[f'view{role}'][0].numpy(), expected[camera][12])
        np.testing.assert_array_equal(sample['latent_targets'][role, :, 0].numpy(), np.arange(3, 9))


# ── Zarr v3 stores (EgoVerse) ──────────────────────────────────────────────────
# The fixture reproduces EgoVerse's exact codec stack -- sharding_indexed wrapping
# inner chunk_shape [1] with vlen-bytes + zstd level 0, data_type variable_length_bytes,
# one shard file c/0 -- so a reader that works here works on the real store.

@pytest.fixture
def zarr_store(tmp_path):
    zarr = pytest.importorskip('zarr', minversion='3.0')
    from PIL import Image
    from zarr.codecs import VLenBytesCodec
    try:
        from zarr.codecs import ZstdCodec as Zstd
    except ImportError:
        from numcodecs import Zstd as Zstd

    path = tmp_path / 'episode'
    path.mkdir()
    group = zarr.open_group(str(path), mode='w')
    n = 80
    array = group.create_array('images.front_1', shape=(n,), dtype='variable_length_bytes',
                               shards=(n,), chunks=(1,), serializer=VLenBytesCodec(),
                               compressors=Zstd(level=0))
    # Stands in for RynnLAM's `total_frames` attribute, which is SMALLER than the array
    # (2808 vs 2900 on a real episode). decode_zarr_rgb clamps to it; FrameReader.frames
    # does not, and FrameReader is what produced the latents -- so a reader that honours
    # this attribute would silently misalign every frame past it.
    group.attrs['total_frames'] = n - 12
    expected = []
    blobs = []
    for number in range(n):
        pixels = np.zeros((24, 32, 3), dtype=np.uint8)
        pixels[:, :, 1] = 30 + number
        pixels[number % 24, number % 32] = [200, 10, 5]   # per-frame marker: catches off-by-one
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format='JPEG')
        blobs.append(encoded.getvalue())
        expected.append(np.array(Image.open(io.BytesIO(encoded.getvalue())).convert('RGB')))
    array[:] = np.array(blobs, dtype=object)
    return str(path), expected


def test_zarr_matches_sequential_decode_in_random_order(read_frame, zarr_store):
    path, expected = zarr_store
    for index in (79, 0, 68, 31, 5, 67, 12):     # 67/68/79 are past total_frames=68
        actual = read_frame(path, index, view='head')
        assert actual.dtype == torch.uint8
        assert actual.shape == expected[index].shape
        np.testing.assert_array_equal(actual.numpy(), expected[index])


def test_zarr_is_not_capped_at_the_total_frames_attribute(read_frame, zarr_store):
    """The labeler iterated array.shape[0]; clamping to attrs['total_frames'] shifts frames."""
    path, expected = zarr_store
    total = len(expected)
    for index in range(total - 12, total):        # every frame past the attribute
        np.testing.assert_array_equal(read_frame(path, index, view='head').numpy(),
                                      expected[index])


def test_zarr_view_falls_back_to_first_images_key(read_frame, zarr_store):
    """EgoVerse's manifest view is 'head' but its array is 'images.front_1'."""
    path, expected = zarr_store
    for view in ('head', 'front_1', None, 'nonexistent_camera'):
        np.testing.assert_array_equal(read_frame(path, 7, view=view).numpy(), expected[7])


def test_zarr_explicit_view_key_wins_over_the_fallback(read_frame, tmp_path, zarr_store):
    zarr = pytest.importorskip('zarr', minversion='3.0')
    from PIL import Image
    from zarr.codecs import VLenBytesCodec
    try:
        from zarr.codecs import ZstdCodec as Zstd
    except ImportError:
        from numcodecs import Zstd as Zstd

    src, _ = zarr_store
    group = zarr.open_group(src, mode='a')
    other = group.create_array('images.wrist_left', shape=(4,), dtype='variable_length_bytes',
                               shards=(4,), chunks=(1,), serializer=VLenBytesCodec(),
                               compressors=Zstd(level=0))
    blobs, expected = [], []
    for number in range(4):
        pixels = np.full((8, 8, 3), number + 1, dtype=np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format='JPEG')
        blobs.append(encoded.getvalue())
        expected.append(np.array(Image.open(io.BytesIO(encoded.getvalue())).convert('RGB')))
    other[:] = np.array(blobs, dtype=object)
    # 'wrist_left' must resolve to images.wrist_left, not to the sorted-first images.front_1.
    np.testing.assert_array_equal(read_frame(src, 2, view='wrist_left').numpy(), expected[2])


def test_zarr_out_of_range_is_not_a_success(read_frame, zarr_store):
    path, expected = zarr_store
    with pytest.raises(IndexError):
        read_frame(path, len(expected), view='head')
    with pytest.raises(IndexError):
        read_frame(path, -1, view='head')


def test_zarr_store_without_an_image_array_raises(read_frame, tmp_path):
    zarr = pytest.importorskip('zarr', minversion='3.0')
    empty = tmp_path / 'no_images'
    empty.mkdir()
    zarr.open_group(str(empty), mode='w')
    with pytest.raises(ValueError, match='No Zarr image array'):
        read_frame(str(empty), 0, view='head')


def test_zarr_dataset_keeps_frame_alignment(read_frame, zarr_store, tmp_path):
    """End to end: latent i must decode source frame start_frame + i*pair_stride."""
    from rynnvla.datasets.vla_datasets.latent_pretrain import LatentPretrainDataset, build_index

    path, expected = zarr_store
    latent = tmp_path / 'latent.npz'
    np.savez(latent, latent_action=np.repeat(np.arange(20, dtype=np.float32)[:, None], 608, axis=1))
    manifest = tmp_path / 'manifest.json'
    views = [{'view': 'head', 'video_path': path, 'latent_path': str(latent),
              'num_latents': 20, 'fps': 30, 'pair_stride': 4, 'start_frame': 0}]
    manifest.write_text(json.dumps([{'dataset': 'EgoVerse', 'episode_id': 'one',
                                     'caption': 'fold clothes', 'views': views}]))
    index = tmp_path / 'index.npz'
    build_index(str(manifest), str(index))
    dataset = LatentPretrainDataset(data_path=str(index), processor=None, action_chunk_size=30,
                                    use_delta_action=False, latent_action_dim=608, num_view_slots=6)
    for latent_start in (0, 3, 17):                # source frames 0, 12, 68
        images = dataset.load_images(0, latent_start)
        source_frame = latent_start * 4
        assert source_frame < len(expected)
        np.testing.assert_array_equal(images['view0'][0].numpy(), expected[source_frame])

