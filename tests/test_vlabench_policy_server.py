"""VLABench policy-server wire protocol and helpers (model-free).

The server's own module docstring specifies the framing in full; ``_SpecClient`` below is a
straight implementation of that specification, so these tests check the server against the
documented protocol rather than against a copy of its own internals. What they cannot prove is
byte-level interop with the (unbundled) benchmark harness's client -- set
``VLABENCH_HARNESS_SCRIPTS`` to a directory containing ``policy_client.py`` and
``test_roundtrip_with_the_bundled_harness_client`` runs the same round trip through the real
thing.

Loading a checkpoint needs a GPU, transformers-5.2 and a staged export, so
``RynnVLABenchPolicy`` itself is exercised only through a stub infer function.
"""

import importlib.util
import os
import pickle
import socket
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


server = _load("vlabench_policy_server_under_test", REPO_ROOT / "scripts/vlabench_policy_server.py")


class _SpecClient:
    """The wire protocol from the server docstring, implemented from that text alone.

    Deliberately does not import anything from the server module: if the framing here and the
    framing in ``_InferHandler`` drift, these tests fail, which is the point.
    """

    def __init__(self, remote_ip, port):
        self.address = (remote_ip, port)
        self.conn = None

    def wait_for_connection(self, timeout=10.0):
        try:
            self.conn = socket.create_connection(self.address, timeout=timeout)
            return True
        except OSError:
            self.conn = None
            return False

    def request(self, payload, timeout=30.0):
        self.conn.settimeout(timeout)
        body = pickle.dumps(payload)
        self.conn.sendall(struct.pack(">I", len(body) + 1) + server.REQUEST_TYPE + body)
        header = self._recv(4)
        (length,) = struct.unpack(">I", header)
        reply = self._recv(length)
        assert reply[:1] == server.RESPONSE_TYPE, reply[:1]
        return pickle.loads(reply[1:])

    def _recv(self, count):
        chunks = []
        got = 0
        while got < count:
            part = self.conn.recv(count - got)
            if not part:
                raise ConnectionError(f"server closed after {got}/{count} bytes")
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def _harness_client():
    root = os.environ.get("VLABENCH_HARNESS_SCRIPTS")
    if not root:
        return None
    path = Path(root) / "policy_client.py"
    if not path.is_file():
        pytest.skip(f"VLABENCH_HARNESS_SCRIPTS={root!r} has no policy_client.py")
    return _load("harness_policy_client", path).PolicyClient


@pytest.fixture
def serving():
    """Run the server's socket layer with a stub policy; yield a connected spec client."""
    errors = []

    def infer_func(payload):
        try:
            assert payload["mode"] == "sync"
            obs = payload["obs"]
            assert set(obs["images"]) == {"image", "second_image", "wrist_image"}
            assert np.asarray(obs["state"]).shape == (7,)
            assert isinstance(obs["prompt"], str) and obs["prompt"]
            torch_seed = payload["policy_seed"]
            # Echo a (T, 7) chunk whose first row encodes the seed, proving the payload round-trips.
            chunk = np.zeros((3, 7), dtype=np.float32)
            chunk[0, 0] = float(torch_seed % 1000)
            chunk[:, 6] = 1.0
            # The real policy returns .tolist(); the stub must match or these tests would
            # exercise a wire contract the server never sends.
            return chunk.tolist()
        except BaseException as exc:  # surfaced to the test, not swallowed by the handler thread
            errors.append(exc)
            raise

    httpd = server._InferServer(("127.0.0.1", 0), server._InferHandler)
    httpd.infer_func = infer_func
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    client = _SpecClient("127.0.0.1", port)
    try:
        assert client.wait_for_connection(timeout=10)
        yield client, errors
    finally:
        client.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _obs(seed_prompt="pick up the red block"):
    return {
        "mode": "sync",
        "policy_seed": 12345,
        "obs": {
            "images": {name: np.zeros((224, 224, 3), dtype=np.uint8)
                       for name in ("image", "second_image", "wrist_image")},
            "state": np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "prompt": seed_prompt,
        },
    }


def test_roundtrip_over_the_documented_protocol(serving):
    client, errors = serving
    reply = client.request(_obs(), timeout=30)
    assert errors == []
    assert isinstance(reply, list), "reply must be numpy-free; see test_reply_is_numpy_version_neutral"
    # Exactly what a validating client does with it.
    chunk = np.asarray(reply, dtype=np.float32)
    assert chunk.shape == (3, 7) and chunk.dtype == np.float32
    assert np.isfinite(chunk).all()
    assert chunk[0, 0] == pytest.approx(12345 % 1000)   # policy_seed reached the server
    assert np.all(chunk[:, 6] == 1.0)                    # gripper channel survives the round trip


def test_roundtrip_with_the_bundled_harness_client():
    """Interop against the harness's own frozen client, when it is available.

    This is the only check that can catch a framing detail the docstring got wrong, so it is
    kept behind an env var rather than deleted: without it every assertion above is the server
    agreeing with a re-description of itself.
    """
    policy_client = _harness_client()
    if policy_client is None:
        pytest.skip("set VLABENCH_HARNESS_SCRIPTS to a directory containing policy_client.py")
    httpd = server._InferServer(("127.0.0.1", 0), server._InferHandler)
    httpd.infer_func = lambda payload: np.zeros((3, 7), dtype=np.float32).tolist()
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    client = None
    try:
        client = policy_client(remote_ip="127.0.0.1", port=httpd.server_address[1])
        assert client.wait_for_connection(timeout=10)
        reply = client.request(_obs(), timeout=30)
        assert np.asarray(reply, dtype=np.float32).shape == (3, 7)
    finally:
        if client is not None:
            client.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_reply_is_numpy_version_neutral():
    """Regression: the policy and the simulator run in different interpreters.

    A pickled ndarray embeds its own module path, so a numpy-2.x server sends "numpy._core..."
    references that a numpy-1.25 simulator cannot import. That failed every episode of the first
    real end-to-end smoke with "ModuleNotFoundError: No module named 'numpy._core'" raised inside
    the client's pickle.loads -- invisible to same-interpreter tests, so pin the wire type here.
    """
    array = np.zeros((3, 7), dtype=np.float32)
    assert b"numpy" in pickle.dumps(array)              # the failure mode
    assert b"numpy" not in pickle.dumps(array.tolist())  # what the server actually sends
    assert b"_core" not in pickle.dumps(array.tolist())


def test_persistent_connection_serves_repeat_queries(serving):
    client, errors = serving
    first = client.request(_obs(), timeout=30)
    second = client.request(_obs("select the painting"), timeout=30)
    assert errors == []
    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)
    assert client.conn is not None  # the harness reuses one connection; the handler must not close it


def test_wire_framing_header_covers_type_plus_body(serving):
    """Byte-level check of the documented framing.

    The 4-byte header counts the type byte plus the pickled body, so reading exactly `length`
    bytes and stripping one must yield the reply.
    """
    client, _ = serving
    payload = pickle.dumps(_obs())
    request = struct.pack(">I", len(payload) + 1) + b"\x00" + payload
    raw = socket.create_connection(("127.0.0.1", client.address[1]), timeout=30)
    try:
        raw.sendall(request)
        header = b""
        while len(header) < 4:
            header += raw.recv(4 - len(header))
        assert len(header) == 4
        (length,) = struct.unpack(">I", header)
        assert 1 <= length <= server.MAX_MESSAGE_BYTES
        body = b""
        while len(body) < length:
            body += raw.recv(length - len(body))
        assert len(body) == length          # header length covers exactly type + body
        assert body[:1] == server.RESPONSE_TYPE
        reply = pickle.loads(body[1:])      # and stripping one byte leaves valid pickle
        chunk = np.asarray(reply, dtype=np.float32)
        assert chunk.shape == (3, 7) and chunk.dtype == np.float32
    finally:
        raw.close()


def test_unknown_message_type_closes_connection(serving):
    client, _ = serving
    payload = pickle.dumps(_obs())
    raw = socket.create_connection(("127.0.0.1", client.address[1]), timeout=30)
    try:
        raw.sendall(struct.pack(">I", len(payload) + 1) + b"\x07" + payload)
        raw.settimeout(30)
        assert raw.recv(4) == b""  # server drops the connection instead of answering
    finally:
        raw.close()


def test_oversized_length_is_refused(serving):
    client, _ = serving
    raw = socket.create_connection(("127.0.0.1", client.address[1]), timeout=30)
    try:
        raw.sendall(struct.pack(">I", server.MAX_MESSAGE_BYTES + 1))
        raw.settimeout(30)
        assert raw.recv(4) == b""
    finally:
        raw.close()


def test_interpolate_actions():
    chunk = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(server.interpolate_actions(chunk, 1), chunk)
    up = server.interpolate_actions(chunk, 2)
    assert up.shape == (3, 2)
    np.testing.assert_allclose(up[:, 0], [0.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(up[:, 1], [0.0, 2.0, 4.0], atol=1e-6)
    assert up.dtype == np.float32


@pytest.mark.parametrize("ip,allow,expected", [
    ("127.0.0.1", False, "127.0.0.1"),
    ("localhost", False, "localhost"),
    ("::1", False, "::1"),
    ("0.0.0.0", True, "0.0.0.0"),
])
def test_resolve_bind_host_permits(ip, allow, expected):
    assert server.resolve_bind_host(ip, allow) == expected


@pytest.mark.parametrize("ip", ["0.0.0.0", "10.0.0.5", "172.17.0.1"])
def test_resolve_bind_host_refuses_reachable_without_opt_in(ip):
    """pickle deserialization is RCE; a reachable bind needs an explicit opt-in."""
    with pytest.raises(ValueError, match="unpickles every request"):
        server.resolve_bind_host(ip, False)


def test_cli_accepts_the_harness_command_line():
    """A benchmark driver is documented to build this exact argv; a rejected flag fails every
    eval at startup, long after the GPUs are allocated."""
    args = server.build_arg_parser().parse_args([
        "--model-path", "/tmp/staged", "--direct", "--weights", "ema", "--device", "cuda:0",
        "--port", "0", "--ready-file", "/tmp/ready.json",
        "--attn-implementation", "flash_attention_2", "--num-steps", "10", "--seed", "0"])
    assert str(args.model_path) == "/tmp/staged"
    assert args.weights == "ema" and args.num_steps == 10 and args.port == 0
    assert args.ip == "127.0.0.1" and not args.allow_remote_bind


def test_ready_file_reports_bound_port(tmp_path):
    ready = tmp_path / "nested" / "ready.json"
    server.write_ready_file(ready, "127.0.0.1", 47123)
    import json
    record = json.loads(ready.read_text())
    assert record["port"] == 47123  # the driver reads exactly this key
    assert record["host"] == "127.0.0.1" and record["pid"] > 0
    assert Path(record["code_root"]) == REPO_ROOT
