#!/usr/bin/env python3
"""RynnVLA-Latent VLABench policy server.

Serves a Stage-2 checkpoint over a socket so a VLABench runner can score it. The
simulator side is NOT bundled with this release: VLABench itself is a public
benchmark, but the orchestration that starts one policy server per GPU and drives
it against a pinned dm_control/MuJoCo VLABench tree is not. That side is
policy-agnostic -- the only model-coupled piece is this server -- so any runner
that speaks the wire protocol below can drive it, and the protocol is specified
here in full rather than left to be reverse-engineered.

Wire protocol::

    request : uint32be len(type + body) | b'\\x00' | pickle(payload)
    response: uint32be len(type + body) | b'\\x01' | pickle(result)

``payload`` is what ``eval_vlabench._obs_message`` + ``policy_seed.seeded_message``
build::

    {"mode": "sync", "policy_seed": int,
     "obs": {"images": {image, second_image, wrist_image},  # 224x224 RGB uint8 HWC
             "state": (7,),   # [ee_pos - base_pos(3), ee_euler_xyz_world(3), raw gripper flag]
             "prompt": str}}

The raw gripper flag (1 = closed, the upstream ``get_ee_open_state`` bug) is
inverted inside ``vlabench_state_to_robot_state``, matching the training dataset.

``result`` is a ``(T, 7)`` float32 chunk of ABSOLUTE targets in the training
layout ``[ee_pos_base_frame(3), ee_euler_xyz_world(3), gripper(1 = open)]``; the
client adds ``base_pos`` back, runs IK and binarizes the gripper at
``gripper_threshold``.

Only the explicit ``{images, state, prompt}`` observation form is accepted --
that is what the validated client sends.  Converting a raw VLABench observation
(``rgb`` (4, H, W, 3) + 8-D ``ee_state``) is deliberately left to
``eval_vlabench._obs_message``: duplicating its camera indexing and base-frame
subtraction here would be a second, independently-drifting copy of a convention
that has already caused silent train/eval skew.

``pickle`` deserialization executes arbitrary code, so this server binds to
loopback by default and is meant to be driven by the harness on the same node.

    python scripts/vlabench_policy_server.py --model-path <staged export> \
        --direct --weights ema --device cuda:0 --port 0 --ready-file /tmp/ready.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import socketserver
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rynnvla.api.predict import load_policy
from rynnvla.inference_wrappers.rynn_brain_vla import vlabench_observation_to_sample

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

REQUEST_TYPE = b"\x00"
RESPONSE_TYPE = b"\x01"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
ACTION_DIM = 7


def _recv_all(sock, count):
    """Read exactly count bytes, or None once the peer closes the connection."""
    data = bytearray()
    while len(data) < count:
        packet = sock.recv(count - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def interpolate_actions(actions, sample_stride=1):
    """Resample a (T, D) chunk to the stride the simulator steps at."""
    actions = np.asarray(actions, dtype=np.float32)
    if sample_stride == 1:
        return actions
    horizon, dims = actions.shape
    source = np.arange(horizon) * sample_stride
    target = np.arange(horizon * sample_stride - (sample_stride - 1))
    return np.stack(
        [np.interp(target, source, actions[:, dim]) for dim in range(dims)], axis=1
    ).astype(np.float32)


class RynnVLABenchPolicy:
    """Stage-2 policy serving absolute VLABench end-effector targets."""

    def __init__(self, model_path, device, dtype, attn_implementation, denoising_steps,
                 sample_stride=1, seed=0):
        if denoising_steps < 1:
            raise ValueError("--num-steps must be positive")
        if sample_stride < 1:
            raise ValueError("--sample-stride must be positive")
        self.device = device
        self.denoising_steps = int(denoising_steps)
        self.sample_stride = int(sample_stride)
        # load_policy validates the device/attn combination, seeds every RNG and
        # rejects a checkpoint whose exported schema is not the VLABench contract.
        self.wrapper = load_policy(
            argparse.Namespace(
                model_path=Path(model_path), seed=int(seed), device=device,
                dtype=dtype, attn_implementation=attn_implementation),
            schema="vlabench",
        )
        config = self.wrapper.model.config
        self.action_horizon = int(config.action_chunk_size)
        self._init_rtc()
        LOGGER.info("Loaded RynnVLA-Latent checkpoint: %s", model_path)
        LOGGER.info(
            "action_chunk_size=%s action_dim=%s use_latent_actions=%s "
            "use_latent_head_readout=%s num_view_slots=%s denoising_steps=%s sample_stride=%s",
            self.action_horizon, config.action_dim,
            getattr(config, "use_latent_actions", None),
            getattr(config, "use_latent_head_readout", None),
            getattr(config, "num_view_slots", None),
            self.denoising_steps, self.sample_stride,
        )

    def _init_rtc(self):
        """Wire up RTC (real-time chunking) boundary smoothing. OFF unless RYNNVLA_RTC=1.

        decode_rtc has existed in rynnvla/inference_wrappers/rynn_brain_vla.py since it was
        ported and has never had a caller; this is the first one. Two things make it non-trivial:

        * It is an exact IDENTITY at execution_horizon == action_chunk_size. _decode_rtc_inner
          does s = max(execution_horizon, delay_steps) and _rtc_soft_mask fills range(d, H-s);
          our client is synchronous so delay_steps = 0, and at eh == H that range is empty, so
          W == 0, correction == 0 and v_guided == v bit-for-bit. RTC only does something when
          the client executes a strict PREFIX of each chunk (eh < H), leaving an unexecuted tail
          to be consistent with.
        * The tail must be the model-space chunk, not the denormalized EE poses, and it must be
          sliced at [eh:] so that new-chunk index i lines up with old-chunk index eh+i. That is
          exactly the region where W is nonzero; _decode_rtc_inner zero-pads the rest.
        """
        self.rtc_eh = None
        self.rtc_beta = 10.0
        self._rtc_prev = None
        self._rtc_ep = None
        self._rtc_q = 0
        self._rtc_q = 0
        if os.environ.get("RYNNVLA_RTC", "").strip() not in ("1", "true", "True"):
            LOGGER.info("RTC off (set RYNNVLA_RTC=1 to enable); using the historical decode path")
            return
        if self.sample_stride != 1:
            # interpolate_actions resamples the chunk, so "the client consumed eh actions" no
            # longer maps onto model-space index eh. Refuse rather than misalign silently.
            raise ValueError(
                f"RYNNVLA_RTC=1 requires --sample-stride 1, got {self.sample_stride}: the "
                f"unexecuted tail could not be sliced at a model-space index")
        raw = os.environ.get("RYNNVLA_RTC_EH", "").strip()
        if not raw:
            raise ValueError("RYNNVLA_RTC=1 requires RYNNVLA_RTC_EH, and it must equal the "
                             "client's --execution-horizon exactly or the tail misaligns")
        self.rtc_eh = int(raw)
        if not 0 < self.rtc_eh < self.action_horizon:
            raise ValueError(
                f"RYNNVLA_RTC_EH={self.rtc_eh} must satisfy 0 < eh < action_chunk_size="
                f"{self.action_horizon}. At eh == chunk there is no unexecuted tail and "
                f"decode_rtc reduces to an exact identity (W == 0), so the run would be a "
                f"no-op that still costs an hour of GPU.")
        beta = os.environ.get("RYNNVLA_RTC_BETA", "").strip()
        if beta:
            self.rtc_beta = float(beta)
        if not os.environ.get("RYNNVLA_RTC_EPISODE_CONTEXT"):
            raise ValueError(
                "RYNNVLA_RTC=1 also requires RYNNVLA_RTC_EPISODE_CONTEXT=1 in the SIM process "
                "env, so the benchmark client sends rtc_context. Without an episode boundary the "
                "first query of every episode would be guided toward the previous episode's "
                "tail -- a different scene with different object poses.")
        LOGGER.info("RTC ON: execution_horizon=%d (chunk=%d, so %d of %d chunk slots are "
                    "re-planned per query), beta=%.1f, delay_steps=0 (synchronous client)",
                    self.rtc_eh, self.action_horizon, self.rtc_eh, self.action_horizon,
                    self.rtc_beta)

    def _rtc_request(self, request):
        """Build the decode_rtc kwargs for this query, or None to take the plain path."""
        if self.rtc_eh is None:
            return None
        ctx = request.get("rtc_context")
        if ctx is None:
            raise ValueError("RYNNVLA_RTC=1 but the request carries no rtc_context; export "
                             "RYNNVLA_RTC_EPISODE_CONTEXT=1 where the benchmark client runs")
        ep = (ctx.get("task"), ctx.get("episode"))
        if ep != self._rtc_ep:
            # New episode: the previous tail describes a different scene. Drop it and fall back
            # to the plain decode for this one query.
            self._rtc_ep = ep
            self._rtc_prev = None
            self._rtc_q = 0
        # The server cannot see the client's config, so a RYNNVLA_RTC_EH that disagrees with the
        # client's --execution-horizon would misalign the tail silently: we would slice [eh:] at
        # the wrong index and guide the new plan toward actions that were already executed. The
        # step counter in rtc_context detects it -- query k of an episode starts at env step k*eh.
        step = ctx.get("step")
        if step is not None and int(step) != self._rtc_q * self.rtc_eh:
            raise ValueError(
                f"RTC step mismatch: query #{self._rtc_q} of {ep} arrived at env step {step}, "
                f"expected {self._rtc_q * self.rtc_eh}. RYNNVLA_RTC_EH={self.rtc_eh} does not "
                f"match the client's --execution-horizon, so the unexecuted tail would be "
                f"sliced at the wrong index.")
        self._rtc_q += 1
        if self._rtc_prev is None:
            return None
        return {"prev_actions": self._rtc_prev, "delay_steps": 0,
                "execution_horizon": self.rtc_eh, "beta": self.rtc_beta}

    def _rtc_advance(self):
        """Keep the unexecuted tail of the chunk we just returned."""
        if self.rtc_eh is None:
            return
        acts = getattr(self.wrapper, "last_vlabench_model_actions", None)
        if acts is None or acts.ndim != 3 or acts.shape[1] <= self.rtc_eh:
            self._rtc_prev = None
            return
        self._rtc_prev = acts[:, self.rtc_eh:, :].contiguous()

    @torch.no_grad()
    def infer(self, request):
        if not isinstance(request, dict):
            raise ValueError("Policy request must be a mapping")
        if "policy_seed" in request:
            # Per-query seeding keeps the flow-matching noise a function of
            # (benchmark, split, track, seed, episode, task, step), so a rerun
            # reproduces the 4B harness numbers exactly.
            torch.manual_seed(int(request["policy_seed"]))
        mode = request.get("mode", "sync")
        if mode != "sync":
            raise ValueError(f"Unsupported inference mode {mode!r}; only 'sync' is supported")
        sample = vlabench_observation_to_sample(request.get("obs", request))
        # The @torch.no_grad() on this method is correct for both paths: decode_rtc opens its own
        # torch.inference_mode(False) and a local torch.enable_grad() around the autograd.grad call
        # that computes the guidance correction, so it does not need grad enabled out here.
        actions = np.asarray(
            self.wrapper.predict_vlabench(
                text=sample["text"], images=sample["images"], state=sample["state"],
                denoising_steps=self.denoising_steps, rtc=self._rtc_request(request),
            ),
            dtype=np.float32,
        )
        self._rtc_advance()
        if (actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not len(actions)
                or not np.isfinite(actions).all()):
            raise ValueError(f"expected finite nonempty (T, {ACTION_DIM}) actions, got {actions.shape}")
        # Return plain nested lists, never an ndarray. The simulator runs in a different
        # interpreter than the policy: pickled numpy arrays embed their own module path, so a
        # numpy-2.x server sends "numpy._core..." references that a numpy-1.25 simulator cannot
        # import (ModuleNotFoundError: No module named 'numpy._core'), failing every episode at
        # the response boundary. Both the client-side chunk validator and the simulator-side
        # eval script immediately do np.asarray(reply, dtype=np.float32), so a list is
        # contract-identical and immune to the version skew. The request direction is safe:
        # numpy 2.x still reads numpy 1.x pickles.
        return interpolate_actions(actions, self.sample_stride).tolist()

    def warmup(self):
        """One synthetic round trip before the ready file is published.

        The harness waits for the ready file and treats an early server exit as a
        startup failure, so any adapter, schema or shape bug surfaces here with a
        readable log line instead of as one error per simulated episode.
        """
        started = time.monotonic()
        images = {name: np.zeros((224, 224, 3), dtype=np.uint8)
                  for name in ("image", "second_image", "wrist_image")}
        state = np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        payload = {"mode": "sync", "policy_seed": 0,
                   "obs": {"images": images, "state": state, "prompt": "warmup"}}
        if self.rtc_eh is not None:
            # Two round trips, not one. The first has no previous chunk, so it takes the plain
            # decode path and only fills _rtc_prev; the second carries a tail and is the first
            # call that actually runs decode_rtc's autograd guidance. Warming only the plain path
            # would let a shape, dtype or grad bug in the guidance survive until the middle of a
            # scored episode. The synthetic episode key never collides with a real (task, episode)
            # pair, so the first real request is correctly treated as a boundary and starts clean.
            payload["rtc_context"] = {"task": "__warmup__", "episode": -1, "step": 0}
            self.infer(dict(payload))
            payload["rtc_context"] = {"task": "__warmup__", "episode": -1, "step": self.rtc_eh}
        actions = self.infer(payload)
        array = np.asarray(actions, dtype=np.float32)
        LOGGER.info("warmup chunk shape=%s in %.1fs (rtc=%s)", array.shape,
                    time.monotonic() - started, "on" if self.rtc_eh is not None else "off")
        return array


class _InferHandler(socketserver.BaseRequestHandler):
    """One persistent connection per client; inference is serialized by the server lock."""

    def handle(self):
        peer = self.client_address
        LOGGER.info("Client connected from %s", peer)
        while True:
            header = _recv_all(self.request, 4)
            if header is None:
                break
            (total,) = struct.unpack(">I", header)
            if not 1 <= total <= MAX_MESSAGE_BYTES:
                LOGGER.error("Client %s sent an invalid message length %d", peer, total)
                break
            raw = _recv_all(self.request, total)
            if raw is None:
                break
            if raw[:1] != REQUEST_TYPE:
                LOGGER.error("Client %s sent message type %r, expected %r", peer, raw[:1], REQUEST_TYPE)
                break
            try:
                reply = self.server.infer_func(pickle.loads(raw[1:]))
            except Exception:
                # Drop the connection rather than half-answer: the client treats a
                # short read as an episode error, which the harness counts explicitly.
                LOGGER.error("Inference error for %s:\n%s", peer, traceback.format_exc())
                break
            payload = pickle.dumps(reply)
            self.request.sendall(struct.pack(">I", len(payload) + 1) + RESPONSE_TYPE + payload)
        LOGGER.info("Client %s disconnected", peer)


class _InferServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def write_ready_file(path, host, port):
    """Direct write + fsync; the harness polls for this file to learn the bound port."""
    payload = json.dumps({"host": host, "port": port, "pid": os.getpid(),
                          "code_root": str(REPO_ROOT)}, indent=2) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_server(policy, host, port, ready_file=""):
    lock = threading.Lock()

    def handler(payload):
        with lock:
            return policy.infer(payload)

    server = _InferServer((host, port), _InferHandler)
    server.infer_func = handler
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    LOGGER.info("VLABench policy server listening on %s:%d (multi-client)", bound_host, bound_port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
    thread.start()
    policy.warmup()
    if ready_file:
        write_ready_file(ready_file, bound_host, bound_port)
        LOGGER.info("ready file written: %s", ready_file)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Server shutting down...")
    finally:
        server.shutdown()
        server.server_close()


def build_arg_parser():
    """Accept the command line a benchmark driver is already likely to use.

    ``--direct`` and ``--weights`` are part of that contract but carry no meaning
    here: Stage-2 checkpoints are always expert flow-matching, and the weight
    selection was already resolved when the checkpoint was exported. They are
    accepted, logged and otherwise inert, so a driver written for another policy
    does not need a model-specific command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, type=Path,
                        help="Exported Stage-2 checkpoint directory (rynnvla.api.export_checkpoint)")
    parser.add_argument("--direct", action="store_true",
                        help="Accepted for driver parity; Stage-2 exports are always direct")
    parser.add_argument("--weights", choices=("model", "ema"), default="ema",
                        help="Accepted for harness parity; already resolved by the export")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching integration steps")
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="Interpolate the chunk to this stride before returning it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation",
                        choices=("sdpa", "eager", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument("--ip", default="127.0.0.1",
                        help="Bind address; keep loopback -- the protocol pickles untrusted input")
    parser.add_argument("--allow-remote-bind", action="store_true",
                        help="Required to bind a non-loopback address; see the pickle warning above")
    parser.add_argument("--port", type=int, default=0, help="0 allocates an available OS port")
    parser.add_argument("--ready-file", default="",
                        help="JSON the harness polls for the bound port; published after warmup")
    parser.add_argument("--seed", type=int, default=0)
    return parser


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def resolve_bind_host(ip, allow_remote):
    """Refuse a reachable bind unless it was explicitly asked for.

    The harness protocol deserializes each request with pickle, which executes
    arbitrary code. The format is fixed by the frozen benchmark client, so the
    exposure is controlled by never listening where another host can connect.
    """
    if ip in _LOOPBACK_HOSTS or allow_remote:
        return ip
    raise ValueError(
        f"--ip {ip!r} is reachable from other hosts and this server unpickles every request; "
        "use the loopback default, or pass --allow-remote-bind only on a trusted isolated network"
    )


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if not args.direct:
        LOGGER.warning("--direct was not passed; Stage-2 exports are direct regardless")
    host = resolve_bind_host(args.ip, args.allow_remote_bind)
    if host != "127.0.0.1":
        LOGGER.warning("Binding %s: any host that can reach this port can execute code here", host)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 4294967296)
    policy = RynnVLABenchPolicy(
        model_path=args.model_path, device=args.device, dtype=args.dtype,
        attn_implementation=args.attn_implementation, denoising_steps=args.num_steps,
        sample_stride=args.sample_stride, seed=args.seed,
    )
    run_server(policy, host, args.port, args.ready_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
