"""Streaming readers for the two LatentPretrainDataset manifest formats.

* a JSON array -- the small-corpus manifest format
* JSONL, one episode per line -- what ``scripts/rebuild_latent_manifest.py`` writes for a
  full-corpus manifest

A full-corpus manifest is why this module exists: ``json.load`` costs roughly six times the
text size in RSS, because parsed JSON needs that much in Python objects (measured in
this repo: 281 MB of JSON -> 1.7 GB of objects; see latent_pretrain.py's module docstring).
Both formats are consumed incrementally, so neither ``build_index`` nor ``build_latent_stats``
ever holds more than one episode at a time.

Lives under ``utils`` rather than ``datasets.vla_datasets`` on purpose: the manifest builders
in ``scripts/`` import it, and that package's ``__init__`` pulls in torch.
"""

import json
from typing import Iterator, Optional

__all__ = ["iter_json_array", "iter_jsonl", "iter_manifest", "join_caption", "manifest_format"]


def iter_json_array(path: str, chunk_bytes: int = 1 << 25) -> Iterator[dict]:
    """Yield elements of a top-level JSON array without loading the file whole.

    Uses ``raw_decode`` at C speed; refills the buffer on truncation. Elements must be
    objects (the separator skip consumes ``[``, so nested top-level arrays are not supported
    -- the RynnVLA-Base schema never has them).
    """
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf, pos, eof = "", 0, False

        def refill() -> bool:
            nonlocal buf, pos, eof
            if eof:
                return False
            nxt = f.read(chunk_bytes)
            if not nxt:
                eof = True
                return False
            buf = buf[pos:] + nxt
            pos = 0
            return True

        while True:
            while True:  # skip whitespace / ',' / opening '['
                while pos < len(buf) and buf[pos] in " \t\r\n,[":
                    pos += 1
                if pos < len(buf):
                    break
                if not refill():
                    return
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except ValueError:
                    if not refill():
                        raise
            yield obj
            pos = end


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield one episode per line, reporting the line number of any malformed entry.

    A truncated line means the writer died mid-publish; naming the line is the difference
    between "the manifest is corrupt at 4,182,113" and an unparseable 10 GB file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: malformed manifest line: {exc}") from exc


def manifest_format(path: str) -> str:
    """'array', 'jsonl' or 'empty', from the first non-whitespace byte."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    first = next((c for c in head if not c.isspace()), "")
    if first == "[":
        return "array"
    if first == "{":
        return "jsonl"
    if not first:
        return "empty"
    raise ValueError(f"{path}: not a JSON array or JSONL manifest (starts with {first!r})")


def iter_manifest(path: str, fmt: Optional[str] = None) -> Iterator[dict]:
    """Stream a manifest in either format; pass ``fmt`` to skip re-sniffing the file."""
    fmt = fmt or manifest_format(path)
    if fmt == "array":
        return iter_json_array(path)
    if fmt == "jsonl":
        return iter_jsonl(path)
    return iter(())  # empty manifest: build_index reports 0 episodes, the caller decides


def join_caption(caption, max_chars: int) -> str:
    """Flatten a RynnVLA-Base caption segment list into one string.

    Segments are ordered by start_time, deduplicated (human-video corpora repeat one
    description across dozens of segments), joined with spaces and cut at a word boundary.
    Empty captions stay empty; build_index substitutes its fallback.
    """
    if isinstance(caption, str):
        return caption.strip()[:max_chars]
    if not isinstance(caption, list):
        return ""
    segs = []
    for s in caption:
        if isinstance(s, dict):
            text = (s.get("description") or "").strip()
            if text:
                segs.append((float(s.get("start_time") or 0.0), text))
        elif isinstance(s, str) and s.strip():
            segs.append((0.0, s.strip()))
    segs.sort(key=lambda x: x[0])
    seen, parts = set(), []
    for _, text in segs:
        if text not in seen:
            seen.add(text)
            parts.append(text)
    out = " ".join(parts)
    if len(out) > max_chars:
        cut = out[:max_chars]
        out = cut[: cut.rfind(" ") + 1].strip() or cut
    return out
