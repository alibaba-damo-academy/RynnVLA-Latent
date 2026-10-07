from typing import Optional

import torch


_STATS_KEYS = ("mean", "std", "min", "max", "q01", "q99", "count")


def _is_leaf(value):
    return isinstance(value, dict) and "dim" in value


def _merge_leaf(a, b):
    """Merge two finalized leaves, combining their stats and preserving the
    structural metadata shared between them."""
    n1, n2 = a["count"], b["count"]
    n = n1 + n2

    mean1 = torch.tensor(a["mean"], dtype=torch.float64)
    mean2 = torch.tensor(b["mean"], dtype=torch.float64)
    std1 = torch.tensor(a["std"], dtype=torch.float64)
    std2 = torch.tensor(b["std"], dtype=torch.float64)

    mean = (n1 * mean1 + n2 * mean2) / n
    var = (n1 * (std1 ** 2 + (mean1 - mean) ** 2) + n2 * (std2 ** 2 + (mean2 - mean) ** 2)) / n

    out = {k: v for k, v in a.items() if k not in _STATS_KEYS}
    out["mean"] = mean.tolist()
    out["std"] = var.sqrt().tolist()
    out["min"] = [min(x, y) for x, y in zip(a["min"], b["min"])]
    out["max"] = [max(x, y) for x, y in zip(a["max"], b["max"])]
    # Quantiles are not a mergeable statistic: the reservoir that produced them was
    # discarded at finalize time, so the pooled q01/q99 cannot be recovered exactly.
    # Fall back to a count-weighted average -- exact when the sources share a
    # distribution, otherwise bounded between the two sources' values. A
    # single-dataset mixture never reaches this function and keeps the exact
    # reservoir quantiles.
    if "q01" in a and "q01" in b:
        out["q01"] = [(n1 * x + n2 * y) / n for x, y in zip(a["q01"], b["q01"])]
        out["q99"] = [(n1 * x + n2 * y) / n for x, y in zip(a["q99"], b["q99"])]
    out["count"] = n
    return out


def _merge_schema_node(dst, src):
    """Recursively merge ``src`` schema node into ``dst`` in place."""
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        elif _is_leaf(v):
            assert _is_leaf(dst[k]), f"Type mismatch for '{k}'"
            dst[k] = _merge_leaf(dst[k], v)
        elif isinstance(v, dict):
            assert isinstance(dst[k], dict) and not _is_leaf(dst[k]), \
                f"Type mismatch for '{k}'"
            _merge_schema_node(dst[k], v)
        else:
            assert dst[k] == v, \
                f"Inconsistent schema value for '{k}': {dst[k]!r} vs {v!r}"


def _merge_schemas(schema_list):
    assert len(schema_list) > 0
    merged = schema_list[0]
    for other in schema_list[1:]:
        for robot_type, robot_schema in other.items():
            if robot_type not in merged:
                merged[robot_type] = robot_schema
            else:
                _merge_schema_node(merged[robot_type], robot_schema)
    return merged


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, weights=None):
        super().__init__(datasets)
        # Weighted mixing: build a virtual index map so that uniform sampling over it
        # yields the requested per-dataset proportions (small datasets upsampled, large
        # ones downsampled). Disabled when weights are None or all equal.
        self._weighted = False
        self._vmap = None
        if weights is not None and len(weights) == len(datasets) and len(set(float(w) for w in weights)) > 1:
            self._build_weighted_index([float(w) for w in weights])

    def _build_weighted_index(self, weights):
        sizes = [len(d) for d in self.datasets]
        total = sum(sizes)
        wsum = sum(weights)
        # Target virtual count per dataset (keep overall scale ~= total samples).
        targets = [max(1, round((w / wsum) * total)) for w in weights]
        vmap = []
        for di, (n, t) in enumerate(zip(sizes, targets)):
            if n == 0:
                continue
            # Evenly spread t picks over [0, n) (handles both up- and down-sampling).
            for k in range(t):
                vmap.append((di, (k * n) // t))
        self._vmap = vmap
        self._weighted = True

    def __len__(self):
        if self._weighted:
            return len(self._vmap)
        return super().__len__()

    def __getitem__(self, idx):
        if self._weighted:
            di, ri = self._vmap[idx]
            return self.datasets[di][ri]
        return super().__getitem__(idx)

    def get_sequence_lengths(self, **kwargs):
        per = [
            d.get_sequence_lengths(**kwargs) if hasattr(d, "get_sequence_lengths") else [0] * len(d)
            for d in self.datasets
        ]
        if self._weighted:
            return [per[di][ri] for (di, ri) in self._vmap]
        out = []
        for p in per:
            out.extend(p)
        return out

    @property
    def processor(self):
        """Processor is stored on sub-datasets; return the first one."""
        for ds in self.datasets:
            if hasattr(ds, "processor"):
                return ds.processor
        return None

    @processor.setter
    def processor(self, value):
        """Propagate processor assignment to all sub-datasets so that
        each sub-dataset's __getitem__ can call the processor to produce
        input_ids (required by the DataCollator)."""
        for ds in self.datasets:
            if hasattr(ds, "processor") or isinstance(ds, type(self)):
                ds.processor = value

    def get_schema(
        self,
        num_workers: int = 8,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        per_dataset = [
            dataset.get_schema(num_workers=num_workers, process_group=process_group)
            for dataset in self.datasets if hasattr(dataset, "get_schema")
        ]
        if not per_dataset:
            return {"action": {}, "state": {}}
        return {
            "action": _merge_schemas([s["action"] for s in per_dataset]),
            "state": _merge_schemas([s["state"] for s in per_dataset]),
        }
