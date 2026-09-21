"""Shared-raw packed-cache discovery/loading utilities for ANARESID X10 V2."""
from __future__ import annotations

import json
import os
from typing import Dict, List
import torch


def read_manifest(cache_dir: str, split: str) -> List[str]:
    manifest = os.path.join(cache_dir, f"{split}_manifest.txt")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with open(manifest, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    paths = [os.path.join(cache_dir, name) for name in names]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Manifest lists missing cache files:\n" + "\n".join(missing[:20])
        )
    return paths


def sample_id_from_cache_path(path: str) -> int:
    import re
    name = os.path.basename(str(path))
    m = re.search(r"sample[_-]?0*(\d+)", name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse sample id from cache path: {path}")
    return int(m.group(1))


def load_normalization_stats(stats_dir: str) -> Dict:
    path = os.path.join(stats_dir, "normalization_stats.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing normalization stats: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    required = (
        "edge_mean", "edge_std", "train_sample_ids",
        "shared_cache_signature", "feature_contract",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"TrainN normalization stats missing keys {missing}: {path}")

    return {
        "edge_mean": torch.tensor(raw["edge_mean"], dtype=torch.float32),
        "edge_std": torch.tensor(raw["edge_std"], dtype=torch.float32),
        "train_sample_ids": [int(x) for x in raw["train_sample_ids"]],
        "shared_cache_signature": str(raw["shared_cache_signature"]),
        "feature_contract": str(raw["feature_contract"]),
        "raw": raw,
    }


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
