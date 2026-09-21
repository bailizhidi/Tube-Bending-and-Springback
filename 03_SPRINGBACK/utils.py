# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _scalar(x, default=None):
    try:
        a = np.asarray(x)
        if a.size == 0:
            return default
        v = a.reshape(-1)[0]
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return default


def parse_sample_id_from_name(path: str) -> Optional[int]:
    m = re.search(r"sample[_-]?(\d{4})", os.path.basename(path), re.I)
    return int(m.group(1)) if m else None


def parse_angle_from_name(path: str) -> Optional[float]:
    name = os.path.basename(path)
    for p in [r"angle[_-]?(\d{1,3})", r"(\d{1,3})deg"]:
        m = re.search(p, name, re.I)
        if m:
            return float(m.group(1))
    return None


def get_sample_id(z, path: str) -> int:
    for k in ("sample_id", "Sample_ID"):
        if k in z.files:
            try:
                return int(float(_scalar(z[k], -1)))
            except Exception:
                pass
    p = parse_sample_id_from_name(path)
    return -1 if p is None else p


def get_angle(z, path: str) -> float:
    if "angle_deg" in z.files:
        try:
            return float(_scalar(z["angle_deg"], np.nan))
        except Exception:
            pass
    p = parse_angle_from_name(path)
    return np.nan if p is None else float(p)


def get_scalar_npz(z, key: str, default: float) -> float:
    if key not in z.files:
        return float(default)
    try:
        v = float(_scalar(z[key], default))
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def get_array(z, key: str, shape: Tuple[int, int]) -> np.ndarray:
    if key not in z.files:
        return np.zeros(shape, dtype=np.float32)
    a = np.asarray(z[key], dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] != shape[0]:
        raise ValueError("{} first dimension mismatch: {} vs {}".format(key, a.shape[0], shape[0]))
    if a.shape[1] == shape[1]:
        return a
    if a.shape[1] > shape[1]:
        return a[:, :shape[1]]
    pad = np.zeros((shape[0], shape[1]-a.shape[1]), dtype=np.float32)
    return np.concatenate([a, pad], axis=1)


def get_x_bend(z) -> np.ndarray:
    return np.asarray(z["X_bend"], dtype=np.float32)


def get_x_spring(z) -> np.ndarray:
    return np.asarray(z["X_spring"], dtype=np.float32)


def get_target(z) -> np.ndarray:
    if "dU_springback" in z.files:
        return np.asarray(z["dU_springback"], dtype=np.float32)
    return get_x_spring(z) - get_x_bend(z)


def get_cells(z) -> np.ndarray:
    return np.asarray(z["cells"], dtype=np.int64)


def build_undirected_edges_from_cells(cells: np.ndarray, n: int) -> np.ndarray:
    edges = set()
    for c in np.asarray(cells):
        ids = [int(v) for v in c.tolist() if int(v) >= 0]
        for i in range(len(ids)):
            a, b = ids[i], ids[(i+1) % len(ids)]
            if a != b and 0 <= a < n and 0 <= b < n:
                edges.add((a,b)); edges.add((b,a))
    if not edges:
        raise ValueError("No graph edges built from cells")
    return np.asarray(sorted(edges), dtype=np.int64).T


def get_edge_index(z, n: int) -> np.ndarray:
    if "edge_index" in z.files:
        ei = np.asarray(z["edge_index"], dtype=np.int64)
        if ei.ndim == 2 and ei.shape[0] == 2:
            return ei
        if ei.ndim == 2 and ei.shape[1] == 2:
            return ei.T
    return build_undirected_edges_from_cells(get_cells(z), n)


def make_edge_features(pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    rel = pos[dst] - pos[src]
    length = np.linalg.norm(rel, axis=1, keepdims=True)
    return np.concatenate([rel, length], axis=1).astype(np.float32)


def make_node_features(z, path: str, feature_set: str = "geom") -> np.ndarray:
    X = get_x_bend(z)
    n = X.shape[0]
    clamp = np.asarray(z["clamp_mask"], dtype=np.float32).reshape(n,1)
    angle = get_angle(z, path)
    angle_norm = angle / 180.0
    t = get_scalar_npz(z, "Thickness", get_scalar_npz(z, "thickness_value", 1.0))
    D = get_scalar_npz(z, "D_outer", 8.0)
    R = get_scalar_npz(z, "R_bending", 24.0)
    scalars = np.repeat(np.asarray([[t,D,R,angle_norm]], dtype=np.float32), n, axis=0)

    fs = str(feature_set).strip().lower()
    if fs in {"geom", "a_geom"}:
        return np.concatenate([X, clamp, scalars], axis=1).astype(np.float32)
    if fs in {"stress_peeq", "b_stress_peeq"}:
        sm = get_array(z, "S_mises_bend", (n,1))
        peeq = get_array(z, "PEEQ_bend", (n,1))
        return np.concatenate([X, sm, peeq, clamp, scalars], axis=1).astype(np.float32)
    raise ValueError("Unsupported feature_set={!r}; use geom or stress_peeq".format(feature_set))


def read_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text_lines(lines: Sequence[str], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x) + "\n")


def read_text_lines(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def discover_npz_files(data_dir: str, usable_files: str = "") -> List[str]:
    if usable_files and os.path.exists(usable_files):
        raw = read_text_lines(usable_files)
        out = []
        for p in raw:
            if os.path.isabs(p) and os.path.exists(p): out.append(p)
            elif os.path.exists(p): out.append(os.path.abspath(p))
            else:
                q = os.path.join(data_dir, p)
                if os.path.exists(q): out.append(os.path.abspath(q))
        return sorted(out)
    return sorted(os.path.abspath(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith(".npz"))


def is_reasonable_file(path: str, cfg: Dict) -> Tuple[bool, List[str]]:
    issues = []
    try:
        with np.load(path, allow_pickle=False) as z:
            Xb, Xs, y = get_x_bend(z), get_x_spring(z), get_target(z)
            if Xb.ndim != 2 or Xb.shape[1] != 3: issues.append("bad_X_bend_shape")
            if Xs.shape != Xb.shape or y.shape != Xb.shape: issues.append("core_shape_mismatch")
            if not np.isfinite(Xb).all() or not np.isfinite(Xs).all() or not np.isfinite(y).all(): issues.append("nonfinite")
            if Xb.shape == Xs.shape == y.shape:
                c = np.max(np.linalg.norm((Xs-Xb)-y, axis=1))
                if c > float(cfg.get("du_consistency_tol", 1e-4)): issues.append("du_inconsistent_{:.3e}".format(c))
            maxdu = float(np.linalg.norm(y, axis=1).max())
            if maxdu <= float(cfg.get("min_max_du", 1e-10)): issues.append("du_nearly_zero")
            if maxdu >= float(cfg.get("max_reasonable_du", 50.0)): issues.append("du_too_large_{:.4g}".format(maxdu))
            ei = get_edge_index(z, Xb.shape[0])
            if ei.size == 0 or ei.min() < 0 or ei.max() >= Xb.shape[0]: issues.append("edge_index_invalid")
    except Exception as e:
        issues.append("read_error:{}".format(e))
    return len(issues) == 0, issues


def filter_reasonable_files(files: Sequence[str], cfg: Dict):
    good, bad = [], {}
    for f in files:
        ok, issues = is_reasonable_file(f, cfg)
        if ok: good.append(f)
        else: bad[f] = issues
    return good, bad


def split_by_sample_id(files: Sequence[str], cfg: Dict) -> Dict[str, List[str]]:
    id_to_files: Dict[int, List[str]] = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            sid = get_sample_id(z, f)
        id_to_files.setdefault(sid, []).append(f)
    train_ids = [int(x) for x in cfg.get("train_sample_ids", [])]
    val_ids = [int(x) for x in cfg.get("val_sample_ids", [])]
    test_ids = [int(x) for x in cfg.get("test_sample_ids", [])]
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError("Explicit 120/15/15 sample-id split is required in config")
    def collect(ids: Iterable[int]) -> List[str]:
        out = []
        for sid in ids: out.extend(sorted(id_to_files.get(sid, [])))
        return sorted(out)
    return {"train": collect(train_ids), "val": collect(val_ids), "test": collect(test_ids)}
