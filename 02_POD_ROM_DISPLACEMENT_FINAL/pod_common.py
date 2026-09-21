from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import re
from typing import Dict, Iterable, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi
VALID_REPRESENTATIONS = {"direct", "global_residual", "local_residual"}


def decode_scalar_string(x) -> str:
    x = np.asarray(x).reshape(-1)[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def scalar(z, key: str) -> float:
    return float(np.asarray(z[key]).reshape(-1)[0])


def canonical_representation(value: str) -> str:
    rep = str(value).strip().lower()
    aliases = {
        "global": "global_residual",
        "local": "local_residual",
        "residual_global": "global_residual",
        "residual_local": "local_residual",
    }
    rep = aliases.get(rep, rep)
    if rep not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"unsupported representation={value!r}; expected one of {sorted(VALID_REPRESENTATIONS)}"
        )
    return rep


def list_npz_files(data_dir: Path):
    files = sorted(data_dir.glob("trajectory_bend_Base_TC4_sample_*.npz"))
    if not files:
        files = sorted(data_dir.glob("*.npz"))
    return [p for p in files if p.is_file()]


def list_split_files(data_dir: Path, split: str):
    files = []
    for p in list_npz_files(data_dir):
        with np.load(p, allow_pickle=False) as z:
            if "split" not in z.files:
                continue
            s = decode_scalar_string(z["split"]).strip().lower()
        if s == split:
            files.append(p)
    return files


def sample_id_from_path(path: Path) -> int:
    m = re.search(r"sample[_-]?0*(\d+)", path.name, flags=re.I)
    return -1 if not m else int(m.group(1))


def dataset_signature(files: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        st = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(str(st.st_size).encode("ascii"))
        h.update(str(st.st_mtime_ns).encode("ascii"))
    return h.hexdigest()[:20]


def principal_axis(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    k = np.argmax(np.abs(axis))
    if axis[k] < 0:
        axis = -axis
    return axis


def make_cross_basis(axis: np.ndarray):
    candidates = np.eye(3, dtype=np.float64)
    seed = candidates[np.argmin(np.abs(candidates @ axis))]
    e1 = seed - np.dot(seed, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def build_structured_parameterization(X: np.ndarray, cells: np.ndarray):
    """Recover geometry-derived [axial station, circumferential node] indexing."""
    X = np.asarray(X, dtype=np.float64)
    cells = np.asarray(cells)

    n_nodes = X.shape[0]
    n_cells = cells.shape[0]
    n_circ = n_nodes - n_cells
    if n_circ <= 0 or n_nodes % n_circ != 0:
        raise RuntimeError(f"invalid tube grid: nodes={n_nodes}, cells={n_cells}")
    n_sta = n_nodes // n_circ
    if (n_sta - 1) * n_circ != n_cells:
        raise RuntimeError("structured tube-grid count relation failed")

    axis = principal_axis(X)
    e1, e2 = make_cross_basis(axis)
    origin = X.mean(axis=0)
    s_node = (X - origin) @ axis

    axial_order = np.argsort(s_node)
    station_ids_raw = axial_order.reshape(n_sta, n_circ)

    grid_ids = np.empty_like(station_ids_raw)
    phi_src = np.empty((n_sta, n_circ), dtype=np.float64)
    station_s = np.empty(n_sta, dtype=np.float64)

    for j in range(n_sta):
        ids = station_ids_raw[j]
        pts = X[ids]
        center = pts.mean(axis=0)
        station_s[j] = s_node[ids].mean()
        radial = pts - center
        phi = np.mod(np.arctan2(radial @ e2, radial @ e1), TWO_PI)
        circ_order = np.argsort(phi)
        grid_ids[j] = ids[circ_order]
        phi_src[j] = phi[circ_order]

    s_src = (station_s - station_s[0]) / (station_s[-1] - station_s[0])
    if np.any(np.diff(s_src) <= 0):
        raise RuntimeError("recovered axial coordinate is not strictly increasing")
    return grid_ids, s_src, phi_src


def linear_weights(x_src: np.ndarray, x_dst: np.ndarray):
    x_src = np.asarray(x_src, dtype=np.float64)
    x_dst = np.asarray(x_dst, dtype=np.float64)
    idx1 = np.searchsorted(x_src, x_dst, side="right")
    idx1 = np.clip(idx1, 1, len(x_src) - 1)
    idx0 = idx1 - 1
    x0, x1 = x_src[idx0], x_src[idx1]
    w = (x_dst - x0) / np.maximum(x1 - x0, 1e-15)
    return idx0, idx1, w.astype(np.float32)


def periodic_weights(phi_src: np.ndarray, phi_dst: np.ndarray):
    phi_src = np.asarray(phi_src, dtype=np.float64)
    phi_dst = np.mod(np.asarray(phi_dst, dtype=np.float64), TWO_PI)
    n = len(phi_src)
    xp = np.concatenate(([phi_src[-1] - TWO_PI], phi_src, [phi_src[0] + TWO_PI]))
    source_ids = np.concatenate(([n - 1], np.arange(n), [0]))
    j = np.searchsorted(xp, phi_dst, side="right") - 1
    j = np.clip(j, 0, len(xp) - 2)
    x0, x1 = xp[j], xp[j + 1]
    idx0, idx1 = source_ids[j], source_ids[j + 1]
    w = (phi_dst - x0) / np.maximum(x1 - x0, 1e-15)
    return idx0, idx1, w.astype(np.float32)


def native_to_fixed(U_native, grid_ids, s_src, phi_src, n_s_dst: int, n_phi_dst: int):
    U_native = np.asarray(U_native, dtype=np.float32)
    T = U_native.shape[0]
    n_sta, n_circ = grid_ids.shape
    U_grid = U_native[:, grid_ids.reshape(-1), :].reshape(T, n_sta, n_circ, 3)

    phi_dst = np.arange(n_phi_dst, dtype=np.float64) * TWO_PI / n_phi_dst
    U_phi = np.empty((T, n_sta, n_phi_dst, 3), dtype=np.float32)
    for j in range(n_sta):
        i0, i1, w = periodic_weights(phi_src[j], phi_dst)
        U_phi[:, j] = (
            U_grid[:, j, i0, :] * (1.0 - w[None, :, None])
            + U_grid[:, j, i1, :] * w[None, :, None]
        )

    s_dst = np.linspace(0.0, 1.0, n_s_dst, dtype=np.float64)
    i0, i1, w = linear_weights(s_src, s_dst)
    return (
        U_phi[:, i0] * (1.0 - w[None, :, None, None])
        + U_phi[:, i1] * w[None, :, None, None]
    ).astype(np.float32)


def fixed_to_native(U_fixed, grid_ids, s_src, phi_src):
    U_fixed = np.asarray(U_fixed, dtype=np.float32)
    T, n_s_dst, n_phi_dst, _ = U_fixed.shape
    n_sta, n_circ = grid_ids.shape
    n_nodes = grid_ids.size

    s_dst = np.linspace(0.0, 1.0, n_s_dst, dtype=np.float64)
    phi_dst = np.arange(n_phi_dst, dtype=np.float64) * TWO_PI / n_phi_dst

    i0, i1, w = linear_weights(s_dst, s_src)
    U_axial = (
        U_fixed[:, i0] * (1.0 - w[None, :, None, None])
        + U_fixed[:, i1] * w[None, :, None, None]
    )

    U_grid_rec = np.empty((T, n_sta, n_circ, 3), dtype=np.float32)
    for j in range(n_sta):
        k0, k1, wc = periodic_weights(phi_dst, phi_src[j])
        U_grid_rec[:, j] = (
            U_axial[:, j, k0, :] * (1.0 - wc[None, :, None])
            + U_axial[:, j, k1, :] * wc[None, :, None]
        )

    out = np.empty((T, n_nodes, 3), dtype=np.float32)
    out[:, grid_ids.reshape(-1), :] = U_grid_rec.reshape(T, n_nodes, 3)
    return out


def _frame_angles(z, T: int):
    if "angle_deg" in z.files:
        angle = np.asarray(z["angle_deg"], dtype=np.float32).reshape(-1)
    else:
        angle = np.linspace(0.0, 180.0, T, dtype=np.float32)
    if angle.size != T:
        raise RuntimeError(f"angle_deg length {angle.size} != frames {T}")

    if "angle_progress" in z.files:
        tau = np.asarray(z["angle_progress"], dtype=np.float32).reshape(-1)
        if tau.size != T:
            raise RuntimeError(f"angle_progress length {tau.size} != frames {T}")
    else:
        tau = angle / max(float(angle[-1]), 1.0)
    return angle, tau


def analytical_fields(X_ref: np.ndarray, R_bending: float, angle_deg: np.ndarray):
    """Analytical baseline used by FINAL_BENDING_EXPERIMENTS.

    X_ref is the tube reference position in canonical world coordinates [x,y,s].
    Returns analytical displacement, psi, cos(psi), sin(psi).
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    angle_deg = np.asarray(angle_deg, dtype=np.float64).reshape(-1)
    r = float(R_bending)
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"invalid R_bending={R_bending}")

    theta = np.deg2rad(angle_deg)[:, None]
    x = X_ref[:, 0][None, :]
    y = X_ref[:, 1][None, :]
    s = X_ref[:, 2][None, :]

    psi = np.clip(theta - s / r, 0.0, None)
    psi = np.minimum(psi, theta)
    c = np.cos(psi)
    sn = np.sin(psi)
    u = s - r * (theta - psi)

    A = np.empty((len(angle_deg), X_ref.shape[0], 3), dtype=np.float64)
    A[..., 0] = x
    A[..., 1] = -r + (r + y) * c + u * sn
    A[..., 2] = -(r + y) * sn + u * c
    U_ana = A - X_ref[None, :, :]
    return U_ana.astype(np.float32), psi.astype(np.float32), c.astype(np.float32), sn.astype(np.float32)


def world_residual_to_local(e_global: np.ndarray, c: np.ndarray, sn: np.ndarray):
    e_global = np.asarray(e_global, dtype=np.float32)
    out = np.empty_like(e_global)
    out[..., 0] = e_global[..., 0]
    out[..., 1] = c * e_global[..., 1] - sn * e_global[..., 2]
    out[..., 2] = sn * e_global[..., 1] + c * e_global[..., 2]
    return out


def local_residual_to_world(e_local: np.ndarray, c: np.ndarray, sn: np.ndarray):
    e_local = np.asarray(e_local, dtype=np.float32)
    out = np.empty_like(e_local)
    out[..., 0] = e_local[..., 0]
    out[..., 1] = c * e_local[..., 1] + sn * e_local[..., 2]
    out[..., 2] = -sn * e_local[..., 1] + c * e_local[..., 2]
    return out


def free_tube_mask_from_npz(z, n_tube: int) -> np.ndarray:
    if "node_type" not in z.files:
        raise RuntimeError("node_type is required for free-tube evaluation")
    nt = np.asarray(z["node_type"]).reshape(-1)
    if "tube_global_index" in z.files:
        idx = np.asarray(z["tube_global_index"], dtype=np.int64).reshape(-1)
        if idx.size != n_tube:
            raise RuntimeError(f"tube_global_index length {idx.size} != tube nodes {n_tube}")
        if np.any(idx < 0) or np.any(idx >= nt.size):
            raise RuntimeError("tube_global_index out of bounds")
        tube_nt = nt[idx]
    elif nt.size == n_tube:
        tube_nt = nt
    else:
        raise RuntimeError("cannot map global node_type to tube nodes")
    free = tube_nt == 0
    if not np.any(free):
        raise RuntimeError("free tube mask is empty")
    return free


@dataclass
class CaseData:
    path: Path
    sample_id: int
    split: str
    X_ref: np.ndarray
    U_true: np.ndarray
    representation_native: np.ndarray
    representation_fixed: np.ndarray
    grid_ids: np.ndarray
    s_src: np.ndarray
    phi_src: np.ndarray
    angle_deg: np.ndarray
    tau: np.ndarray
    geom: np.ndarray
    free_mask: np.ndarray
    U_ana: np.ndarray
    cos_psi: np.ndarray
    sin_psi: np.ndarray
    R_bending: float


def load_case(path: Path, n_s: int, n_phi: int, representation: str) -> CaseData:
    rep = canonical_representation(representation)
    with np.load(path, allow_pickle=False) as z:
        required = ["tube_mesh_pos", "tube_cells", "U_tube", "split", "D_outer", "t_over_D", "R_over_D", "R_bending", "node_type"]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"{path.name}: missing keys {missing}")

        X_ref = np.asarray(z["tube_mesh_pos"], dtype=np.float32)
        cells = np.asarray(z["tube_cells"])
        U_true = np.asarray(z["U_tube"], dtype=np.float32)
        split = decode_scalar_string(z["split"]).strip().lower()
        D = scalar(z, "D_outer")
        tD = scalar(z, "t_over_D")
        RD = scalar(z, "R_over_D")
        R = scalar(z, "R_bending")
        angle, tau = _frame_angles(z, U_true.shape[0])
        free_mask = free_tube_mask_from_npz(z, X_ref.shape[0])

    if U_true.ndim != 3 or U_true.shape[1:] != X_ref.shape:
        raise RuntimeError(f"{path.name}: U_tube shape {U_true.shape} incompatible with X_ref {X_ref.shape}")
    if not np.isfinite(X_ref).all() or not np.isfinite(U_true).all():
        raise RuntimeError(f"{path.name}: non-finite displacement data")
    if abs(R / D - RD) > 1e-4:
        raise RuntimeError(f"{path.name}: R_bending/D_outer inconsistent with R_over_D")

    U_ana, psi, c, sn = analytical_fields(X_ref, R, angle)
    if float(np.max(np.abs(U_ana[0]))) > 5e-5:
        raise RuntimeError(f"{path.name}: analytical frame0 is not identity")

    if rep == "direct":
        field_native = U_true
    else:
        e_global = U_true - U_ana
        if rep == "global_residual":
            field_native = e_global
        else:
            field_native = world_residual_to_local(e_global, c, sn)
            rec = local_residual_to_world(field_native, c, sn)
            max_rt = float(np.max(np.abs(rec - e_global)))
            if max_rt > 2e-5:
                raise RuntimeError(f"{path.name}: local residual roundtrip failed: {max_rt}")

    grid_ids, s_src, phi_src = build_structured_parameterization(X_ref, cells)
    field_fixed = native_to_fixed(field_native, grid_ids, s_src, phi_src, n_s, n_phi)

    return CaseData(
        path=path,
        sample_id=sample_id_from_path(path),
        split=split,
        X_ref=X_ref,
        U_true=U_true,
        representation_native=field_native.astype(np.float32),
        representation_fixed=field_fixed.astype(np.float32),
        grid_ids=grid_ids,
        s_src=s_src,
        phi_src=phi_src,
        angle_deg=angle,
        tau=tau,
        geom=np.asarray([D, tD, RD], dtype=np.float32),
        free_mask=free_mask,
        U_ana=U_ana,
        cos_psi=c,
        sin_psi=sn,
        R_bending=R,
    )


def representation_native_to_displacement(case: CaseData, field_native: np.ndarray, representation: str):
    rep = canonical_representation(representation)
    field_native = np.asarray(field_native, dtype=np.float32)
    if field_native.shape != case.U_true.shape:
        raise RuntimeError(f"field shape {field_native.shape} != U_true {case.U_true.shape}")
    if rep == "direct":
        return field_native
    if rep == "global_residual":
        return case.U_ana + field_native
    e_global = local_residual_to_world(field_native, case.cos_psi, case.sin_psi)
    return case.U_ana + e_global


def case_features(path: Path):
    with np.load(path, allow_pickle=False) as z:
        D = scalar(z, "D_outer")
        tD = scalar(z, "t_over_D")
        RD = scalar(z, "R_over_D")
        T = int(np.asarray(z["U_tube"]).shape[0])
        angle, tau = _frame_angles(z, T)
    return np.asarray([D, tD, RD], dtype=np.float32), angle, tau


def json_dump(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
