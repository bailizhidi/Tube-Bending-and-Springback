from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

EXPECTED_FRAMES = 181
REQUIRED = (
    "mesh_pos", "world_pos", "node_type", "is_tube_node", "is_tool_node",
    "sample_id", "geometry_id", "D_outer", "Thickness", "R_bending",
    "t_over_D", "R_over_D", "split",
)


def scalar(data, *keys):
    for k in keys:
        if k in data.files:
            v = np.asarray(data[k]).reshape(-1)[0]
            if isinstance(v, bytes):
                return v.decode("utf-8", "replace")
            return v.item() if hasattr(v, "item") else v
    raise KeyError(keys)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        sid = int(r["Sample_ID"])
        if sid in out:
            raise RuntimeError(f"Duplicate Sample_ID in CSV: {sid}")
        out[sid] = r
    return out


def sid_from_filename(path: Path):
    m = re.search(r"sample[_-]?0*(\d+)", path.name, flags=re.I)
    return None if not m else int(m.group(1))


def close(a, b, atol=2.0e-4):
    return abs(float(a) - float(b)) <= atol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--expect-samples", type=int, default=150)
    ap.add_argument("--strict", type=int, choices=[0,1], default=1)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    csv_rows = read_csv(args.csv)
    files = sorted(args.dataset_dir.glob("*.npz"))
    errors, warnings, rows = [], [], []

    if len(files) != args.expect_samples:
        errors.append(f"NPZ count {len(files)} != expected {args.expect_samples}")
    if len(csv_rows) != args.expect_samples:
        errors.append(f"CSV rows {len(csv_rows)} != expected {args.expect_samples}")

    seen = set()
    splits = Counter()

    for i, path in enumerate(files, 1):
        try:
            with np.load(path, allow_pickle=False) as z:
                missing = [k for k in REQUIRED if k not in z.files]
                if missing:
                    raise RuntimeError(f"missing keys {missing}")
                sid = int(scalar(z, "sample_id", "Sample_ID"))
                gid = int(scalar(z, "geometry_id", "Geometry_ID"))
                if sid in seen:
                    raise RuntimeError(f"duplicate Sample_ID={sid}")
                seen.add(sid)
                if sid not in csv_rows:
                    raise RuntimeError(f"Sample_ID={sid} absent from clean CSV")
                r = csv_rows[sid]

                mesh = np.asarray(z["mesh_pos"])
                world = np.asarray(z["world_pos"])
                nt = np.asarray(z["node_type"]).reshape(-1)
                tube = np.asarray(z["is_tube_node"]).reshape(-1).astype(bool)
                tool = np.asarray(z["is_tool_node"]).reshape(-1).astype(bool)

                if mesh.ndim != 2 or mesh.shape[1] != 3:
                    raise RuntimeError(f"mesh_pos shape={mesh.shape}")
                if world.shape != (EXPECTED_FRAMES, mesh.shape[0], 3):
                    raise RuntimeError(f"world_pos shape={world.shape}")
                if len(nt) != mesh.shape[0] or len(tube) != mesh.shape[0] or len(tool) != mesh.shape[0]:
                    raise RuntimeError("node metadata count mismatch")
                if np.any(tube & tool) or not np.all(tube | tool):
                    raise RuntimeError("tube/tool masks invalid")
                if not np.isfinite(mesh).all() or not np.isfinite(world).all():
                    raise RuntimeError("mesh_pos/world_pos contains NaN/Inf")

                split = str(scalar(z, "split", "Split")).strip().lower()
                splits[split] += 1

                checks = {
                    "geometry_id": (gid, int(r["Geometry_ID"])),
                    "D_outer": (float(scalar(z, "D_outer")), float(r["D_outer"])),
                    "Thickness": (float(scalar(z, "Thickness")), float(r["Thickness"])),
                    "R_bending": (float(scalar(z, "R_bending")), float(r["R_bending"])),
                    "t_over_D": (float(scalar(z, "t_over_D")), float(r["t_over_D"])),
                    "R_over_D": (float(scalar(z, "R_over_D")), float(r["R_over_D"])),
                }
                for name, (a,b) in checks.items():
                    if name == "geometry_id":
                        if a != b:
                            raise RuntimeError(f"{name}: NPZ={a} CSV={b}")
                    elif not close(a,b):
                        raise RuntimeError(f"{name}: NPZ={a} CSV={b}")

                if split != r["Split"].strip().lower():
                    raise RuntimeError(f"split: NPZ={split} CSV={r['Split']}")

                d = float(r["D_outer"])
                t = float(r["Thickness"])
                rb = float(r["R_bending"])
                if abs(t/d - float(r["t_over_D"])) > 6e-5:
                    raise RuntimeError("CSV t_over_D inconsistent with Thickness/D_outer")
                if abs(rb/d - float(r["R_over_D"])) > 6e-5:
                    raise RuntimeError("CSV R_over_D inconsistent with R_bending/D_outer")

                if "U_tube" in z.files and "tube_global_index" in z.files and "tube_mesh_pos" in z.files:
                    idx = np.asarray(z["tube_global_index"], dtype=np.int64)
                    tube_mesh = np.asarray(z["tube_mesh_pos"], dtype=np.float64)
                    u = np.asarray(z["U_tube"], dtype=np.float64)
                    recon = world[:, idx, :].astype(np.float64) - tube_mesh[None,:,:]
                    err = float(np.max(np.abs(recon-u)))
                    if err > 2e-4:
                        raise RuntimeError(f"U_tube reconstruction maxdiff={err}")

                rows.append({
                    "sample_id": sid,
                    "split": split,
                    "nodes": int(mesh.shape[0]),
                    "tube_nodes": int(tube.sum()),
                    "source": path.name,
                })
                print(f"[{i:03d}/{len(files):03d}] PASS sample={sid:04d} split={split} nodes={mesh.shape[0]}", flush=True)
        except Exception as e:
            errors.append(f"{path.name}: {e}")
            print(f"[{i:03d}/{len(files):03d}] FAIL {path.name}: {e}", flush=True)

    missing_ids = sorted(set(csv_rows) - seen)
    extra_ids = sorted(seen - set(csv_rows))
    if missing_ids:
        errors.append(f"Missing CSV Sample_IDs in NPZ: {missing_ids}")
    if extra_ids:
        errors.append(f"Unexpected NPZ Sample_IDs: {extra_ids}")

    expected_split = Counter({"train":120,"val":15,"test":15})
    if splits != expected_split:
        errors.append(f"split counts {dict(splits)} != {dict(expected_split)}")

    report = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "csv": str(args.csv.resolve()),
        "num_npz": len(files),
        "num_csv": len(csv_rows),
        "split_counts": dict(splits),
        "errors": errors,
        "warnings": warnings,
        "status": "PASSED" if not errors else "FAILED",
    }
    out = args.output_json or (args.dataset_dir / "CLEAN150_NPZ_CHECK_REPORT.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("="*100)
    print("CLEAN150 NPZ CHECK:", report["status"])
    print("split counts:", dict(splits))
    print("errors:", len(errors))
    if errors:
        for e in errors[:30]:
            print(" -", e)
    print("report:", out)
    print("="*100)

    if errors and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
