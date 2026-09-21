from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

SUBSET_SIZES = (40, 60, 80, 120)
DESIGN_RANGES = {
    "D_outer": (6.0, 20.0),
    "t_over_D": (0.04, 0.14),
    "R_over_D": (1.25, 4.50),
}


def parse_sid_from_name(name: str) -> int:
    m = re.search(r"sample[_-]?0*(\d+)", os.path.basename(name), flags=re.I)
    if not m:
        raise ValueError(f"Cannot parse Sample_ID from manifest entry: {name}")
    return int(m.group(1))


def read_manifest(cache_dir: Path, split: str) -> List[Tuple[int, Path]]:
    p = cache_dir / f"{split}_manifest.txt"
    if not p.is_file():
        raise FileNotFoundError(p)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        src = cache_dir / line
        if not src.is_file():
            raise FileNotFoundError(src)
        out.append((parse_sid_from_name(line), src.resolve()))
    return out


def norm_design(v: float, lo: float, hi: float) -> float:
    return 2.0 * (float(v) - lo) / (hi - lo) - 1.0


def load_design_csv(path: Path) -> Dict[int, Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    out: Dict[int, Dict] = {}
    for r in rows:
        sid = int(r["Sample_ID"])
        d = float(r["D_outer"])
        t = float(r["Thickness"])
        rb = float(r["R_bending"])
        td = float(r["t_over_D"]) if r.get("t_over_D", "").strip() else t / d
        rd = float(r["R_over_D"]) if r.get("R_over_D", "").strip() else rb / d
        out[sid] = {
            "Sample_ID": sid,
            "D_outer": d,
            "Thickness": t,
            "R_bending": rb,
            "t_over_D": td,
            "R_over_D": rd,
            "Region": str(r.get("Region", "")),
            "Split": str(r.get("Split", "")).strip().lower(),
            "x": [
                norm_design(d, *DESIGN_RANGES["D_outer"]),
                norm_design(td, *DESIGN_RANGES["t_over_D"]),
                norm_design(rd, *DESIGN_RANGES["R_over_D"]),
            ],
        }
    return out


def sqdist(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def farthest_traversal(rows: List[Dict], first_idx: int) -> List[int]:
    n = len(rows)
    selected = [first_idx]
    selected_set = {first_idx}
    min_d2 = [sqdist(rows[i]["x"], rows[first_idx]["x"]) for i in range(n)]
    min_d2[first_idx] = -1.0
    while len(selected) < n:
        best = max(
            (i for i in range(n) if i not in selected_set),
            key=lambda i: (min_d2[i], -rows[i]["Sample_ID"]),
        )
        selected.append(best)
        selected_set.add(best)
        for i in range(n):
            if i in selected_set:
                continue
            d2 = sqdist(rows[i]["x"], rows[best]["x"])
            if d2 < min_d2[i]:
                min_d2[i] = d2
        min_d2[best] = -1.0
    return selected


def covering_radius(rows: List[Dict], order: List[int], k: int) -> float:
    sel = order[:k]
    worst = 0.0
    for r in rows:
        d2 = min(sqdist(r["x"], rows[j]["x"]) for j in sel)
        worst = max(worst, math.sqrt(d2))
    return worst


def choose_best_order(rows: List[Dict]) -> Tuple[List[int], Dict]:
    if len(rows) != 120:
        raise RuntimeError(f"Expected Train120, got {len(rows)}")
    best = None
    best_meta = None
    for first in range(len(rows)):
        order = farthest_traversal(rows, first)
        radii = {k: covering_radius(rows, order, k) for k in (40, 60, 80)}
        score = radii[40] + radii[60] + radii[80]
        key = (score, radii[40], radii[60], radii[80], rows[first]["Sample_ID"])
        if best is None or key < best[0]:
            best = (key, order)
            best_meta = {
                "first_sample_id": rows[first]["Sample_ID"],
                "coverage_radius": radii,
                "score": score,
            }
    assert best is not None
    return best[1], best_meta


def safe_replace_symlink(link: Path, target: Path):
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def write_manifest(view: Path, split: str, ids: List[int], source_map: Dict[int, Path]):
    names = []
    for sid in ids:
        src = source_map[sid]
        name = f"sample_{sid:04d}.pt"
        dst = view / name
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            raise RuntimeError(f"Refusing to replace non-symlink: {dst}")
        dst.symlink_to(src)
        names.append(name)
    (view / f"{split}_manifest.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", type=Path, required=True)
    ap.add_argument("--design-csv", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, default=Path("dataeff_workspace"))
    ap.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    base_cache = args.base_cache.resolve()
    design_csv = args.design_csv.resolve()
    ws = args.workspace.resolve()

    manifests = {s: read_manifest(base_cache, s) for s in ("train", "val", "test")}
    counts = {k: len(v) for k, v in manifests.items()}
    if counts != {"train": 120, "val": 15, "test": 15}:
        raise RuntimeError(f"Expected 120/15/15 base split, got {counts}")

    design = load_design_csv(design_csv)
    # The clean CSV shows t/D down to about 0.0397; V2 therefore uses the
    # intended fixed design range 0.04..0.14 instead of the old 0.05 lower bound.
    bad_td = [
        sid for sid,row in design.items()
        if not (0.039 <= float(row["t_over_D"]) <= 0.141)
    ]
    if bad_td:
        raise RuntimeError(f"Clean CSV has t/D outside expected ~0.04..0.14: {bad_td}")
    manifest_ids = {s: [sid for sid, _ in manifests[s]] for s in manifests}
    all_manifest_ids = set(sum(manifest_ids.values(), []))
    if len(all_manifest_ids) != 150:
        raise RuntimeError("Base manifests do not contain 150 unique Sample_IDs")
    missing_csv = sorted(all_manifest_ids - set(design))
    if missing_csv:
        raise RuntimeError(f"CSV missing Sample_IDs: {missing_csv}")

    for split, ids in manifest_ids.items():
        bad = [sid for sid in ids if design[sid]["Split"] and design[sid]["Split"] != split]
        if bad:
            raise RuntimeError(f"CSV split mismatch for {split}: {bad}")

    train_rows = [design[sid] for sid in manifest_ids["train"]]
    order_idx, order_meta = choose_best_order(train_rows)
    ordered_train_ids = [train_rows[i]["Sample_ID"] for i in order_idx]

    source_map = {}
    for split in manifests:
        for sid, src in manifests[split]:
            if sid in source_map:
                raise RuntimeError(f"Duplicate Sample_ID in manifests: {sid}")
            source_map[sid] = src

    if ws.exists() and args.overwrite:
        # Only remove generated workspace; caller explicitly opted in.
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "subsets").mkdir(exist_ok=True)
    (ws / "cache_views").mkdir(exist_ok=True)
    (ws / "residual_stats").mkdir(exist_ok=True)
    (ws / "stats").mkdir(exist_ok=True)

    (ws / "subsets" / "train_order_maximin.json").write_text(
        json.dumps(
            {
                "method": "multi-start greedy farthest-point traversal",
                "design_coordinates": ["D_outer", "t_over_D", "R_over_D"],
                "design_ranges": DESIGN_RANGES,
                "ordered_train_sample_ids": ordered_train_ids,
                "selection_metadata": order_meta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "base_cache": str(base_cache),
        "design_csv": str(design_csv),
        "base_split_counts": counts,
        "subsets": {},
        "selection_metadata": order_meta,
    }

    for n in SUBSET_SIZES:
        tag = f"N{n:03d}"
        train_ids = ordered_train_ids[:n]
        (ws / "subsets" / f"train{n}_ids.txt").write_text(
            "\n".join(str(x) for x in train_ids) + "\n", encoding="utf-8"
        )
        view = ws / "cache_views" / tag
        view.mkdir(parents=True, exist_ok=True)
        write_manifest(view, "train", train_ids, source_map)
        write_manifest(view, "val", manifest_ids["val"], source_map)
        write_manifest(view, "test", manifest_ids["test"], source_map)

        region_counts = {}
        for sid in train_ids:
            r = design[sid]["Region"] or "<empty>"
            region_counts[r] = region_counts.get(r, 0) + 1
        summary["subsets"][tag] = {
            "train_ids": train_ids,
            "train_count": n,
            "val_ids": manifest_ids["val"],
            "test_ids": manifest_ids["test"],
            "region_counts": region_counts,
            "coverage_radius": 0.0 if n == 120 else covering_radius(train_rows, order_idx, n),
        }

    shutil.copy2(design_csv, ws / "design_csv_snapshot.csv")
    (ws / "workspace_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Hard nestedness gates.
    sets = {n: set(ordered_train_ids[:n]) for n in SUBSET_SIZES}
    assert sets[40] < sets[60] < sets[80] < sets[120]
    assert set(manifest_ids["val"]).isdisjoint(sets[120])
    assert set(manifest_ids["test"]).isdisjoint(sets[120])

    print("=" * 100)
    print("ANARESID X10 DATA-EFFICIENCY WORKSPACE PREPARED")
    print("workspace:", ws)
    print("best first sample:", order_meta["first_sample_id"])
    print("coverage radii:", order_meta["coverage_radius"])
    for n in SUBSET_SIZES:
        tag = f"N{n:03d}"
        print(tag, "train/val/test =", n, 15, 15, "coverage=", summary["subsets"][tag]["coverage_radius"])
        print("  ids:", summary["subsets"][tag]["train_ids"])
    print("WORKSPACE PREPARATION PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()
