#!/usr/bin/env python3
"""Camera-ready reproduction of NRR and condition-specific Noise Ceilings."""
import argparse
import csv
from pathlib import Path

GRID = (1, 5, 10, 20, 30, 40, 50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("artifacts/camera_ready/score_nrr_frozen.csv"))
    ap.add_argument("--nrr-output", type=Path, default=Path("artifacts/camera_ready/nrr_reproduced.csv"))
    ap.add_argument("--kc-output", type=Path, default=Path("artifacts/camera_ready/noise_ceilings_reproduced.csv"))
    args = ap.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8-sig")))
    out, ceilings = [], []
    for dataset in sorted({r["Dataset"] for r in rows}):
        for reader in sorted({r["Model"] for r in rows if r["Dataset"] == dataset}):
            group = sorted((r for r in rows if r["Dataset"] == dataset and r["Model"] == reader), key=lambda r: int(r["K"]))
            if tuple(int(r["K"]) for r in group) != GRID:
                raise ValueError(f"Incomplete K grid: {dataset}/{reader}")
            for condition, score_col, frozen_col in (
                ("soft", "Score (Soft mean)", "NRR(soft)"),
                ("hard", "Score (Hard mean)", "NRR(hard)"),
                ("bge_m3", "Score (BGE-m3)", "NRR (BGE-m3)"),
            ):
                base = float(group[0][score_col])
                passing = []
                for r in group:
                    nrr = float(r[score_col]) / base
                    frozen = float(r[frozen_col])
                    if abs(nrr - frozen) > 0.0006:
                        raise ValueError(f"Frozen NRR mismatch: {dataset}/{reader}/{condition}/K={r['K']}")
                    out.append({"dataset": dataset, "reader": reader, "K": r["K"], "condition": condition, "score": r[score_col], "nrr_calculated": f"{nrr:.8f}", "nrr_frozen": r[frozen_col]})
                    if condition != "bge_m3" and nrr >= 0.9:
                        passing.append(int(r["K"]))
                if condition != "bge_m3":
                    kc = max(passing)
                    ceilings.append({"dataset": dataset, "reader": reader, "condition": condition, "Kc": ">=50" if kc == 50 else str(kc)})
    args.nrr_output.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((args.nrr_output, out), (args.kc_output, ceilings)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    print(f"Verified {len(out)} NRR cells and wrote {len(ceilings)} Noise Ceilings.")

if __name__ == "__main__":
    main()
