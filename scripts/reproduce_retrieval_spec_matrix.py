#!/usr/bin/env python3
"""Camera-ready deterministic derivation of Table 1 from frozen Kc data."""
import argparse
import csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("artifacts/camera_ready/noise_ceilings_reproduced.csv"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/camera_ready/retrieval_spec_matrix.csv"))
    args = ap.parse_args()
    rows = list(csv.DictReader(args.input.open()))
    grouped = {}
    for r in rows: grouped.setdefault((r["dataset"], r["reader"]), {})[r["condition"]] = r["Kc"]
    out = [{"dataset": d, "reader": m, "Kc_hard": v["hard"], "Kc_soft": v["soft"]} for (d,m),v in sorted(grouped.items())]
    if len(out) != 12: raise ValueError(f"Expected 12 reader rows / 24 cells; got {len(out)}")
    with args.output.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print("Verified and wrote all 24 camera-ready Hard/Soft matrix cells.")
if __name__ == "__main__": main()
