"""Run Sub_PCA on TSB-AD-U series and compare TSB-AD's VUS-PR with fast_vus.vus_pr."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from TSB_AD.evaluation.basic_metrics import generate_curve
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.utils.slidingWindows import find_length_rank

from fast_vus import vus_pr


def timed(f, *a):
    t = time.perf_counter()
    return f(*a), time.perf_counter() - t


def tsb_vus_pr(label, score, window):
    return float(generate_curve(label, score, window, "opt", 250)[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, help="TSB-AD-U directory")
    p.add_argument("--files", type=Path, help="file list csv (column file_name); default: all csv in data")
    p.add_argument("-n", "--limit", type=int)
    p.add_argument("-o", "--out", type=Path, default=Path("results.csv"))
    a = p.parse_args()

    files = pd.read_csv(a.files).file_name.tolist() if a.files else sorted(f.name for f in a.data.glob("*.csv"))
    rows = []
    for f in files[: a.limit]:
        # load and score as tsb-ad's Run_Detector_U.py does
        df = pd.read_csv(a.data / f).dropna()
        x = df.iloc[:, :-1].to_numpy(float)
        y = df.Label.to_numpy(int)
        w = int(find_length_rank(x[:, :1], rank=1))
        score = run_Unsupervise_AD("Sub_PCA", x, periodicity=1, n_components=None)

        # same score, both implementations
        fast, t_fast = timed(vus_pr, y, score, w)
        ref, t_ref = timed(tsb_vus_pr, y, score, w)
        rows.append(dict(file=f, n=len(y), window=w, tsb_ad=ref, fast=fast, diff=abs(ref - fast),
                         t_tsb_ad=t_ref, t_fast=t_fast, speedup=t_ref / t_fast))
        print(f"{f[:40]:40s} n={len(y):7d} w={w:3d}  vus_pr={ref:.4f}  diff={abs(ref - fast):.1e}  "
              f"{t_ref:7.2f}s -> {t_fast:6.3f}s  x{t_ref / t_fast:.0f}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    print(f"\n{len(r)} series  max diff {r['diff'].max():.1e}  "
          f"total {r.t_tsb_ad.sum():.1f}s -> {r.t_fast.sum():.1f}s  median speedup x{r.speedup.median():.0f}")


if __name__ == "__main__":
    main()
