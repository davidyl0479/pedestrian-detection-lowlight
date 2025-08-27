#!/usr/bin/env python3
"""
Explore image-characteristics distributions for train/val JSONs.

Adds Pearson/Spearman correlation heatmaps (per split).

Inputs
------
- --train_json results/images/train.json
- --val_json   results/images/val.json

Outputs (in --out_dir, default: eda_chars/)
-------------------------------------------
- stats.csv
- ks_tests.csv
- corr_pearson_{split}.csv, corr_spearman_{split}.csv
- quantile_edges_q{4,5}_{split}.csv
- plots/hist_{feature}.png, plots/ecdf_{feature}.png
- plots/corr_pearson_{split}.png, plots/corr_spearman_{split}.png

Notes
-----
- Uses matplotlib only (no seaborn). One chart per figure; no custom colors.
"""

import json, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_chars(json_path: Path, split_name: str) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for fname, payload in data.items():
        img = payload.get("image_info", {})
        iid = img.get("id")
        feats = payload.get("characteristics", {})
        if iid is None or not isinstance(feats, dict):
            continue
        row = {"split": split_name, "image_id": int(iid), "file_name": fname}
        for k, v in feats.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def ks_2sample(x: np.ndarray, y: np.ndarray):
    x = np.sort(x.astype(float))
    y = np.sort(y.astype(float))
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, np.nan
    i = j = 0
    D = 0.0
    vals = np.r_[x, y]
    vals.sort()
    for v in vals:
        while i < nx and x[i] <= v:
            i += 1
        while j < ny and y[j] <= v:
            j += 1
        cdf_x = i / nx
        cdf_y = j / ny
        D = max(D, abs(cdf_x - cdf_y))
    en = math.sqrt(nx * ny / (nx + ny))
    if en == 0:
        return D, np.nan
    lam = (en + 0.12 + 0.11 / en) * D
    p = 0.0
    for k in range(1, 101):
        p += (-1) ** (k - 1) * math.exp(-2 * (k**2) * (lam**2))
    p = max(min(2 * p, 1.0), 0.0)
    return D, p


def safe_numeric_cols(df: pd.DataFrame, blacklist=("image_id",)) -> list:
    cols = []
    for c in df.columns:
        if c in ("split", "file_name") or c in blacklist:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def describe_split(df: pd.DataFrame, split: str, features: list) -> pd.DataFrame:
    sub = df[df["split"] == split]
    rows = []
    for f in features:
        s = pd.to_numeric(sub[f], errors="coerce")
        n = s.notna().sum()
        miss = s.isna().sum()
        if n > 0:
            q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append(
                {
                    "split": split,
                    "feature": f,
                    "count": int(n),
                    "missing": int(miss),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)),
                    "min": float(s.min()),
                    "p05": float(q.loc[0.05]),
                    "p25": float(q.loc[0.25]),
                    "p50": float(q.loc[0.5]),
                    "p75": float(q.loc[0.75]),
                    "p95": float(q.loc[0.95]),
                    "max": float(s.max()),
                    "iqr": float(q.loc[0.75] - q.loc[0.25]),
                    "skew": float(s.skew()),
                    "kurtosis": float(s.kurtosis()),
                }
            )
        else:
            rows.append(
                {
                    "split": split,
                    "feature": f,
                    "count": 0,
                    "missing": int(miss),
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "skew": np.nan,
                    "kurtosis": np.nan,
                }
            )
    return pd.DataFrame(rows)


def suggest_quantile_edges(df: pd.DataFrame, features: list, q: int) -> pd.DataFrame:
    rows = []
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce").dropna().values
        if s.size == 0:
            rows.append({"feature": f, "edges": ""})
            continue
        qs = np.linspace(0, 1, q + 1)
        edges = np.unique(np.quantile(s, qs))
        edges = edges.astype(float).tolist()
        if len(edges) == 1:
            edges = [edges[0], edges[0] + 1e-9]
        elif len(edges) > 1:
            edges[-1] = edges[-1] + 1e-9
        rows.append({"feature": f, "edges": " ".join(f"{v:.6g}" for v in edges)})
    return pd.DataFrame(rows)


def corr_mats(df: pd.DataFrame, split: str, features: list):
    sub = df[df["split"] == split][features]
    return sub.corr(method="pearson"), sub.corr(method="spearman")


def plot_hist_ecdf(
    df: pd.DataFrame, features: list, out_dir: Path, splits: list, bins=40
):
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    for f in features:
        # Histogram
        fig = plt.figure()
        for split in splits:
            s = (
                pd.to_numeric(df[df["split"] == split][f], errors="coerce")
                .dropna()
                .values
            )
            if s.size == 0:
                continue
            plt.hist(s, bins=bins, alpha=0.5, label=split, density=True)
        if len(splits) > 1:
            plt.legend()
        plt.title(f"Histogram: {f}")
        plt.xlabel(f)
        plt.ylabel("Density")
        fig.savefig(pdir / f"hist_{f}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ECDF
        fig2 = plt.figure()
        for split in splits:
            s = (
                pd.to_numeric(df[df["split"] == split][f], errors="coerce")
                .dropna()
                .values
            )
            if s.size == 0:
                continue
            s.sort()
            y = np.arange(1, s.size + 1) / s.size
            plt.step(s, y, where="post", label=split)
        if len(splits) > 1:
            plt.legend()
        plt.title(f"ECDF: {f}")
        plt.xlabel(f)
        plt.ylabel("F(x)")
        fig2.savefig(pdir / f"ecdf_{f}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)


def plot_corr_heatmap(corr_df: pd.DataFrame, title: str, out_path: Path):
    # corr_df: symmetric DataFrame, index/columns are feature names
    fig = plt.figure(
        figsize=(
            max(6, 0.5 * len(corr_df.columns) + 2),
            max(5, 0.5 * len(corr_df.index) + 2),
        )
    )
    ax = plt.gca()
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def read_index_set(path: Path) -> set:
    """
    Reads one filename per line (ignores blanks/# comments).
    Returns a set of BASE NAMES (e.g., '58c57ff6....png').
    """
    if path is None:
        return set()
    names = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        names.add(Path(ln).name)
    return names


def main():
    ap = argparse.ArgumentParser(
        description="Explore distributions of image characteristics for chosen splits."
    )
    ap.add_argument(
        "--train_json",
        required=True,
        type=Path,
        help="results/images/train.json (original train)",
    )
    ap.add_argument(
        "--val_json",
        required=True,
        type=Path,
        help="results/images/val.json (original val)",
    )
    # NEW: index files to select the exact subsets you used
    ap.add_argument(
        "--train_index", type=Path, help="index/train_mixed.txt (filenames)"
    )
    ap.add_argument("--val_index", type=Path, help="index/val_internal.txt (filenames)")
    ap.add_argument(
        "--test_index",
        type=Path,
        help="index/test_all.txt (filenames). If omitted, uses all of val.json",
    )

    # NEW: control which features to analyse and where to write outputs
    ap.add_argument(
        "--features",
        nargs="+",
        help="Subset of feature names to analyse (default: all numeric)",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path("results"),
        help="Base output dir (default: results)",
    )
    ap.add_argument(
        "--run_name", type=str, help="Optional custom folder name under out_root"
    )

    args = ap.parse_args()

    # 1) Load original splits
    df_train_all = load_chars(args.train_json, "orig_train")
    df_val_all = load_chars(args.val_json, "orig_val")

    # 2) Read index files (filenames only)
    idx_train = read_index_set(args.train_index) if args.train_index else set()
    idx_vali = read_index_set(args.val_index) if args.val_index else set()
    idx_test = read_index_set(args.test_index) if args.test_index else set()

    # 3) Build the three analysis splits
    frames, split_names = [], []

    if idx_train:
        df_tm = df_train_all[df_train_all["file_name"].isin(idx_train)].copy()
        df_tm["split"] = "train_mixed"
        frames.append(df_tm)
        split_names.append("train_mixed")

    if idx_vali:
        df_vi = df_train_all[df_train_all["file_name"].isin(idx_vali)].copy()
        df_vi["split"] = "val_internal"
        frames.append(df_vi)
        split_names.append("val_internal")

    # test_all comes from original val.json (use index if provided; else take all val)
    if idx_test:
        df_te = df_val_all[df_val_all["file_name"].isin(idx_test)].copy()
    else:
        df_te = df_val_all.copy()
    df_te["split"] = "test_all"
    frames.append(df_te)
    split_names.append("test_all")

    df = pd.concat(frames, ignore_index=True, sort=False)

    # 4) Feature selection
    all_feats = [
        c
        for c in df.columns
        if c not in ("split", "file_name", "image_id")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    feats = args.features if args.features else all_feats

    # 5) Output directory under results/, auto-named by features unless run_name is given
    if args.run_name:
        out_dir = args.out_root / f"eda_chars_{args.run_name}"
    else:
        tag = "all_features" if not args.features else "_".join(feats)
        out_dir = args.out_root / f"eda_chars_[{tag}]"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) Descriptive stats (per split)
    stats_rows = []
    for sp in split_names:
        stats_rows.append(describe_split(df, sp, feats))
    stats_all = pd.concat(stats_rows, ignore_index=True)
    stats_all.to_csv(out_dir / "stats.csv", index=False)

    # KS tests (all pairwise combinations among the chosen splits)
    import itertools

    ks_rows = []
    for f in feats:
        for a, b in itertools.combinations(split_names, 2):
            xa = pd.to_numeric(df[df["split"] == a][f], errors="coerce").dropna().values
            xb = pd.to_numeric(df[df["split"] == b][f], errors="coerce").dropna().values
            D, p = ks_2sample(xa, xb)
            ks_rows.append(
                {
                    "feature": f,
                    "split_a": a,
                    "split_b": b,
                    "ks_D": D,
                    "p_approx": p,
                    "n_a": int(xa.size),
                    "n_b": int(xb.size),
                }
            )
    pd.DataFrame(ks_rows).to_csv(out_dir / "ks_tests.csv", index=False)

    # Correlations + heatmaps, one file per split
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    for sp in split_names:
        pear, spear = corr_mats(df, sp, feats)
        pear.to_csv(out_dir / f"corr_pearson_{sp}.csv")
        spear.to_csv(out_dir / f"corr_spearman_{sp}.csv")
        if not pear.empty:
            plot_corr_heatmap(
                pear, f"Pearson correlation ({sp})", pdir / f"corr_pearson_{sp}.png"
            )
            plot_corr_heatmap(
                spear, f"Spearman correlation ({sp})", pdir / f"corr_spearman_{sp}.png"
            )

    # Quantile edge suggestions (per split): Q4 and Q5
    for q in (4, 5):
        for sp in split_names:
            sub = df[df["split"] == sp].copy()
            qe = suggest_quantile_edges(sub, feats, q)
            qe.to_csv(out_dir / f"quantile_edges_q{q}_{sp}.csv", index=False)

    # Plots (hist + ECDF) over the chosen splits
    plot_hist_ecdf(df, feats, out_dir, split_names)

    print(f"[OK] Wrote outputs to: {out_dir.resolve()}")
    print(f"Features analysed ({len(feats)}): {', '.join(feats)}")


if __name__ == "__main__":
    main()
