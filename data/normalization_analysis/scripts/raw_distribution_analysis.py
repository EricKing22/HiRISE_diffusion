import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path("/scratch_root/as5023/HiRISE/data")
CSV_PATH = DATA_ROOT / "data_record_bin12.csv"
OUT_DIR = Path("src/output/normalization_analysis/raw")
HIST_BINS = 512
PIXEL_PAIR_STRIDE = 257
MAX_PIXEL_PAIRS = 500_000


def resolve_data_path(data_root: Path, rel_path: str) -> Path:
    rel_path = str(rel_path)
    path = data_root / rel_path
    if path.is_file():
        return path

    for marker in ("npy_files_b12", "npy_files_b22", "npy_files_evals"):
        if marker in rel_path:
            suffix = rel_path[rel_path.index(marker):]
            candidate = data_root / suffix
            if candidate.is_file():
                return candidate

    return path


def image_stats(arr: np.ndarray) -> dict:
    arr64 = arr.astype(np.float64, copy=False)
    return {
        "min": float(np.min(arr64)),
        "max": float(np.max(arr64)),
        "mean": float(np.mean(arr64)),
        "std": float(np.std(arr64, ddof=1)),
        "p01": float(np.percentile(arr64, 1)),
        "p05": float(np.percentile(arr64, 5)),
        "p50": float(np.percentile(arr64, 50)),
        "p95": float(np.percentile(arr64, 95)),
        "p99": float(np.percentile(arr64, 99)),
    }


def describe(series: pd.Series) -> dict:
    q = series.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(q.loc[0.0]),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p50": float(q.loc[0.5]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(q.loc[1.0]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_record = pd.read_csv(CSV_PATH)

    required = {"IR10", "RED4"}
    complete = data_record.groupby("Set")["CCD"].apply(lambda c: required.issubset(set(c)))
    paired_sets = np.array(sorted(complete[complete].index))

    metadata = {
        "data_root": str(DATA_ROOT),
        "csv_path": str(CSV_PATH),
        "csv_rows": int(len(data_record)),
        "observations": int(data_record["Observation"].nunique()),
        "sets_total": int(data_record["Set"].nunique()),
        "paired_ir10_red4_sets": int(len(paired_sets)),
        "ccd_counts": {k: int(v) for k, v in data_record["CCD"].value_counts().sort_index().items()},
        "hist_bins": HIST_BINS,
        "pixel_pair_stride": PIXEL_PAIR_STRIDE,
    }

    # CSV Pix_min/Pix_max provide a cheap full-table range estimate for bin edges.
    ir_rows = data_record[data_record["CCD"] == "IR10"]
    red_rows = data_record[data_record["CCD"] == "RED4"]
    global_min = float(min(ir_rows["Pix_min"].min(), red_rows["Pix_min"].min()))
    global_max = float(max(ir_rows["Pix_max"].max(), red_rows["Pix_max"].max()))
    hist_edges = np.linspace(global_min, global_max, HIST_BINS + 1)
    ir_hist = np.zeros(HIST_BINS, dtype=np.int64)
    red_hist = np.zeros(HIST_BINS, dtype=np.int64)

    rows = []
    pair_ir = []
    pair_red = []
    examples = []
    missing = []

    for i, set_id in enumerate(paired_sets):
        frame = data_record[data_record["Set"] == set_id].set_index("CCD")
        ir_path = resolve_data_path(DATA_ROOT, frame.at["IR10", "Path"])
        red_path = resolve_data_path(DATA_ROOT, frame.at["RED4", "Path"])
        if not ir_path.is_file() or not red_path.is_file():
            missing.append({"set": int(set_id), "ir_path": str(ir_path), "red_path": str(red_path)})
            continue

        ir = np.load(ir_path).astype(np.float32, copy=False)
        red = np.load(red_path).astype(np.float32, copy=False)

        if i < 4:
            examples.append(
                {
                    "set": int(set_id),
                    "observation": str(frame.at["IR10", "Observation"]),
                    "ir": ir.copy(),
                    "red": red.copy(),
                }
            )

        ir_hist += np.histogram(ir, bins=hist_edges)[0]
        red_hist += np.histogram(red, bins=hist_edges)[0]

        if len(pair_ir) < MAX_PIXEL_PAIRS:
            ir_flat = ir.reshape(-1)[::PIXEL_PAIR_STRIDE]
            red_flat = red.reshape(-1)[::PIXEL_PAIR_STRIDE]
            remaining = MAX_PIXEL_PAIRS - len(pair_ir)
            pair_ir.extend(ir_flat[:remaining].tolist())
            pair_red.extend(red_flat[:remaining].tolist())

        ir_s = image_stats(ir)
        red_s = image_stats(red)
        rows.append(
            {
                "set": int(set_id),
                "observation": str(frame.at["IR10", "Observation"]),
                "date": str(frame.at["IR10", "Date"]),
                "ir_min": ir_s["min"],
                "ir_max": ir_s["max"],
                "ir_mean": ir_s["mean"],
                "ir_std": ir_s["std"],
                "ir_p05": ir_s["p05"],
                "ir_p50": ir_s["p50"],
                "ir_p95": ir_s["p95"],
                "red_min": red_s["min"],
                "red_max": red_s["max"],
                "red_mean": red_s["mean"],
                "red_std": red_s["std"],
                "red_p05": red_s["p05"],
                "red_p50": red_s["p50"],
                "red_p95": red_s["p95"],
                "mean_red_minus_ir": red_s["mean"] - ir_s["mean"],
                "std_red_over_ir": red_s["std"] / (ir_s["std"] + 1e-12),
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"processed {i + 1}/{len(paired_sets)} paired sets", flush=True)

    per_set = pd.DataFrame(rows)
    per_set.to_csv(OUT_DIR / "raw_per_set_stats.csv", index=False)

    hist = pd.DataFrame(
        {
            "bin_left": hist_edges[:-1],
            "bin_right": hist_edges[1:],
            "bin_center": 0.5 * (hist_edges[:-1] + hist_edges[1:]),
            "ir_count": ir_hist,
            "red_count": red_hist,
        }
    )
    hist.to_csv(OUT_DIR / "raw_pixel_histogram.csv", index=False)

    summary = {
        "metadata": metadata,
        "missing_pairs": missing,
        "loaded_pairs": int(len(per_set)),
        "raw_distribution_summary": {
            "ir_mean_per_set": describe(per_set["ir_mean"]),
            "red_mean_per_set": describe(per_set["red_mean"]),
            "ir_std_per_set": describe(per_set["ir_std"]),
            "red_std_per_set": describe(per_set["red_std"]),
            "mean_red_minus_ir_per_set": describe(per_set["mean_red_minus_ir"]),
            "std_red_over_ir_per_set": describe(per_set["std_red_over_ir"]),
        },
        "pixel_histogram_totals": {
            "ir_pixels": int(ir_hist.sum()),
            "red_pixels": int(red_hist.sum()),
            "bin_min": global_min,
            "bin_max": global_max,
        },
    }
    with open(OUT_DIR / "raw_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Figure 1: raw pixel histogram from all loaded paired sets.
    centers = hist["bin_center"].to_numpy()
    bin_width = hist_edges[1] - hist_edges[0]
    ir_density = ir_hist / (ir_hist.sum() * bin_width)
    red_density = red_hist / (red_hist.sum() * bin_width)
    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(centers, ir_density, label="IR10", color="#1f77b4", linewidth=1.8)
    plt.plot(centers, red_density, label="RED4", color="#d62728", linewidth=1.8)
    plt.xlabel("Raw reflectance")
    plt.ylabel("Pixel density")
    plt.title("Raw pixel distribution before normalization")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "raw_pixel_distribution.png")
    plt.close()

    # Figure 2: per-set mean/std distributions and band offset.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=180)
    axes[0, 0].hist(per_set["ir_mean"], bins=80, alpha=0.65, label="IR10", color="#1f77b4")
    axes[0, 0].hist(per_set["red_mean"], bins=80, alpha=0.65, label="RED4", color="#d62728")
    axes[0, 0].set_title("Per-set raw mean")
    axes[0, 0].set_xlabel("Mean reflectance")
    axes[0, 0].legend()

    axes[0, 1].hist(per_set["ir_std"], bins=80, alpha=0.65, label="IR10", color="#1f77b4")
    axes[0, 1].hist(per_set["red_std"], bins=80, alpha=0.65, label="RED4", color="#d62728")
    axes[0, 1].set_title("Per-set raw standard deviation")
    axes[0, 1].set_xlabel("Std reflectance")
    axes[0, 1].legend()

    axes[1, 0].hist(per_set["mean_red_minus_ir"], bins=80, color="#555555")
    axes[1, 0].axvline(0.0, color="black", linewidth=1.0)
    axes[1, 0].set_title("RED4 mean - IR10 mean")
    axes[1, 0].set_xlabel("Reflectance offset")

    axes[1, 1].scatter(per_set["ir_mean"], per_set["red_mean"], s=2, alpha=0.25, color="#2ca02c")
    low = min(per_set["ir_mean"].min(), per_set["red_mean"].min())
    high = max(per_set["ir_mean"].max(), per_set["red_mean"].max())
    axes[1, 1].plot([low, high], [low, high], color="black", linewidth=1.0)
    axes[1, 1].set_title("Per-set mean relationship")
    axes[1, 1].set_xlabel("IR10 mean")
    axes[1, 1].set_ylabel("RED4 mean")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "raw_per_set_distributions.png")
    plt.close(fig)

    # Figure 3: pixel-pair relationship from a deterministic subsample.
    plt.figure(figsize=(6, 5), dpi=180)
    plt.hexbin(pair_ir, pair_red, gridsize=90, bins="log", cmap="viridis")
    low = min(min(pair_ir), min(pair_red))
    high = max(max(pair_ir), max(pair_red))
    plt.plot([low, high], [low, high], color="white", linewidth=1.0, alpha=0.8)
    plt.xlabel("IR10 raw reflectance")
    plt.ylabel("RED4 raw reflectance")
    plt.title("Raw pixel-pair relationship")
    cb = plt.colorbar()
    cb.set_label("log10(count)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "raw_pixel_pair_hexbin.png")
    plt.close()

    # Figure 4: example paired patches.
    if examples:
        fig, axes = plt.subplots(len(examples), 2, figsize=(6, 2.7 * len(examples)), dpi=180)
        for row, example in enumerate(examples):
            vmin = min(float(example["ir"].min()), float(example["red"].min()))
            vmax = max(float(example["ir"].max()), float(example["red"].max()))
            axes[row, 0].imshow(example["ir"], cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f"IR10 set {example['set']}")
            axes[row, 1].imshow(example["red"], cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f"RED4 {example['observation']}")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT_DIR / "raw_example_patches.png")
        plt.close(fig)

    print(f"wrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
