"""
Compute SCI prior statistics (mu, sigma, histogram) from the training dataset
and save prior_ir.pt / prior_red.pt to --prior_dir.

Run once after training; the saved files are loaded by inference.py.

Usage:
    cd <project_root>
    python src/compute_prior.py \
        --data_root /path/to/data  \
        --csv_path  /path/to/data_record_bin12.csv \
        --prior_dir src/models/saves
"""

import os
import sys
import argparse

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import DiffusionDataset


# =============================================================================
# Histogram and prior statistics
# =============================================================================

def soft_histogram(
    x:       torch.Tensor,
    bins:    int,
    min_val: float = -1.0,
    max_val: float =  1.0,
) -> torch.Tensor:
    """
    Differentiable soft histogram via Gaussian bin assignment.
    Returns normalised histogram [bins] that sums to ~1.
    """
    x_flat      = x.reshape(-1)
    bin_edges   = torch.linspace(min_val, max_val, bins + 1, device=x.device, dtype=x.dtype)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width   = (max_val - min_val) / bins

    diffs   = x_flat.unsqueeze(1) - bin_centers.unsqueeze(0)
    weights = torch.exp(-0.5 * (diffs / bin_width) ** 2)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    hist = weights.sum(dim=0)
    hist = hist / (hist.sum() + 1e-8)
    return hist


def sci_l_scl(
    x_0_tilde:   torch.Tensor,   # [B, 1, H, W]
    mu_prior:    torch.Tensor,   # scalar
    sigma_prior: torch.Tensor,   # scalar
) -> torch.Tensor:
    """
    Statistical Constraint Loss (paper Eq. 14).
    L_scl = |mu_pred - mu_prior| + |sigma_pred - sigma_prior|
    """
    mu_pred    = x_0_tilde.mean()
    sigma_pred = x_0_tilde.std()
    return (mu_pred - mu_prior).abs() + (sigma_pred - sigma_prior).abs()


def sci_l_ccl(
    x_0_tilde: torch.Tensor,   # [B, 1, H, W]
    h_prior:   torch.Tensor,   # [bins]
    bins:      int,
) -> torch.Tensor:
    """
    Channel Constraint Loss (paper Eq. 13).
    Chi-squared distance between predicted-image histogram and prior histogram.
    """
    h_pred = soft_histogram(x_0_tilde, bins)
    eps    = 1e-6
    return ((h_pred - h_prior) ** 2 / (h_pred + h_prior + eps)).sum()


def compute_prior_stats(
    tensors,                              # list or iterator of 1-D tensors
    bins:    int           = 256,
    device:  torch.device  = torch.device("cpu"),
) -> dict:
    """
    Compute mu, sigma, and histogram over all pixels of the target domain.

    Accepts a list or generator of flat tensors. Uses streaming Welford
    algorithm and incremental hard histogram — O(bins) memory regardless
    of dataset size.
    """
    min_val, max_val = -1.0, 1.0

    # Welford's online mean/variance
    count  = torch.tensor(0,   dtype=torch.float64, device=device)
    mean   = torch.tensor(0.0, dtype=torch.float64, device=device)
    M2     = torch.tensor(0.0, dtype=torch.float64, device=device)

    # Hard bin counts for histogram
    bin_counts = torch.zeros(bins, dtype=torch.float64, device=device)

    for t in tensors:
        flat = t.reshape(-1).to(device, dtype=torch.float64)

        # Welford update
        n      = flat.numel()
        delta  = flat - mean
        count += n
        mean  += delta.sum() / count
        delta2 = flat - mean
        M2    += (delta * delta2).sum()

        # Hard histogram accumulation
        indices = ((flat - min_val) / (max_val - min_val) * bins).long().clamp(0, bins - 1)
        bin_counts.scatter_add_(0, indices, torch.ones_like(flat))

    mu    = mean.float()
    sigma = (M2 / (count - 1)).sqrt().float()

    # Normalise histogram
    h = (bin_counts / bin_counts.sum()).float()

    return {"mu": mu, "sigma": sigma, "histogram": h, "bins": torch.tensor(bins)}


def save_prior_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(stats, path)
    print(f"[prior] saved → {path}")


def load_prior_stats(path: str, device: torch.device) -> dict:
    data = torch.load(path, map_location=device)
    return {k: v.to(device) for k, v in data.items()}


def compute_prior_from_dataset(
    prior_dir: str,
    device:    torch.device,
    bins:      int = 256,
    data_root: str = "",
    csv_path:  str = "",
) -> tuple:
    """
    Load the dataset, collect all IR10 and RED4 pixels, compute prior stats,
    and save prior_ir.pt / prior_red.pt to prior_dir.

    Returns (prior_ir, prior_red) dicts.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not data_root:
        data_root = os.path.join(project_root, "data")
    if not csv_path:
        csv_path = os.path.join(project_root, "data", "files", "data_record_bin12.csv")

    print(f"[prior] Loading dataset from {csv_path} ...")
    dr = pd.read_csv(csv_path)

    def _exists(rel_path):
        p = os.path.join(data_root, rel_path)
        if os.name == "nt":
            p = p.replace("/", "\\")
        return os.path.isfile(p)

    dr = dr[dr["Path"].apply(_exists)]
    if dr.empty:
        raise FileNotFoundError(
            f"No .npy files found under data_root='{data_root}'. "
            "Pass --data_root pointing to the directory that contains the CSV paths."
        )
    print(f"[prior] {dr['Observation'].nunique()} observations available — building prior ...")

    dataset = DiffusionDataset(data_record=dr, data_root=data_root, sweep=True)
    print(f"[prior] {len(dataset)} sets — collecting pixels ...")

    def _iter_modality(key):
        for i in range(len(dataset)):
            yield dataset[i][key].reshape(-1)

    prior_ir  = compute_prior_stats(_iter_modality("ir"),  bins=bins, device=device)
    prior_red = compute_prior_stats(_iter_modality("red"), bins=bins, device=device)

    save_prior_stats(prior_ir,  os.path.join(prior_dir, "prior_ir.pt"))
    save_prior_stats(prior_red, os.path.join(prior_dir, "prior_red.pt"))

    return prior_ir, prior_red


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SCI prior statistics")
    parser.add_argument("--data_root", required=True,
                        help="Root directory prepended to CSV paths")
    parser.add_argument("--csv_path",  required=True,
                        help="Path to data_record CSV")
    parser.add_argument("--prior_dir", required=True,
                        help="Output directory for prior_ir.pt and prior_red.pt")
    parser.add_argument("--bins",      type=int, default=256)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.prior_dir, exist_ok=True)

    print(f"Loading CSV  : {args.csv_path}")
    print(f"Data root    : {args.data_root}")
    print(f"Prior dir    : {args.prior_dir}")
    print(f"Bins         : {args.bins}")
    print(f"Device       : {device}")
    print()

    dr = pd.read_csv(args.csv_path)

    def _exists(rel):
        p = os.path.join(args.data_root, rel)
        if os.name == "nt":
            p = p.replace("/", "\\")
        return os.path.isfile(p)

    before = dr["Observation"].nunique()
    dr     = dr[dr["Path"].apply(_exists)]
    after  = dr["Observation"].nunique()
    print(f"Observations : {after} / {before} available")

    if dr.empty:
        raise FileNotFoundError(
            f"No .npy files found under data_root='{args.data_root}'."
        )

    dataset = DiffusionDataset(data_record=dr, data_root=args.data_root, sweep=True)
    n = len(dataset)
    print(f"Dataset sets : {n}")
    print("Collecting pixels ...")

    def _iter_modality(key):
        for i in range(n):
            sample = dataset[i]
            if (i + 1) % 500 == 0 or (i + 1) == n:
                print(f"  [{key}] {i+1}/{n}", flush=True)
            yield sample[key].reshape(-1)

    print("Computing IR10 prior ...")
    prior_ir = compute_prior_stats(_iter_modality("ir"), bins=args.bins, device=device)
    save_prior_stats(prior_ir, os.path.join(args.prior_dir, "prior_ir.pt"))
    print(f"  mu={prior_ir['mu'].item():.4f}  sigma={prior_ir['sigma'].item():.4f}")

    print("Computing RED4 prior ...")
    prior_red = compute_prior_stats(_iter_modality("red"), bins=args.bins, device=device)
    save_prior_stats(prior_red, os.path.join(args.prior_dir, "prior_red.pt"))
    print(f"  mu={prior_red['mu'].item():.4f}  sigma={prior_red['sigma'].item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
