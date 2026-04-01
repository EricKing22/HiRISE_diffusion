"""
Ardan-variant evaluation of a CM-Diff checkpoint.

Reproduces the metric set from HiRISE_img_reconstruction (NMAE%, PWT@1%,
Bias%) alongside the existing metrics (MSE, MAE, PSNR, SSIM, Pearson r, FID)
so that diffusion-model results can be directly compared against the CNN
benchmark.

Key differences from eval.py (which is NOT modified):
  • Three additional metrics: NMAE%, PWT@1%, Bias%  (both physical + normalized)
  • Per-image CSV export
  • Optional per-category evaluation (--category_eval)
  • All helpers live in metrics_ardan.py — no changes to any existing file

Usage:
    cd <project_root>
    python src/eval_ardan.py \\
        --checkpoint src/output/latest.pt \\
        --data_root  /path/to/data \\
        --csv_path   /path/to/data_record_bin12.csv \\
        --save_dir   src/output/evals

    # with per-category evaluation:
    python src/eval_ardan.py ... --category_eval \\
        --evals_config src/configs/evals_ardan.yaml
"""

import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import linalg as sp_linalg

# ── Path setup (same pattern as eval.py) ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ModelConfig, DataConfig, InferenceConfig
from models.cm_diff_unet import UNet
from diffusion.scheduler import DDPMScheduler
from compute_prior import load_prior_stats
from inference import sample
from eval import get_val_split          # reuse identical train/val split
from data.dataset import DiffusionDataset, diffusion_collate_fn, get_loader
from metrics_ardan import (
    nmae_batch, pwt_batch, bias_batch,
    ssim_safe, pearson_batch, psnr_pooled,
)


# =============================================================================
# FID helpers (copied from eval.py — no changes)
# =============================================================================

def _build_inception(device):
    from torchvision import models
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = torch.nn.Identity()
    inception.eval()
    return inception.to(device)


@torch.no_grad()
def _inception_features(images, inception):
    imgs = images.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
    imgs = ((imgs + 10.0) / 20.0).clamp(0.0, 1.0)
    return inception(imgs).cpu().numpy()


def _fid_from_features(real_feats, fake_feats, eps=1e-6):
    mu_r = real_feats.mean(axis=0)
    mu_g = fake_feats.mean(axis=0)
    sr = np.cov(real_feats, rowvar=False) + np.eye(real_feats.shape[1]) * eps
    sg = np.cov(fake_feats, rowvar=False) + np.eye(fake_feats.shape[1]) * eps
    diff_sq = np.sum((mu_r - mu_g) ** 2)
    sqrt_cov, _ = sp_linalg.sqrtm(sr @ sg, disp=False)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real
    return float(np.real(diff_sq + np.trace(sr + sg - 2.0 * sqrt_cov)))


# =============================================================================
# Core evaluation loop
# =============================================================================

def evaluate_loop(
    model,
    scheduler,
    dataset:      DiffusionDataset,
    prior_red:    dict,
    prior_ir:     dict,
    cfg_inf:      InferenceConfig,
    device:       torch.device,
    max_samples:  int  = 0,
    batch_size:   int  = 4,
    seed:         int  = 42,
    compute_fid:  bool = False,
):
    """
    Full T-step reverse diffusion evaluation over a dataset.

    Returns
    -------
    results : dict
        Aggregated metrics (means / pooled PSNRs).
    per_image_rows : list[dict]
        One dict per image per direction — ready for pd.DataFrame / CSV export.
    """
    model.eval()
    n = len(dataset)
    if max_samples > 0:
        n = min(n, max_samples)

    subset = (torch.utils.data.Subset(dataset, list(range(n)))
              if n < len(dataset) else dataset)
    # num_workers=0 on Windows dev machines; 4 on Linux cluster.
    # prefetch_factor and persistent_workers require num_workers > 0.
    _nw = 4 if os.name != "nt" else 0
    loader = get_loader(subset, batch_size=batch_size,
                        collate_fn=diffusion_collate_fn,
                        num_workers=_nw, shuffle=False,
                        prefetch_factor=(2 if _nw > 0 else None),
                        persistent_workers=(_nw > 0))

    # ── Per-image accumulators ────────────────────────────────────────────────
    # Physical space (denormalized DN values)
    mse_p0, mae_p0, nmae_p0, pwt_p0, bias_p0, ssim_0 = [], [], [], [], [], []
    mse_p1, mae_p1, nmae_p1, pwt_p1, bias_p1, ssim_1 = [], [], [], [], [], []
    # Normalized space (model operating range, comparable across scenes)
    mse_n0, nmae_n0, pwt_n0, bias_n0, pearson_0 = [], [], [], [], []
    mse_n1, nmae_n1, pwt_n1, bias_n1, pearson_1 = [], [], [], [], []
    per_image_rows = []

    if compute_fid:
        inception = _build_inception(device)
        real_f0, fake_f0 = [], []
        real_f1, fake_f1 = [], []

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    n_done = 0

    for batch in loader:
        ir_batch  = batch["ir"].to(device)    # [B, 1, H, W]
        red_batch = batch["red"].to(device)   # [B, 1, H, W]
        B = ir_batch.shape[0]

        with torch.no_grad():
            pred_red = sample(model, scheduler, ir_batch, direction=0,
                              prior_stats=prior_red, cfg_inf=cfg_inf,
                              device=device, verbose=False)
            pred_ir  = sample(model, scheduler, red_batch, direction=1,
                              prior_stats=prior_ir, cfg_inf=cfg_inf,
                              device=device, verbose=False)

        pred_red_norm = pred_red.clamp(-10.0, 10.0)
        pred_ir_norm  = pred_ir.clamp(-10.0, 10.0)

        # ── Denormalize → physical DN space ─────────────────────────────────
        # norm_stats: [B, 3] = [center, scale, dc]
        # x_raw = (x_norm + dc) * scale + center
        ns     = batch["norm_stats"]
        center = ns[:, :1,  None, None].to(device)
        scale  = ns[:, 1:2, None, None].to(device)
        dc     = ns[:, 2:3, None, None].to(device)

        pred_red_phys = (pred_red_norm + dc) * scale + center
        pred_ir_phys  = (pred_ir_norm  + dc) * scale + center
        red_gt_phys   = (red_batch     + dc) * scale + center
        ir_gt_phys    = (ir_batch      + dc) * scale + center

        n_done += B

        # ── Physical-space metrics ────────────────────────────────────────────
        mse_p0_b  = F.mse_loss(pred_red_phys, red_gt_phys, reduction="none").mean(dim=[1,2,3])
        mse_p1_b  = F.mse_loss(pred_ir_phys,  ir_gt_phys,  reduction="none").mean(dim=[1,2,3])
        mae_p0_b  = F.l1_loss( pred_red_phys, red_gt_phys, reduction="none").mean(dim=[1,2,3])
        mae_p1_b  = F.l1_loss( pred_ir_phys,  ir_gt_phys,  reduction="none").mean(dim=[1,2,3])
        nmae_p0_b = nmae_batch(pred_red_phys, red_gt_phys)
        nmae_p1_b = nmae_batch(pred_ir_phys,  ir_gt_phys)
        pwt_p0_b  = pwt_batch( pred_red_phys, red_gt_phys)
        pwt_p1_b  = pwt_batch( pred_ir_phys,  ir_gt_phys)
        bias_p0_b = bias_batch(pred_red_phys, red_gt_phys)
        bias_p1_b = bias_batch(pred_ir_phys,  ir_gt_phys)

        # ── Normalized-space metrics ──────────────────────────────────────────
        mse_n0_b  = F.mse_loss(pred_red_norm, red_batch, reduction="none").mean(dim=[1,2,3])
        mse_n1_b  = F.mse_loss(pred_ir_norm,  ir_batch,  reduction="none").mean(dim=[1,2,3])
        nmae_n0_b = nmae_batch(pred_red_norm, red_batch)
        nmae_n1_b = nmae_batch(pred_ir_norm,  ir_batch)
        pwt_n0_b  = pwt_batch( pred_red_norm, red_batch)
        pwt_n1_b  = pwt_batch( pred_ir_norm,  ir_batch)
        bias_n0_b = bias_batch(pred_red_norm, red_batch)
        bias_n1_b = bias_batch(pred_ir_norm,  ir_batch)
        pearson0_b = pearson_batch(pred_red_norm, red_batch)
        pearson1_b = pearson_batch(pred_ir_norm,  ir_batch)

        # ── SSIM: physical space, GT-normalized to [0,1] per image ───────────
        ssim0_vals, ssim1_vals = [], []
        for i in range(B):
            gt0_np   = red_gt_phys[i].squeeze().cpu().numpy()
            pred0_np = pred_red_phys[i].squeeze().cpu().numpy()
            gt1_np   = ir_gt_phys[i].squeeze().cpu().numpy()
            pred1_np = pred_ir_phys[i].squeeze().cpu().numpy()
            ssim0_vals.append(ssim_safe(gt0_np, pred0_np))
            ssim1_vals.append(ssim_safe(gt1_np, pred1_np))

        # ── Accumulate lists ──────────────────────────────────────────────────
        mse_p0.extend(mse_p0_b.cpu().tolist())
        mae_p0.extend(mae_p0_b.cpu().tolist())
        nmae_p0.extend(nmae_p0_b.cpu().tolist())
        pwt_p0.extend(pwt_p0_b.cpu().tolist())
        bias_p0.extend(bias_p0_b.cpu().tolist())
        ssim_0.extend(ssim0_vals)

        mse_p1.extend(mse_p1_b.cpu().tolist())
        mae_p1.extend(mae_p1_b.cpu().tolist())
        nmae_p1.extend(nmae_p1_b.cpu().tolist())
        pwt_p1.extend(pwt_p1_b.cpu().tolist())
        bias_p1.extend(bias_p1_b.cpu().tolist())
        ssim_1.extend(ssim1_vals)

        mse_n0.extend(mse_n0_b.cpu().tolist())
        nmae_n0.extend(nmae_n0_b.cpu().tolist())
        pwt_n0.extend(pwt_n0_b.cpu().tolist())
        bias_n0.extend(bias_n0_b.cpu().tolist())
        pearson_0.extend(pearson0_b.cpu().tolist())

        mse_n1.extend(mse_n1_b.cpu().tolist())
        nmae_n1.extend(nmae_n1_b.cpu().tolist())
        pwt_n1.extend(pwt_n1_b.cpu().tolist())
        bias_n1.extend(bias_n1_b.cpu().tolist())
        pearson_1.extend(pearson1_b.cpu().tolist())

        # ── Per-image CSV rows ────────────────────────────────────────────────
        obs_ids   = batch["obs_id"]
        set_names = batch["set_name"]
        dates     = batch["date"]
        for i in range(B):
            common = dict(
                obs_id=obs_ids[i],
                set_name=set_names[i],
                date=dates[i],
            )
            per_image_rows.append({**common,
                "direction": "IR2RED",
                "mse_phys":  mse_p0_b[i].item(),
                "mae_phys":  mae_p0_b[i].item(),
                "nmae_phys": nmae_p0_b[i].item(),
                "pwt_phys":  pwt_p0_b[i].item(),
                "bias_phys": bias_p0_b[i].item(),
                "ssim":      ssim0_vals[i],
                "mse_norm":  mse_n0_b[i].item(),
                "nmae_norm": nmae_n0_b[i].item(),
                "pwt_norm":  pwt_n0_b[i].item(),
                "bias_norm": bias_n0_b[i].item(),
                "pearson":   pearson0_b[i].item(),
            })
            per_image_rows.append({**common,
                "direction": "RED2IR",
                "mse_phys":  mse_p1_b[i].item(),
                "mae_phys":  mae_p1_b[i].item(),
                "nmae_phys": nmae_p1_b[i].item(),
                "pwt_phys":  pwt_p1_b[i].item(),
                "bias_phys": bias_p1_b[i].item(),
                "ssim":      ssim1_vals[i],
                "mse_norm":  mse_n1_b[i].item(),
                "nmae_norm": nmae_n1_b[i].item(),
                "pwt_norm":  pwt_n1_b[i].item(),
                "bias_norm": bias_n1_b[i].item(),
                "pearson":   pearson1_b[i].item(),
            })

        # ── FID ───────────────────────────────────────────────────────────────
        if compute_fid:
            real_f0.append(_inception_features(red_batch,     inception))
            fake_f0.append(_inception_features(pred_red_norm, inception))
            real_f1.append(_inception_features(ir_batch,      inception))
            fake_f1.append(_inception_features(pred_ir_norm,  inception))

        # ── Progress ──────────────────────────────────────────────────────────
        print(f"  [{n_done:>{len(str(n))}}/{n}]  "
              f"MSE(IR→RED)={mse_p0_b.mean():.4f}/{mse_n0_b.mean():.4f}  "
              f"NMAE={nmae_p0_b.mean():.1f}%  SSIM={np.mean(ssim0_vals):.4f}  | "
              f"MSE(RED→IR)={mse_p1_b.mean():.4f}/{mse_n1_b.mean():.4f}  "
              f"NMAE={nmae_p1_b.mean():.1f}%  SSIM={np.mean(ssim1_vals):.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg = lambda lst: float(np.mean(lst))

    results = dict(
        n_samples=n_done,
        # Physical
        mse_phys_ir2red  = avg(mse_p0),
        mae_phys_ir2red  = avg(mae_p0),
        psnr_phys_ir2red = psnr_pooled(mse_p0, 1.0),
        nmae_phys_ir2red = avg(nmae_p0),
        pwt_phys_ir2red  = avg(pwt_p0),
        bias_phys_ir2red = avg(bias_p0),
        ssim_ir2red      = avg(ssim_0),
        mse_phys_red2ir  = avg(mse_p1),
        mae_phys_red2ir  = avg(mae_p1),
        psnr_phys_red2ir = psnr_pooled(mse_p1, 1.0),
        nmae_phys_red2ir = avg(nmae_p1),
        pwt_phys_red2ir  = avg(pwt_p1),
        bias_phys_red2ir = avg(bias_p1),
        ssim_red2ir      = avg(ssim_1),
        # Normalized
        mse_norm_ir2red  = avg(mse_n0),
        psnr_norm_ir2red = psnr_pooled(mse_n0, 20.0),
        nmae_norm_ir2red = avg(nmae_n0),
        pwt_norm_ir2red  = avg(pwt_n0),
        bias_norm_ir2red = avg(bias_n0),
        pearson_ir2red   = avg(pearson_0),
        mse_norm_red2ir  = avg(mse_n1),
        psnr_norm_red2ir = psnr_pooled(mse_n1, 20.0),
        nmae_norm_red2ir = avg(nmae_n1),
        pwt_norm_red2ir  = avg(pwt_n1),
        bias_norm_red2ir = avg(bias_n1),
        pearson_red2ir   = avg(pearson_1),
    )

    if compute_fid and real_f0:
        real_0 = np.concatenate(real_f0, axis=0)
        fake_0 = np.concatenate(fake_f0, axis=0)
        real_1 = np.concatenate(real_f1, axis=0)
        fake_1 = np.concatenate(fake_f1, axis=0)
        results["fid_ir2red"] = _fid_from_features(real_0, fake_0)
        results["fid_red2ir"] = _fid_from_features(real_1, fake_1)
        print(f"  FID(IR→RED)={results['fid_ir2red']:.2f}  "
              f"FID(RED→IR)={results['fid_red2ir']:.2f}")

    return results, per_image_rows


# =============================================================================
# Per-category evaluation
# =============================================================================

def evaluate_categories(
    model, scheduler, prior_red, prior_ir, cfg_inf, device,
    dr, data_root, evals_config_path,
    batch_size=4, seed=42,
):
    """
    Run per-category evaluation using evals_ardan.yaml.

    For each category:
      1. Filter data_record to the category's observation IDs
      2. Take one Set per Observation (matches img_reconstruction approach)
      3. Run evaluate_loop on that DiffusionDataset subset (FID disabled)

    Returns
    -------
    category_results : dict  category_name → results dict
    all_rows         : list[dict]  per-image rows tagged with 'category'
    """
    import yaml
    with open(evals_config_path) as f:
        eval_cats = yaml.safe_load(f)
    eval_cats = {k: v for k, v in eval_cats.items()
                 if v and v.get("oids")}

    if not eval_cats:
        print("WARNING: evals_ardan.yaml has no categories with observation IDs.")
        return {}, []

    category_results = {}
    all_rows = []

    for cat_name, cat_data in eval_cats.items():
        oids = cat_data["oids"]
        # One set per observation — same logic as img_reconstruction
        cat_sets = (
            dr[dr["Observation"].isin(oids)]
            .drop_duplicates("Observation")["Set"]
            .tolist()
        )
        if not cat_sets:
            print(f"  WARNING: '{cat_name}' — no matching observations in data record, skipping")
            continue

        try:
            cat_dataset = DiffusionDataset(
                data_record=dr,
                data_root=data_root,
                sweep=True,
                allowed_sets=cat_sets,
            )
        except ValueError as e:
            print(f"  WARNING: '{cat_name}' — {e}, skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"Category: {cat_name}  ({len(cat_dataset)} images)")
        print(f"{'─'*60}")

        cat_results, cat_rows = evaluate_loop(
            model, scheduler, cat_dataset,
            prior_red=prior_red, prior_ir=prior_ir,
            cfg_inf=cfg_inf, device=device,
            max_samples=0, batch_size=batch_size,
            seed=seed, compute_fid=False,
        )

        for row in cat_rows:
            row["category"] = cat_name
        all_rows.extend(cat_rows)
        category_results[cat_name] = cat_results

    return category_results, all_rows


# =============================================================================
# Reporting
# =============================================================================

def print_results_table(results, lambda_scl=0.0, lambda_ccl=0.0):
    """Print the full metric table to stdout."""
    W = 20

    def c(k0, k1):
        return (results[k0] + results[k1]) / 2

    print(f"\n{'='*76}")
    print(f"{'Metric':<{W}}  {'IR→RED':>10}  {'RED→IR':>10}  {'Combined':>10}")
    print(f"{'─'*76}")

    print(f"{'[Physical DN]':<{W}}")
    print(f"{'  MSE':<{W}}  {results['mse_phys_ir2red']:>10.4f}  {results['mse_phys_red2ir']:>10.4f}  "
          f"{c('mse_phys_ir2red','mse_phys_red2ir'):>10.4f}")
    print(f"{'  MAE':<{W}}  {results['mae_phys_ir2red']:>10.4f}  {results['mae_phys_red2ir']:>10.4f}  {'—':>10}")
    print(f"{'  PSNR (dB)':<{W}}  {results['psnr_phys_ir2red']:>10.2f}  {results['psnr_phys_red2ir']:>10.2f}  "
          f"{c('psnr_phys_ir2red','psnr_phys_red2ir'):>10.2f}")
    print(f"{'  NMAE%':<{W}}  {results['nmae_phys_ir2red']:>10.2f}  {results['nmae_phys_red2ir']:>10.2f}  "
          f"{c('nmae_phys_ir2red','nmae_phys_red2ir'):>10.2f}")
    print(f"{'  PWT @ 1% (%)':<{W}}  {results['pwt_phys_ir2red']:>10.2f}  {results['pwt_phys_red2ir']:>10.2f}  "
          f"{c('pwt_phys_ir2red','pwt_phys_red2ir'):>10.2f}")
    print(f"{'  Bias%':<{W}}  {results['bias_phys_ir2red']:>+10.2f}  {results['bias_phys_red2ir']:>+10.2f}  "
          f"{c('bias_phys_ir2red','bias_phys_red2ir'):>+10.2f}")
    print(f"{'  SSIM':<{W}}  {results['ssim_ir2red']:>10.4f}  {results['ssim_red2ir']:>10.4f}  "
          f"{c('ssim_ir2red','ssim_red2ir'):>10.4f}")

    print(f"{'[Normalized]':<{W}}")
    print(f"{'  MSE':<{W}}  {results['mse_norm_ir2red']:>10.4f}  {results['mse_norm_red2ir']:>10.4f}  "
          f"{c('mse_norm_ir2red','mse_norm_red2ir'):>10.4f}")
    print(f"{'  PSNR (dB)':<{W}}  {results['psnr_norm_ir2red']:>10.2f}  {results['psnr_norm_red2ir']:>10.2f}  "
          f"{c('psnr_norm_ir2red','psnr_norm_red2ir'):>10.2f}")
    print(f"{'  NMAE%':<{W}}  {results['nmae_norm_ir2red']:>10.2f}  {results['nmae_norm_red2ir']:>10.2f}  "
          f"{c('nmae_norm_ir2red','nmae_norm_red2ir'):>10.2f}")
    print(f"{'  PWT @ 1% (%)':<{W}}  {results['pwt_norm_ir2red']:>10.2f}  {results['pwt_norm_red2ir']:>10.2f}  "
          f"{c('pwt_norm_ir2red','pwt_norm_red2ir'):>10.2f}")
    print(f"{'  Bias%':<{W}}  {results['bias_norm_ir2red']:>+10.2f}  {results['bias_norm_red2ir']:>+10.2f}  "
          f"{c('bias_norm_ir2red','bias_norm_red2ir'):>+10.2f}")
    print(f"{'  Pearson r':<{W}}  {results['pearson_ir2red']:>10.4f}  {results['pearson_red2ir']:>10.4f}  "
          f"{c('pearson_ir2red','pearson_red2ir'):>10.4f}")

    if "fid_ir2red" in results:
        print(f"{'[FID]':<{W}}")
        print(f"{'  FID':<{W}}  {results['fid_ir2red']:>10.2f}  {results['fid_red2ir']:>10.2f}  {'—':>10}")

    print(f"{'='*76}")
    print(f"n={results['n_samples']}  λ_scl={lambda_scl}  λ_ccl={lambda_ccl}")
    print(f"Physical PSNR: MAX=1.0  |  Normalized PSNR: MAX=20 (clamp range [-10,10])")
    print(f"NMAE%: 100×mean(|err|)/mean(|GT|)  |  PWT@1%: within 1% of GT range")
    print(f"Bias%: 100×mean(err)/mean(|GT|), signed (+→over-predict, -→under-predict)")
    print(f"SSIM: physical space, GT-normalized to [0,1] per image (via skimage)")


def print_category_report(category_results):
    """Print per-category metric tables."""
    if not category_results:
        return

    W = 22

    metrics = [
        ("NMAE% [Physical]",    "nmae_phys_ir2red", "nmae_phys_red2ir", ".2f",  False),
        ("PWT@1% [Physical]",   "pwt_phys_ir2red",  "pwt_phys_red2ir",  ".2f",  False),
        ("Bias% [Physical]",    "bias_phys_ir2red", "bias_phys_red2ir", "+.2f", False),
        ("SSIM",                "ssim_ir2red",      "ssim_red2ir",      ".4f",  False),
        ("Pearson r",           "pearson_ir2red",   "pearson_red2ir",   ".4f",  False),
        ("NMAE% [Normalized]",  "nmae_norm_ir2red", "nmae_norm_red2ir", ".2f",  False),
    ]

    for title, k0, k1, fmt, _ in metrics:
        print(f"\n{'='*76}")
        print(f"{title} per category")
        print(f"{'='*76}")
        print(f"| {'Category':<{W}} | {'IR→RED':>10} | {'RED→IR':>10} | {'Combined':>10} | {'N':>5} |")
        print(f"|{'-'*(W+2)}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*7}|")
        for cat, res in category_results.items():
            v0 = res[k0]
            v1 = res[k1]
            vc = (v0 + v1) / 2
            n  = res["n_samples"]
            s0, s1, sc = format(v0, fmt), format(v1, fmt), format(vc, fmt)
            print(f"| {cat:<{W}.{W}} | {s0:>10} | {s1:>10} | {sc:>10} | {n:>5} |")


# =============================================================================
# CSV export
# =============================================================================

def save_csv(rows, save_dir, filename):
    """Save a list of per-image metric dicts to CSV."""
    if not rows:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} rows → {path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CM-Diff Ardan-variant evaluation (NMAE / PWT / Bias + CSV + categories)"
    )
    parser.add_argument("--checkpoint",   default="src/output/latest.pt")
    parser.add_argument("--prior_dir",    default="src/output",
                        help="Directory containing prior_ir.pt and prior_red.pt")
    parser.add_argument("--data_root",    default="")
    parser.add_argument("--csv_path",     default="")
    parser.add_argument("--max_samples",  type=int,   default=0,
                        help="Max samples (0 = all val)")
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lambda_scl",   type=float, default=0.0)
    parser.add_argument("--lambda_ccl",   type=float, default=0.0)
    parser.add_argument("--no_fid",       action="store_true", default=True,
                        help="Skip FID computation (faster)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--save_dir",     default="src/output/evals",
                        help="Directory for per-image CSV and category CSV")
    parser.add_argument("--category_eval", action="store_true",
                        help="Run per-category evaluation (slow — 14 cats × 20 obs)")
    parser.add_argument("--evals_config",
                        default="src/configs/evals_ardan.yaml",
                        help="Path to evals_ardan.yaml")
    args = parser.parse_args()

    cfg_model = ModelConfig()
    cfg_data  = DataConfig()
    cfg_inf   = InferenceConfig(lambda_scl=args.lambda_scl,
                                lambda_ccl=args.lambda_ccl)
    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = (args.data_root or cfg_data.data_root
                 or os.path.join(project_root, "data"))
    csv_path  = (args.csv_path or cfg_data.csv_path
                 or os.path.join(project_root, "data", "files", "data_record_bin12.csv"))

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"SCI          : λ_scl={args.lambda_scl}  λ_ccl={args.lambda_ccl}")
    print(f"Max samples  : {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"FID          : {'disabled' if args.no_fid else 'enabled'}")
    print(f"Category eval: {'yes — ' + args.evals_config if args.category_eval else 'no'}")
    print(f"Save dir     : {args.save_dir}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(
        in_channels=cfg_model.in_channels,
        out_channels=cfg_model.out_channels,
        base_channels=cfg_model.base_channels,
        num_res_blocks=cfg_model.num_res_blocks,
        dropout=cfg_model.dropout,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"Checkpoint step={step}  train_loss={loss:.4f}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = DDPMScheduler(
        timesteps=cfg_model.timesteps,
        beta_start=cfg_model.beta_start,
        beta_end=cfg_model.beta_end,
    ).to(device)

    # ── Priors ────────────────────────────────────────────────────────────────
    prior_red = load_prior_stats(os.path.join(args.prior_dir, "prior_red.pt"), device)
    prior_ir  = load_prior_stats(os.path.join(args.prior_dir, "prior_ir.pt"),  device)
    print(f"Prior RED : mu={prior_red['mu'].item():.4f}  sigma={prior_red['sigma'].item():.4f}")
    print(f"Prior IR  : mu={prior_ir['mu'].item():.4f}   sigma={prior_ir['sigma'].item():.4f}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dr = pd.read_csv(csv_path)
    _, val_sets = get_val_split(dr)
    val_sets = [19645, 7292, 7293, 14774]   # local test sets (override for dev)
    val_dataset = DiffusionDataset(
        data_record=dr, data_root=data_root, sweep=True, allowed_sets=val_sets,
    )
    print(f"Val set      : {len(val_dataset)} images\n")

    # ── Global evaluation ─────────────────────────────────────────────────────
    print("=== Global evaluation ===")
    results, per_image_rows = evaluate_loop(
        model, scheduler, val_dataset,
        prior_red=prior_red, prior_ir=prior_ir,
        cfg_inf=cfg_inf, device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        compute_fid=not args.no_fid,
    )

    print_results_table(results, args.lambda_scl, args.lambda_ccl)

    # ── Export per-image CSV ──────────────────────────────────────────────────
    run_label = f"lscl{args.lambda_scl}_lccl{args.lambda_ccl}"
    save_csv(per_image_rows, args.save_dir,
             f"eval_per_image_{run_label}.csv")

    # ── Per-category evaluation ───────────────────────────────────────────────
    if args.category_eval:
        if not os.path.isfile(args.evals_config):
            print(f"\nERROR: evals_config not found: {args.evals_config}")
            print("  Run:  python src/configs/gen_eval_categories_ardan.py")
            return

        print(f"\n=== Per-category evaluation ({args.evals_config}) ===")
        cat_results, cat_rows = evaluate_categories(
            model, scheduler, prior_red, prior_ir, cfg_inf, device,
            dr=dr, data_root=data_root,
            evals_config_path=args.evals_config,
            batch_size=args.batch_size, seed=args.seed,
        )

        print("\n\n=== Category summary ===")
        print_category_report(cat_results)

        save_csv(cat_rows, args.save_dir,
                 f"eval_categories_{run_label}.csv")


if __name__ == "__main__":
    main()
