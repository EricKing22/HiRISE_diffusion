"""
Generate evals_ardan.yaml — bin12-compatible evaluation category definitions.

Selects ~20 validation-split observations per category using metadata columns
already present in data_record_bin12.csv:
  Emission_angle, Incidence_angle, Image_center_lat, Solar_longitude,
  Pix_min, Pix_max

Run once from the project root:
    python src/configs/gen_eval_categories_ardan.py \\
        --csv_path  data/files/data_record_bin12.csv \\
        --output    src/configs/evals_ardan.yaml

The generated YAML is then consumed by eval_ardan.py --category_eval.

Categories generated:
  Geometry  : nadir, off_nadir
  Illumination: low_sun, high_sun
  Location  : polar, equatorial
  Seasonal  : summer_north, summer_south, winter_north, winter_south
  Contrast  : high_contrast, low_contrast
  Brightness: high_brightness, low_brightness
"""

import os
import sys
import argparse
from datetime import date

import pandas as pd
import yaml

# ── Add project root to path so we can import eval.get_val_split ─────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # src/configs/
_SRC  = os.path.dirname(_HERE)                              # src/
_ROOT = os.path.dirname(_SRC)                               # project root
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)

from eval import get_val_split   # reuse the same 80/20 split as training


N_PER_CAT = 20   # observations per category


def _obs_meta(dr: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per observation using IR10 CCD rows, then deduplicate.
    All metadata columns are identical across CCDs within an observation,
    so filtering to IR10 and deduplicating gives one clean metadata row each.
    """
    ir_rows = dr[dr["CCD"] == "IR10"].copy()
    return ir_rows.drop_duplicates("Observation").set_index("Observation")


def _pick(meta: pd.DataFrame, col: str, ascending: bool,
          n: int = N_PER_CAT) -> list:
    """Sort by `col` and return the top-n observation IDs."""
    sorted_obs = meta.sort_values(col, ascending=ascending)
    return sorted_obs.index.tolist()[:n]


def _pick_filtered(meta: pd.DataFrame, mask: pd.Series,
                   col: str, ascending: bool, n: int = N_PER_CAT) -> list:
    """Filter by boolean mask, then sort by `col`, return top-n."""
    sub = meta[mask].sort_values(col, ascending=ascending)
    return sub.index.tolist()[:n]


def build_categories(dr: pd.DataFrame) -> dict:
    """
    Build category observation lists from the validation-split metadata.

    Returns dict: category_name → list[str] of observation IDs.
    """
    _, val_sets = get_val_split(dr)
    val_dr = dr[dr["Set"].isin(val_sets)]
    meta = _obs_meta(val_dr)

    today = date.today().isoformat()
    cats = {}

    def entry(desc, oids, cat_type="Metric"):
        return {
            "description": desc,
            "generation": {"method": "Automatic", "date": today},
            "oids": oids,
            "eval_category": cat_type,
        }

    # ── Viewing geometry ──────────────────────────────────────────────────────
    cats["nadir"] = entry(
        f"Top {N_PER_CAT} val observations with smallest emission angles (bin12)",
        _pick(meta, "Emission_angle", ascending=True),
    )
    cats["off_nadir"] = entry(
        f"Top {N_PER_CAT} val observations with largest emission angles (bin12)",
        _pick(meta, "Emission_angle", ascending=False),
    )

    # ── Illumination ─────────────────────────────────────────────────────────
    cats["low_sun"] = entry(
        f"Top {N_PER_CAT} val observations with largest incidence angles (bin12)",
        _pick(meta, "Incidence_angle", ascending=False),
    )
    cats["high_sun"] = entry(
        f"Top {N_PER_CAT} val observations with smallest incidence angles (bin12)",
        _pick(meta, "Incidence_angle", ascending=True),
    )

    # ── Location ─────────────────────────────────────────────────────────────
    polar_mask = meta["Image_center_lat"].abs() > 60.0
    polar_obs = meta[polar_mask].index.tolist()
    # Take up to N from north and N from south to balance hemispheres
    north_polar = meta[meta["Image_center_lat"] >  60.0].index.tolist()[:N_PER_CAT // 2]
    south_polar = meta[meta["Image_center_lat"] < -60.0].index.tolist()[:N_PER_CAT // 2]
    polar_combined = (north_polar + south_polar)[:N_PER_CAT]
    if len(polar_combined) < N_PER_CAT:
        # Fall back to all |lat|>60 sorted by |lat| descending
        polar_combined = (meta[polar_mask]
                          .assign(_abslat=meta["Image_center_lat"].abs())
                          .sort_values("_abslat", ascending=False)
                          .index.tolist()[:N_PER_CAT])
    cats["polar"] = entry(
        f"Up to {N_PER_CAT} val observations with |lat| > 60° (bin12)",
        polar_combined,
    )

    eq_mask = meta["Image_center_lat"].abs() < 5.0
    cats["equatorial"] = entry(
        f"Top {N_PER_CAT} val observations with |lat| < 5° (bin12)",
        meta[eq_mask].index.tolist()[:N_PER_CAT],
    )

    # ── Seasonal (Ls = Solar longitude, hemisphere = latitude sign) ───────────
    # Northern summer:  Ls 90–180, lat > 0
    # Southern summer:  Ls 90–180, lat < 0
    # Northern winter:  Ls 270–360, lat > 0
    # Southern winter:  Ls 270–360, lat < 0
    ls = meta["Solar_longitude"]
    lat = meta["Image_center_lat"]

    cats["summer_north"] = entry(
        f"Top {N_PER_CAT} val obs, Ls 90–180, lat > 0 (Northern summer, bin12)",
        meta[(ls >= 90) & (ls < 180) & (lat > 0)].index.tolist()[:N_PER_CAT],
    )
    cats["summer_south"] = entry(
        f"Top {N_PER_CAT} val obs, Ls 90–180, lat < 0 (Southern winter, bin12)",
        meta[(ls >= 90) & (ls < 180) & (lat < 0)].index.tolist()[:N_PER_CAT],
    )
    cats["winter_north"] = entry(
        f"Top {N_PER_CAT} val obs, Ls 270–360, lat > 0 (Northern winter, bin12)",
        meta[(ls >= 270) & (ls < 360) & (lat > 0)].index.tolist()[:N_PER_CAT],
    )
    cats["winter_south"] = entry(
        f"Top {N_PER_CAT} val obs, Ls 270–360, lat < 0 (Southern summer, bin12)",
        meta[(ls >= 270) & (ls < 360) & (lat < 0)].index.tolist()[:N_PER_CAT],
    )

    # ── Contrast (using per-observation pixel range from Pix_min / Pix_max) ──
    meta = meta.copy()
    meta["_contrast"] = meta["Pix_max"] - meta["Pix_min"]
    cats["high_contrast"] = entry(
        f"Top {N_PER_CAT} val observations with largest pixel range (bin12)",
        meta.sort_values("_contrast", ascending=False).index.tolist()[:N_PER_CAT],
    )
    cats["low_contrast"] = entry(
        f"Top {N_PER_CAT} val observations with smallest pixel range (bin12)",
        meta.sort_values("_contrast", ascending=True).index.tolist()[:N_PER_CAT],
    )

    # ── Brightness ────────────────────────────────────────────────────────────
    meta["_brightness"] = (meta["Pix_min"] + meta["Pix_max"]) / 2.0
    cats["high_brightness"] = entry(
        f"Top {N_PER_CAT} val observations with highest mean brightness (bin12)",
        meta.sort_values("_brightness", ascending=False).index.tolist()[:N_PER_CAT],
    )
    cats["low_brightness"] = entry(
        f"Top {N_PER_CAT} val observations with lowest mean brightness (bin12)",
        meta.sort_values("_brightness", ascending=True).index.tolist()[:N_PER_CAT],
    )

    return cats


def main():
    parser = argparse.ArgumentParser(
        description="Generate evals_ardan.yaml from bin12 metadata"
    )
    parser.add_argument(
        "--csv_path",
        default="data/files/data_record_bin12.csv",
        help="Path to data_record_bin12.csv",
    )
    parser.add_argument(
        "--output",
        default="src/configs/evals_ardan.yaml",
        help="Output path for the generated YAML",
    )
    args = parser.parse_args()

    print(f"Loading data record: {args.csv_path}")
    dr = pd.read_csv(args.csv_path)
    print(f"  {dr['Observation'].nunique()} observations, {dr['Set'].nunique()} sets")

    cats = build_categories(dr)

    # Report
    total_obs = 0
    for name, data in cats.items():
        n = len(data["oids"])
        total_obs += n
        status = "OK" if n >= 10 else "SPARSE"
        print(f"  [{status}] {name:<20} {n:>3} observations")

    print(f"\nTotal: {len(cats)} categories, {total_obs} observations")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(cats, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
