# HiRISE Raw IR10/RED4 Distribution Analysis

This folder contains reproducible materials for the normalization analysis used
to motivate the `DiffusionDataset` preprocessing pipeline. The current stage
documents the data distribution before any normalization is applied.

## Data Source

The analysis uses the shared HiRISE bin12 dataset:

```text
/scratch_root/as5023/HiRISE/data/
/scratch_root/as5023/HiRISE/data/data_record_bin12.csv
```

The script validates paired IR10/RED4 availability through the CSV `Set` field
and resolves exported relative paths such as `../../files/npy_files_b12/...`
to the shared cluster layout.

Dataset coverage in this analysis:

| Quantity | Value |
|---|---:|
| CSV rows | 89,365 |
| Observations | 12,495 |
| Unique sets | 17,873 |
| Paired IR10/RED4 sets loaded | 17,873 |
| Missing paired files | 0 |
| CCD rows per channel | 17,873 each for BG12, IR10, RED3, RED4, RED5 |
| Pixels per modality | 1,171,324,928 |

## Reproducibility

The analysis script is:

```text
data/normalization_analysis/scripts/raw_distribution_analysis.py
```

It writes the raw-stage resources into:

```text
data/normalization_analysis/raw/
```

The script reads all paired IR10 and RED4 `.npy` arrays, computes per-set image
statistics, accumulates full-dataset pixel histograms with 512 bins, and saves
publication-oriented figures. It does not apply median/MAD scaling, DC
subtraction, gain, or clipping.

To rerun:

```bash
/scratch_root/ed425/miniconda3/envs/HiPredict/bin/python \
  data/normalization_analysis/scripts/raw_distribution_analysis.py
```

Note: on this machine the command may need to run outside the local command
sandbox because the project has a known `bwrap: loopback` issue.

## Raw Output Files

| File | Meaning | Use in paper analysis |
|---|---|---|
| `raw_summary.json` | Compact machine-readable summary of dataset counts, per-set distribution statistics, histogram settings, and pixel totals. | Source of exact numeric values used in text and tables. |
| `raw_per_set_stats.csv` | One row per paired set. Includes raw IR10/RED4 min, max, mean, standard deviation, 5th/50th/95th percentiles, RED4-minus-IR10 mean offset, and RED4/IR10 standard-deviation ratio. | Supports scene-level analysis, filtering by outliers, and generating additional plots. |
| `raw_pixel_histogram.csv` | 512-bin full-dataset pixel histogram for raw IR10 and RED4 reflectance. | Supports exact reproduction of the raw pixel distribution curve. |
| `raw_pixel_distribution.png` | Full-dataset pixel-density curves for raw IR10 and RED4. | Figure showing that the two bands occupy similar but shifted reflectance ranges before normalization. |
| `raw_per_set_distributions.png` | Four-panel scene-level plot: per-set means, per-set standard deviations, RED4-minus-IR10 mean offset, and IR10-vs-RED4 mean scatter. | Main evidence for scene-to-scene brightness variability and systematic band offset. |
| `raw_pixel_pair_hexbin.png` | Hexbin plot of sampled aligned raw IR10/RED4 pixel pairs. Pixel pairs are sampled deterministically with stride 257 to keep the figure size manageable. | Shows strong aligned cross-band relationship while preserving a nonzero spectral offset. |
| `raw_example_patches.png` | Example raw paired IR10/RED4 patches displayed with shared grayscale limits per pair. | Qualitative check that the paired bands are spatially aligned and have related structure. |

## Main Raw-Data Findings

The unnormalized raw reflectance values are already physically meaningful, but
they are not in a numerically stable or distributionally standardized form for
diffusion or flow-matching training.

### Scene-Level Brightness Varies Strongly

Per-set raw mean reflectance:

| Statistic | IR10 mean | RED4 mean |
|---|---:|---:|
| Mean across sets | 0.118784 | 0.108312 |
| Std across sets | 0.048004 | 0.040154 |
| 5th percentile | 0.051414 | 0.050688 |
| Median | 0.111950 | 0.103642 |
| 95th percentile | 0.206031 | 0.180548 |
| 99th percentile | 0.234208 | 0.209240 |
| Maximum | 0.405701 | 0.276699 |

This wide scene-level brightness range is the first motivation for per-scene
normalization. A single global affine transform would be dominated by bright
and dark extremes and would leave many low-contrast scenes compressed into a
very small numerical interval.

### RED4 Is Usually Dimmer Than IR10

Per-set mean offset, defined as:

```text
mean(RED4) - mean(IR10)
```

has the following distribution:

| Statistic | Offset |
|---|---:|
| Mean | -0.010472 |
| Std | 0.012358 |
| 5th percentile | -0.028981 |
| Median | -0.008067 |
| 95th percentile | 0.000715 |
| 99th percentile | 0.018933 |

The median and mean offsets are negative, so RED4 is typically darker than
IR10 for the same paired patch. This is a physical cross-band signal, not a
nuisance to remove. Therefore IR10 and RED4 should not be independently
normalized to zero mean or to a common min/max range, because independent
normalization would erase the band offset that the translation model must
learn.

This supports the current `DiffusionDataset` design: compute normalization
statistics from IR10 and apply the same affine transform to both IR10 and RED4.

### Most Patches Are Very Low Contrast

Per-set raw standard deviation:

| Statistic | IR10 std | RED4 std |
|---|---:|---:|
| Mean across sets | 0.005765 | 0.004194 |
| 5th percentile | 0.000871 | 0.001040 |
| Median | 0.003559 | 0.003345 |
| 95th percentile | 0.016441 | 0.010124 |
| 99th percentile | 0.048925 | 0.015889 |

The median patch-level standard deviation is only about `0.0035` reflectance
units. This is crucial for the later median/MAD normalization step. If the raw
MAD of a low-contrast scene were used without a lower bound, the normalized
image could be amplified by hundreds of times, making statistical priors,
histograms, and diffusion/flow targets unstable. This motivates the current
scale floor in `DiffusionDataset`:

```text
scale = max(1.4826 * MAD(IR10), 0.05)
```

The scale floor deliberately avoids treating tiny raw contrast as permission
to create huge normalized values.

### Pixel-Level Relationship Is Strong But Not Identity

The `raw_pixel_pair_hexbin.png` figure shows that aligned IR10 and RED4 pixels
are strongly correlated. This is expected because they observe the same
surface patch. However, the dense region is not exactly the identity line.
The model should therefore learn a cross-band translation, not merely copy the
source image.

This motivates a paired conditional generative model:

```text
IR10 -> RED4
RED4 -> IR10
```

and also motivates preserving the raw inter-band offset during normalization.

## Interpretation For The Normalization Pipeline

These raw-data results support the following normalization choices:

1. **Use per-scene normalization rather than only global normalization.**  
   Scene brightness varies substantially across the dataset, with IR10 per-set
   means spanning from approximately `0` to `0.406` and RED4 means spanning
   from approximately `0` to `0.277`.

2. **Use robust IR10 statistics as the shared affine reference.**  
   IR10 is available as the source for the primary IR10-to-RED4 task and is
   pixel-aligned with RED4. Applying the same IR10-derived affine transform to
   both bands preserves the physical RED4-minus-IR10 offset.

3. **Do not normalize IR10 and RED4 independently.**  
   Independent per-band normalization would force both bands into similar
   artificial distributions and remove the systematic spectral difference
   that the translation model is intended to learn.

4. **Use a scale floor.**  
   The raw standard deviations show that many patches are extremely
   low-contrast. A lower bound on the robust scale prevents pathological
   amplification and keeps the model-space distribution suitable for priors,
   losses, and sampling.

5. **Treat DC subtraction and gain as later model-space calibration steps.**  
   The raw analysis establishes why an affine normalization is needed. Later
   stages should separately quantify how DC subtraction centers the IR10
   model-space prior and how `norm_gain` adjusts the clean image scale relative
   to Gaussian noise in Flow Matching.

## Caveats

- A few sets have zero or near-zero raw standard deviation. Ratios such as
  `std_red_over_ir` can therefore be numerically extreme and should not be
  summarized by the arithmetic mean. Percentiles and medians are more
  meaningful for that ratio.
- Pixel histograms use fixed bins from the CSV-level raw min/max range. The
  histogram is appropriate for visualization and distribution comparison, but
  exact percentile values should be taken from `raw_per_set_stats.csv` or
  `raw_summary.json`.
- This folder currently documents the raw, pre-normalization distribution only.
  Subsequent normalization stages should be documented in separate sibling
  folders or sections to avoid mixing raw physical-space results with
  model-space results.
