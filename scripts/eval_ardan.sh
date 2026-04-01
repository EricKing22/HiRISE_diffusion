#!/bin/bash

#SBATCH --job-name=eval_ardan_HiRISE_ed425
#SBATCH --output=/scratch_root/ed425/HiRISE_diffusion/scripts/logs/eval_ardan_%j.log

#SBATCH --partition=root

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /scratch_root/ed425/miniconda3/etc/profile.d/conda.sh
conda activate HiPredict

start_time=$(date +%s)
echo -e "Job started on $(date)\n"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/scratch_root/ed425/HiRISE_diffusion
DATA_ROOT=/scratch_root/ed425/HiRISE/
CSV_PATH=/scratch_root/ed425/HiRISE/files/data_record_bin12.csv
CHECKPOINT=/scratch_root/ed425/HiRISE_diffusion/src/output/latest.pt
PRIOR_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output
EVALS_CONFIG=/scratch_root/ed425/HiRISE_diffusion/src/configs/evals_ardan.yaml
SAVE_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output/evals

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs
mkdir -p $SAVE_DIR

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"
echo "Checkpoint   : $CHECKPOINT"
echo "Prior dir    : $PRIOR_DIR"
echo "Evals config : $EVALS_CONFIG"
echo "Save dir     : $SAVE_DIR"
echo ""

# ── Step 0: Generate evaluation categories (run once, skip if YAML exists) ───
if [ ! -f "$EVALS_CONFIG" ]; then
    echo "Generating evaluation categories..."
    python src/configs/gen_eval_categories_ardan.py \
        --csv_path  $CSV_PATH  \
        --output    $EVALS_CONFIG
    echo ""
fi

# ── Step 1: Global eval, no SCI (baseline) ────────────────────────────────────
python src/eval_ardan.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   0.0           \
    --lambda_ccl   0.0           \
    --no_fid                     \
    --save_dir     $SAVE_DIR     \
    --device       cuda

# ── Step 2: SCI λ sweep ───────────────────────────────────────────────────────
for LAMBDA in 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0; do
    python src/eval_ardan.py \
        --checkpoint   $CHECKPOINT   \
        --prior_dir    $PRIOR_DIR    \
        --data_root    $DATA_ROOT    \
        --csv_path     $CSV_PATH     \
        --max_samples  100           \
        --batch_size   8             \
        --lambda_scl   $LAMBDA       \
        --lambda_ccl   $LAMBDA       \
        --no_fid                     \
        --save_dir     $SAVE_DIR     \
        --device       cuda
done

# ── Step 3: Best λ with per-category evaluation ────────────────────────────────
# (adjust BEST_LAMBDA based on Step 2 sweep results)
BEST_LAMBDA=20.0
python src/eval_ardan.py \
    --checkpoint    $CHECKPOINT    \
    --prior_dir     $PRIOR_DIR     \
    --data_root     $DATA_ROOT     \
    --csv_path      $CSV_PATH      \
    --max_samples   0              \
    --batch_size    4              \
    --lambda_scl    $BEST_LAMBDA   \
    --lambda_ccl    $BEST_LAMBDA   \
    --no_fid                       \
    --save_dir      $SAVE_DIR      \
    --category_eval                \
    --evals_config  $EVALS_CONFIG  \
    --device        cuda

end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
