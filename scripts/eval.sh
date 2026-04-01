#!/bin/bash

#SBATCH --job-name=eval_HiRISE_ed425
#SBATCH --output=/scratch_root/ed425/HiRISE_diffusion/scripts/logs/eval_%j.log

#SBATCH --partition=root

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=06:00:00

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

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"
echo "Checkpoint   : $CHECKPOINT"
echo "Prior dir    : $PRIOR_DIR"
echo ""

# Metrics: MSE / MAE / PSNR / SSIM / Pearson-r  (FID disabled; remove --no_fid to enable)
python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   0.0          \
    --lambda_ccl   0.0          \
    --no_fid                     \
    --device       cuda

python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   10.0          \
    --lambda_ccl   10.0          \
    --no_fid                     \
    --device       cuda


python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   20.0          \
    --lambda_ccl   20.0          \
    --no_fid                     \
    --device       cuda


python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   30.0          \
    --lambda_ccl   30.0          \
    --no_fid                     \
    --device       cuda

python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   40.0          \
    --lambda_ccl   40.0          \
    --no_fid                     \
    --device       cuda

python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   50.0          \
    --lambda_ccl   50.0          \
    --no_fid                     \
    --device       cuda


python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   60.0          \
    --lambda_ccl   60.0          \
    --no_fid                     \
    --device       cuda

python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   70.0          \
    --lambda_ccl   70.0          \
    --no_fid                     \
    --device       cuda

python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   80.0          \
    --lambda_ccl   80.0          \
    --no_fid                     \
    --device       cuda


python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   90.0          \
    --lambda_ccl   90.0          \
    --no_fid                     \
    --device       cuda


python src/eval.py \
    --checkpoint   $CHECKPOINT   \
    --prior_dir    $PRIOR_DIR    \
    --data_root    $DATA_ROOT    \
    --csv_path     $CSV_PATH     \
    --max_samples  100           \
    --batch_size   8             \
    --lambda_scl   100.0          \
    --lambda_ccl   100.0          \
    --no_fid                     \
    --device       cuda
end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
