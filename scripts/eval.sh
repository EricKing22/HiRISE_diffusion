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
PRIOR_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output

# Evaluation mode switch:
#   bidirectional (default) | ir2red | red2ir
EVAL_MODE=${1:-bidirectional}
case "$EVAL_MODE" in
    bidirectional|ir2red|red2ir) ;;
    *)
        echo "Invalid EVAL_MODE: $EVAL_MODE"
        echo "Usage: sbatch eval.sh [bidirectional|ir2red|red2ir] [sobel|dexined]"
        exit 1
        ;;
esac

# Edge detection mode switch:
#   sobel (default) | dexined
EDGE_MODE=${2:-sobel}

# DC offset mode (Method B): dc (default) | no_dc
DC_MODE=${3:-dc}
case "$DC_MODE" in
    dc|no_dc) ;;
    *)
        echo "Invalid DC_MODE: $DC_MODE"
        echo "Usage: sbatch eval.sh [bidirectional|ir2red|red2ir] [sobel|dexined] [dc|no_dc]"
        exit 1
        ;;
esac
DC_FLAG=""
[ "$DC_MODE" = "no_dc" ] && DC_FLAG="--no_dc"

if [ "$EVAL_MODE" = "bidirectional" ]; then
    CHECKPOINT=/scratch_root/ed425/HiRISE_diffusion/src/output/latest_bidirectional.pt
else
    CHECKPOINT=/scratch_root/ed425/HiRISE_diffusion/src/output/latest_${EVAL_MODE}.pt
fi

# Backward-compatible fallback for old bidirectional checkpoint naming.
if [ ! -f "$CHECKPOINT" ] && [ "$EVAL_MODE" = "bidirectional" ]; then
    CHECKPOINT=/scratch_root/ed425/HiRISE_diffusion/src/output/latest.pt
fi

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"
echo "Eval mode    : $EVAL_MODE"
echo "Edge mode    : $EDGE_MODE"
echo "DC mode      : $DC_MODE"
echo "Checkpoint   : $CHECKPOINT"
echo "Prior dir    : $PRIOR_DIR"
echo ""

# Metrics: MSE / MAE / PSNR / SSIM / Pearson-r (FID disabled by default)
for LAMBDA in 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0; do
    echo "Running eval: mode=$EVAL_MODE  edge=$EDGE_MODE  lambda=$LAMBDA"
    python src/eval.py \
        --checkpoint   $CHECKPOINT   \
        --train_mode   $EVAL_MODE    \
        --prior_dir    $PRIOR_DIR    \
        --data_root    $DATA_ROOT    \
        --csv_path     $CSV_PATH     \
        --max_samples  100           \
        --batch_size   8             \
        --lambda_scl   $LAMBDA       \
        --lambda_ccl   $LAMBDA       \
        --no_fid                     \
        --device       cuda          \
        --edge_mode    $EDGE_MODE    \
        $DC_FLAG
done
end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
