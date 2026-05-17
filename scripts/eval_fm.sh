#!/bin/bash

#SBATCH --job-name=eval_fm_HiRISE_ed425
#SBATCH --output=/scratch_root/ed425/HiRISE_diffusion/scripts/logs/eval_fm_%j.log

#SBATCH --partition=root

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /scratch_root/ed425/miniconda3/etc/profile.d/conda.sh
conda activate HiPredict

start_time=$(date +%s)
echo -e "Job started on $(date)\n"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/scratch_root/ed425/HiRISE_diffusion
DATA_ROOT=/scratch_root/as5023/HiRISE/data/
CSV_PATH=/scratch_root/as5023/HiRISE/data/data_record_bin12.csv
CKPT_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output
PRIOR_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"

# Evaluation mode: bidirectional | ir2red | red2ir
# Usage: sbatch eval_fm.sh [bidirectional|ir2red|red2ir] [num_steps] [lambda_sgi_scl] [lambda_sgi_ccl] [dc|no_dc] [velocity|reproject] [norm_gain] [diag|no_diag] [ratio|raw] [guidance_ratio]
TRAIN_MODE=${1:-ir2red}
NUM_STEPS=${2:-50}
LAMBDA_SGI_SCL=${3:-0.0}
LAMBDA_SGI_CCL=${4:-0.0}
DC_MODE=${5:-dc}
SGI_MODE=${6:-velocity}
NORM_GAIN=${7:-4.0}
DIAG_MODE=${8:-no_diag}
SGI_SCALE_MODE=${9:-ratio}
SGI_GUIDANCE_RATIO=${10:-0.01}

case "$TRAIN_MODE" in
    bidirectional|ir2red|red2ir) ;;
    *)
        echo "Invalid TRAIN_MODE: $TRAIN_MODE"
        echo "Usage: sbatch eval_fm.sh [bidirectional|ir2red|red2ir] [num_steps] [lambda_sgi_scl] [lambda_sgi_ccl] [dc|no_dc] [velocity|reproject] [norm_gain]"
        exit 1
        ;;
esac

case "$DC_MODE" in
    dc)    DC_ARGS=() ;;
    no_dc) DC_ARGS=(--no_dc) ;;
    *)
        echo "Invalid DC_MODE: $DC_MODE"
        echo "Usage: sbatch eval_fm.sh [bidirectional|ir2red|red2ir] [num_steps] [lambda_sgi_scl] [lambda_sgi_ccl] [dc|no_dc] [velocity|reproject] [norm_gain]"
        exit 1
        ;;
esac

case "$SGI_MODE" in
    velocity|reproject) ;;
    *)
        echo "Invalid SGI_MODE: $SGI_MODE"
        echo "Usage: sbatch eval_fm.sh [bidirectional|ir2red|red2ir] [num_steps] [lambda_sgi_scl] [lambda_sgi_ccl] [dc|no_dc] [velocity|reproject] [norm_gain]"
        exit 1
        ;;
esac

case "$DIAG_MODE" in
    diag)    DIAG_ARGS=(--sgi_diagnostics) ;;
    no_diag) DIAG_ARGS=() ;;
    *)
        echo "Invalid DIAG_MODE: $DIAG_MODE"
        echo "Usage: sbatch eval_fm.sh [bidirectional|ir2red|red2ir] [num_steps] [lambda_sgi_scl] [lambda_sgi_ccl] [dc|no_dc] [velocity|reproject] [norm_gain] [diag|no_diag]"
        exit 1
        ;;
esac

case "$SGI_SCALE_MODE" in
    ratio|raw) ;;
    *)
        echo "Invalid SGI_SCALE_MODE: $SGI_SCALE_MODE"
        echo "Usage: sbatch eval_fm.sh [bidirectional|ir2red|red2ir] [num_steps] [lambda_sgi_scl] [lambda_sgi_ccl] [dc|no_dc] [velocity|reproject] [norm_gain] [diag|no_diag] [ratio|raw] [guidance_ratio]"
        exit 1
        ;;
esac

CHECKPOINT=${CKPT_DIR}/latest_fm_${TRAIN_MODE}.pt

python src/eval_fm.py \
    --checkpoint  $CHECKPOINT  \
    --prior_dir   $PRIOR_DIR   \
    --data_root   $DATA_ROOT   \
    --csv_path    $CSV_PATH    \
    --train_mode  $TRAIN_MODE  \
    --num_steps   $NUM_STEPS   \
    --lambda_sgi_scl $LAMBDA_SGI_SCL \
    --lambda_sgi_ccl $LAMBDA_SGI_CCL \
    --sgi_mode   $SGI_MODE     \
    --sgi_scale_mode $SGI_SCALE_MODE \
    --sgi_guidance_ratio $SGI_GUIDANCE_RATIO \
    --norm_gain  $NORM_GAIN    \
    "${DC_ARGS[@]}"        \
    "${DIAG_ARGS[@]}"      \
    --batch_size  16           \
    --no_fid

end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
