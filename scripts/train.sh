#!/bin/bash

#SBATCH --job-name=train_HiRISE_ed425
#SBATCH --output=/scratch_root/ed425/HiRISE_diffusion/scripts/logs/train_%j.log

#SBATCH --partition=root

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=05:00:00

export WANDB_API_KEY=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
export WANDB_DIR=/scratch_root/ed425/tmp
export NETRC=/scratch_root/ed425/.netrc
export MPLCONFIGDIR=/scratch_root/ed425/tmp/matplotlib
touch /scratch_root/ed425/.netrc
chmod 600 /scratch_root/ed425/.netrc

export WANDB_DIR=/scratch_root/ed425/tmp
export WANDB_CACHE_DIR=/scratch_root/ed425/tmp/wandb_cache
export WANDB_CONFIG_DIR=/scratch_root/ed425/tmp/wandb_config
export WANDB_DATA_DIR=/scratch_root/ed425/tmp/wandb_data
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR $WANDB_DATA_DIR

source /scratch_root/ed425/miniconda3/etc/profile.d/conda.sh
conda activate HiPredict

start_time=$(date +%s)
echo -e "Job started on $(date)\n"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/scratch_root/ed425/HiRISE_diffusion
DATA_ROOT=/scratch_root/ed425/HiRISE/
CSV_PATH=/scratch_root/ed425/HiRISE/files/data_record_bin12.csv
CKPT_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs
mkdir -p $CKPT_DIR

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"

# Training mode switch:
#   bidirectional (default) | ir2red | red2ir
TRAIN_MODE=${1:-bidirectional}
case "$TRAIN_MODE" in
    bidirectional|ir2red|red2ir) ;;
    *)
        echo "Invalid TRAIN_MODE: $TRAIN_MODE"
        echo "Usage: sbatch train.sh [bidirectional|ir2red|red2ir] [sobel|dexined]"
        exit 1
        ;;
esac

# Edge detection mode switch:
#   sobel (default) | dexined
EDGE_MODE=${2:-dexined}
case "$EDGE_MODE" in
    sobel|dexined) ;;
    *)
        echo "Invalid EDGE_MODE: $EDGE_MODE"
        echo "Usage: sbatch train.sh [bidirectional|ir2red|red2ir] [sobel|dexined] [dc|no_dc]"
        exit 1
        ;;
esac

# DC offset mode (Method B): dc (default) | no_dc
DC_MODE=${3:-dc}
case "$DC_MODE" in
    dc|no_dc) ;;
    *)
        echo "Invalid DC_MODE: $DC_MODE"
        echo "Usage: sbatch train.sh [bidirectional|ir2red|red2ir] [sobel|dexined] [dc|no_dc]"
        exit 1
        ;;
esac
DC_FLAG=""
[ "$DC_MODE" = "no_dc" ] && DC_FLAG="--no_dc"

LATEST_CKPT=${CKPT_DIR}/latest_${TRAIN_MODE}.pt
RUN_CKPT_PATTERN=${CKPT_DIR}/${TRAIN_MODE}_<mmdd_HHMM>/step_XXXXXXX.pt

echo "Checkpoint dir          : $CKPT_DIR"
echo "Latest checkpoint file  : $LATEST_CKPT"
echo "Step checkpoint pattern : $RUN_CKPT_PATTERN"
echo "Train mode   : $TRAIN_MODE"
echo "Edge mode    : $EDGE_MODE"
echo "DC mode      : $DC_MODE"
echo ""

if [ "$TRAIN_MODE" = "bidirectional" ]; then
    wandb_project=HiRISE_diffusion
elif [ "$TRAIN_MODE" = "ir2red" ]; then
    wandb_project=HiRISE_diffusion_ir2red
else
    wandb_project=HiRISE_diffusion_red2ir
fi

python src/train.py \
    --data_root     $DATA_ROOT        \
    --csv_path      $CSV_PATH         \
    --ckpt_dir      $CKPT_DIR         \
    --wandb_project $wandb_project   \
    --run_name      "${TRAIN_MODE}_${EDGE_MODE}_${DC_MODE}_slurm_${SLURM_JOB_ID}" \
    --train_mode    $TRAIN_MODE       \
    --edge_mode     $EDGE_MODE        \
    $DC_FLAG

end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
