#!/bin/bash

#SBATCH --job-name=evalFM_HiRISE_ed425
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
DATA_ROOT=/scratch_root/ed425/HiRISE/
CSV_PATH=/scratch_root/ed425/HiRISE/files/data_record_bin12.csv
CKPT_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"

# Evaluation mode: ir2red | red2ir
# Usage: sbatch eval_FM.sh [ir2red|red2ir] [num_steps]
TRAIN_MODE=${1:-ir2red}
NUM_STEPS=${2:-50}

case "$TRAIN_MODE" in
    ir2red|red2ir|bidirectional) ;;
    *)
        echo "Invalid TRAIN_MODE: $TRAIN_MODE"
        echo "Usage: sbatch eval_FM.sh [ir2red|red2ir|bidirectional] [num_steps]"
        exit 1
        ;;
esac

CHECKPOINT=${CKPT_DIR}/latest_fm_${TRAIN_MODE}.pt

echo "Checkpoint   : $CHECKPOINT"
echo "Train mode   : $TRAIN_MODE"
echo "ODE steps    : $NUM_STEPS"
echo ""

python src/eval_fm.py \
    --checkpoint  $CHECKPOINT  \
    --data_root   $DATA_ROOT   \
    --csv_path    $CSV_PATH    \
    --train_mode  $TRAIN_MODE  \
    --num_steps   $NUM_STEPS   \
    --batch_size  4            \
    --no_fid

end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
