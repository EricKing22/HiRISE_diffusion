#!/bin/bash

#SBATCH --job-name=train_HiRISE_ed425
#SBATCH --output=/scratch_root/ed425/HiRISE_diffusion/scripts/logs/train_%j.log

#SBATCH --partition=root

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=01:00:00

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
PROJECT_ROOT=/scratch_root/ed425/HiRISE/HiRISE_diffusion
DATA_ROOT=/scratch_root/ed425/HiRISE/
CSV_PATH=/scratch_root/ed425/HiRISE/files/data_record_bin12.csv
CKPT_DIR=/scratch_root/ed425/HiRISE_diffusion/src/models/

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs
mkdir -p $CKPT_DIR

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"
echo "Checkpoint   : $CKPT_DIR"
echo ""

python src/train.py \
    --data_root     $DATA_ROOT        \
    --csv_path      $CSV_PATH         \
    --ckpt_dir      $CKPT_DIR         \
    --wandb_project HiRISE_diffusion  \
    --run_name      "slurm_${SLURM_JOB_ID}"

end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
