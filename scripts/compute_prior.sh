#!/bin/bash

#SBATCH --job-name=prior_HiRISE_ed425
#SBATCH --output=/scratch_root/ed425/HiRISE_diffusion/scripts/logs/prior_%j.log

#SBATCH --partition=root

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

#SBATCH --mem=16G
#SBATCH --time=01:00:00

source /scratch_root/ed425/miniconda3/etc/profile.d/conda.sh
conda activate HiPredict

start_time=$(date +%s)
echo -e "Job started on $(date)\n"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/scratch_root/ed425/HiRISE/HiRISE_diffusion
DATA_ROOT=/scratch_root/ed425/HiRISE/
CSV_PATH=/scratch_root/ed425/HiRISE/files/data_record_bin12.csv
PRIOR_DIR=/scratch_root/ed425/HiRISE_diffusion/src/output

mkdir -p /scratch_root/ed425/HiRISE_diffusion/scripts/logs
mkdir -p $PRIOR_DIR

cd $PROJECT_ROOT

echo "Project root : $PROJECT_ROOT"
echo "Data root    : $DATA_ROOT"
echo "CSV path     : $CSV_PATH"
echo "Prior dir    : $PRIOR_DIR"
echo ""

python src/compute_prior.py \
    --data_root $DATA_ROOT  \
    --csv_path  $CSV_PATH   \
    --prior_dir $PRIOR_DIR  \
    --bins      256         \
    --device    cpu

end_time=$(date +%s)

echo -e "\nJob finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))

echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
