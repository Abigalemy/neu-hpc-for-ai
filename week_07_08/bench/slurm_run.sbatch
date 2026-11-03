#!/usr/bin/env bash
# Submit with:  sbatch bench/slurm_run.sbatch
# >>> Fill in the partition / account / time for YOUR cluster (check `sinfo` and your course instructions). <<<
#SBATCH --job-name=qwen3-bench
#SBATCH --partition=CHANGE_ME_gpu_partition
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
## #SBATCH --account=CHANGE_ME_if_required

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from week_07_08/
mkdir -p logs results
# module load cuda/12.x        # if your cluster needs it
source "${VENV:-$HOME/venvs/week07/bin/activate}"
export HF_HOME=${HF_HOME:-$HOME/hf_cache}   # pre-download the model & tokenizer on a login node if compute nodes have no internet
nvidia-smi
bench/run_experiments.sh "${MODEL:-Qwen/Qwen3-4B}" "${KV_GB:-8}"
