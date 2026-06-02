#!/bin/bash
# =============================================================================
#  LUMI Supercomputer — SLURM Batch Script (GPU / ROCm)
#  Job: Universal Magnetization Trajectory Predictor — LSTM (Single GCD)
#  Author: Vahid Moeinifar (AGH University of Science and Technology)
# =============================================================================

#SBATCH --job-name=place-aware-lstm
#SBATCH --account=project_465002915
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=0-output-pa-lstm.txt
#SBATCH --error=0-error-pa-lstm.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.moeinifar@agh.edu.pl

set -euo pipefail

echo "======================================================"
echo " Job:     ${SLURM_JOB_NAME}  (${SLURM_JOB_ID})"
echo " Node:    ${SLURM_NODELIST}"
echo " Started: $(date)"
echo "======================================================"

# -----------------------------------------------------------------------------
# 1. Modules
# -----------------------------------------------------------------------------
module --force purge
module load LUMI/25.03
module load partition/G
module load lumi-container-wrapper/0.4.2-cray-python-3.11.7

echo "[modules] Loaded:"
module list 2>&1 || true

# -----------------------------------------------------------------------------
# 2. Paths
# -----------------------------------------------------------------------------
WORK_DIR="/scratch/project_465002915/GNN_Project"
WRAPPER_DIR="${WORK_DIR}/pyg-wrapper"
PYG_PKGS="${WORK_DIR}/pyg-packages"

export PATH="${WRAPPER_DIR}/bin:${PATH}"
export SINGULARITYENV_PYTHONPATH="${PYG_PKGS}"
export PYTHONPATH="${PYG_PKGS}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

cd "${WORK_DIR}"
mkdir -p saved_models

# MIOpen cache in /tmp (not on Lustre — avoids metadata storms)
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Expose only 1 GCD
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

# All 56 CPUs for dataset build (spectral gap eigsh + MC simulation)
export OMP_NUM_THREADS=56
export SLURM_CPUS_PER_TASK=56    # picked up by the script's NUM_CPUS
export TMPDIR=/tmp

# -----------------------------------------------------------------------------
# 3. Sanity check
# -----------------------------------------------------------------------------
echo ""
echo "[check] Environment:"
python -c "
import sys
sys.path.insert(0, '${PYG_PKGS}')
import torch
print(f'torch:          {torch.__version__}')
n = torch.cuda.device_count()
print(f'GPUs visible:   {n}')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}  VRAM={p.total_memory/1e9:.1f}GB')
if n != 1:
    raise SystemExit(f'Expected 1 GPU, got {n}. Check ROCR_VISIBLE_DEVICES.')
print('OK — single GPU run')
"

# -----------------------------------------------------------------------------
# 4. Train
# -----------------------------------------------------------------------------
echo ""
echo "[run] $(date)"
echo "------------------------------------------------------"

python - << PYEOF
import sys
sys.path.insert(0, '${PYG_PKGS}')
import runpy
sys.argv = [
    '0-pa-lstm.py',
    '--batch_size',   '256',
    '--epochs',       '300',
    '--lr',           '1e-3',
    '--weight_decay', '1e-4',
    '--num_graphs',   '200',
    '--mc_runs',      '40',
    '--T',            '50',
    '--hidden_dim',   '256',
    '--num_layers',   '2',
    '--num_workers',  '6',
    '--dropout',      '0.2',
    '--save_dir',     'saved_models',
    '--save_name',    'pa-lstm.pt',
]
runpy.run_path('${WORK_DIR}/0-pa-lstm.py', run_name='__main__')
PYEOF

EXIT_CODE=$?

echo ""
echo "======================================================"
echo " Finished: $(date) | Exit: ${EXIT_CODE}"
echo " Saved models:"
ls -lh saved_models/ 2>/dev/null || echo "  (no files)"
echo "======================================================"
exit ${EXIT_CODE}