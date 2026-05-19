#!/bin/bash
# =============================================================================
#  LUMI Supercomputer — SLURM Batch Script (GPU / ROCm)
#  Job: Specialist GAT — trains Specialist-Low (Z=2) then Specialist-High (Z=32)
#  Author: Vahid Moeinifar (AGH University of Science and Technology)
#
#  Both specialists trained sequentially in one job.
#  Reuses pyg-wrapper + pyg-packages from the FiLM/Global-GAT setup.
#  Outputs: saved_models/specialist_low_z2.pt
#           saved_models/specialist_high_z32.pt
# =============================================================================

#SBATCH --job-name=specialist_gat
#SBATCH --account=project_465002915
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=1-specialist-output.txt
#SBATCH --error=1-specialist-error.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.moeinifar@agh.edu.pl

set -euo pipefail

echo "======================================================"
echo " Job:     ${SLURM_JOB_NAME}  (${SLURM_JOB_ID})"
echo " Node:    ${SLURM_NODELIST}"
echo " Started: $(date)"
echo "======================================================"

# -----------------------------------------------------------------------------
# 1. Modules — identical to Global-GAT job
# -----------------------------------------------------------------------------
module --force purge
module load LUMI/25.03
module load partition/G
module load lumi-container-wrapper/0.4.2-cray-python-3.11.7

echo "[modules] Loaded:"
module list 2>&1 || true

# -----------------------------------------------------------------------------
# 2. Paths — reuse everything from FiLM/Global-GAT setup
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

# MIOpen cache in /tmp
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"

export SINGULARITYENV_ROCR_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TMPDIR=/tmp

# -----------------------------------------------------------------------------
# 3. Sanity check
# -----------------------------------------------------------------------------
echo ""
echo "[check] Environment:"
python -c "
import sys
sys.path.insert(0, '${PYG_PKGS}')
import torch, torch_geometric
print(f'torch:           {torch.__version__}')
print(f'torch_geometric: {torch_geometric.__version__}')
print(f'GPU available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU:             {p.name}  VRAM={p.total_memory/1e9:.1f}GB')
"

# -----------------------------------------------------------------------------
# 4. Train Specialist-Low (Z=2) then Specialist-High (Z=32)
# -----------------------------------------------------------------------------
echo ""
echo "[run] $(date)"
echo "------------------------------------------------------"

python - << PYEOF
import sys
sys.path.insert(0, '${PYG_PKGS}')
import runpy
sys.argv = [
    '1-local-trainer.py',
    '--batch_size',   '256',
    '--epochs',       '300',
    '--lr',           '1e-3',
    '--weight_decay', '1e-4',
    '--num_graphs',   '200',
    '--mc_steps',     '40',
    '--num_workers',  '6',
    '--hidden_dim',   '256',
    '--dropout',      '0.1',
    '--save_dir',     'saved_models',
]
runpy.run_path('${WORK_DIR}/1-local-trainer.py', run_name='__main__')
PYEOF

EXIT_CODE=$?

echo ""
echo "======================================================"
echo " Finished: $(date) | Exit: ${EXIT_CODE}"
echo " Saved models:"
ls -lh saved_models/ 2>/dev/null || echo "  (no files)"
echo "======================================================"
exit ${EXIT_CODE}