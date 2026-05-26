#!/bin/bash
# =============================================================================
#  LUMI Supercomputer — SLURM Batch: SMALL NETWORKS (N ≤ 2048)
#  Script: eval_small_graphs.py
#  Tables 1, 2, 3, Suppl A, B
#  MC runs: 128 (full precision — main paper tables)
# =============================================================================
#SBATCH --job-name=eval_small
#SBATCH --account=project_465002989
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=6_eval_small_output.txt
#SBATCH --error=6_eval_small_error.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.moeinifar@agh.edu.pl

set -euo pipefail

echo "======================================================"
echo " Job:     ${SLURM_JOB_NAME}  (${SLURM_JOB_ID})"
echo " Node:    ${SLURM_NODELIST}"
echo " Script:  eval_small_graphs.py — N ≤ 2048"
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
WORK_DIR="/scratch/project_465002989/GNN_Project"
WRAPPER_DIR="${WORK_DIR}/pyg-wrapper"
PYG_PKGS="${WORK_DIR}/pyg-packages"

export PATH="${WRAPPER_DIR}/bin:${PATH}"
export SINGULARITYENV_PYTHONPATH="${PYG_PKGS}"
export PYTHONPATH="${PYG_PKGS}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=0
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TMPDIR=/tmp

cd "${WORK_DIR}"

# -----------------------------------------------------------------------------
# 3. Checkpoint paths
# -----------------------------------------------------------------------------
SAVED="${WORK_DIR}/saved_models"

ZT_CKPT="${SAVED}/zealot_transformer.pt"
SPECTRAL_LSTM_CKPT="${SAVED}/SpectralLSTM.pt"
PA_LSTM_CKPT="${SAVED}/pa-lstm.pt"
SPEC_LOW_CKPT="${SAVED}/specialist_low_z2.pt"
GLOBAL_GAT_CKPT="${SAVED}/Global-GAT.pt"

OUT_DIR="${WORK_DIR}/result"
mkdir -p "${OUT_DIR}"

# -----------------------------------------------------------------------------
# 4. Sanity checks
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

echo ""
echo "[check] Script file:"
if [ -f "${WORK_DIR}/6_all_results_tr.py" ]; then
    echo "  FOUND   ${WORK_DIR}/6_all_results_tr.py"
else
    echo "  MISSING ${WORK_DIR}/6_all_results_tr.py"
    exit 1
fi

echo ""
echo "[check] Checkpoint files:"
ALL_FOUND=true
for CKPT in \
    "${ZT_CKPT}" \
    "${SPECTRAL_LSTM_CKPT}" \
    "${PA_LSTM_CKPT}" \
    "${SPEC_LOW_CKPT}" \
    "${GLOBAL_GAT_CKPT}"; do
    if [ -f "${CKPT}" ]; then
        echo "  FOUND   ${CKPT}  ($(du -sh "${CKPT}" | cut -f1))"
    else
        echo "  MISSING ${CKPT}"
        ALL_FOUND=false
    fi
done

if [ ! -f "${ZT_CKPT}" ]; then
    echo ""
    echo "ERROR: ZealotTransformer checkpoint not found — cannot continue."
    exit 1
fi
if [ "${ALL_FOUND}" = false ]; then
    echo ""
    echo "WARNING: Some checkpoints missing — those models will be skipped."
fi

# -----------------------------------------------------------------------------
# 5. Run evaluation
# -----------------------------------------------------------------------------
echo ""
echo "[run] $(date)"
echo "------------------------------------------------------"
echo "Target: Tables 1, 2, 3, Suppl A, B (N ≤ 2048)"
echo ""

python - << PYEOF
import sys
sys.path.insert(0, '${PYG_PKGS}')
import os, runpy

sys.argv = [
    '6_all_results_tr.py',
    '--zt_checkpoint',              '${ZT_CKPT}',
    '--spectral_lstm_checkpoint',   '${SPECTRAL_LSTM_CKPT}',
    '--pa_lstm_checkpoint',         '${PA_LSTM_CKPT}',
    '--spec_low_checkpoint',        '${SPEC_LOW_CKPT}',
    '--global_gat_checkpoint',      '${GLOBAL_GAT_CKPT}',
    '--out_dir',                    '${OUT_DIR}',
    '--workers',                    '16',
    '--seed',                       '42',
    '--mc_runs',                    '128',
    '--val_graphs',                 '10',
]

print(f"  [argv] {' '.join(sys.argv)}\n")
runpy.run_path('${WORK_DIR}/6_all_results_tr.py', run_name='__main__')
PYEOF

EXIT_CODE=$?
echo ""
echo "======================================================"
echo " Finished: $(date) | Exit: ${EXIT_CODE}"
echo ""
echo " Results:"
ls -lh "${OUT_DIR}/"            2>/dev/null || echo "  (no root files)"
ls -lh "${OUT_DIR}/tables/"     2>/dev/null || echo "  (no tables)"
echo "======================================================"
exit ${EXIT_CODE}