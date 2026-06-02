#!/bin/bash
# =============================================================================
#  LUMI Supercomputer — SLURM Batch Script (GPU / ROCm)
#  Job: Model Comparison Evaluation
#  Author: Vahid Moeinifar (AGH University of Science and Technology)
# =============================================================================
#SBATCH --job-name=compare
#SBATCH --account=project_465002989
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=6_output.txt
#SBATCH --error=6_error.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.moeinifar@agh.edu.pl

set -euo pipefail
mkdir -p result

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
# 3. Model checkpoint paths
# -----------------------------------------------------------------------------
SAVED="${WORK_DIR}/saved_models"

ZT_CKPT="${SAVED}/zealot_transformer.pt"
SPEC_LOW_CKPT="${SAVED}/specialist_low_z2.pt"
SPEC_HIGH_CKPT=""                                  # not used
GLOBAL_GAT_CKPT="${SAVED}/Global-GAT.pt"
SPECTRAL_LSTM_CKPT="${SAVED}/SpectralLSTM.pt"
PA_LSTM_CKPT="${SAVED}/pa-lstm.pt"
MLP_DESC_CKPT=""                                   # not used 

# -----------------------------------------------------------------------------
# 4. Sanity check
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
echo "[check] Model files:"
for CKPT in \
    "${ZT_CKPT}" \
    "${SPEC_LOW_CKPT}" \
    "${GLOBAL_GAT_CKPT}" \
    "${SPECTRAL_LSTM_CKPT}" \
    "${PA_LSTM_CKPT}"; do
    if [ -z "${CKPT}" ]; then
        echo "  SKIPPED (empty path)"
    elif [ -f "${CKPT}" ]; then
        echo "  FOUND   ${CKPT}  ($(du -sh "${CKPT}" | cut -f1))"
    else
        echo "  MISSING ${CKPT}"
    fi
done

if [ ! -f "${ZT_CKPT}" ]; then
    echo ""
    echo "ERROR: ZealotTransformer checkpoint not found at ${ZT_CKPT}"
    exit 1
fi

# -----------------------------------------------------------------------------
# 5. Run comparison
# -----------------------------------------------------------------------------
echo ""
echo "[run] $(date)"
echo "------------------------------------------------------"

python - << PYEOF
import sys
sys.path.insert(0, '${PYG_PKGS}')
import os, runpy

argv = [
    '6_compare_models.py',
    '--zt_checkpoint',  '${ZT_CKPT}',
    '--out_dir',        '${WORK_DIR}/result',
    '--n',              '1024',
    '--mc_runs',        '128',
    '--val_graphs',     '10',
    '--seed',           '42',
    '--eval_sizes',
]

optional_flags = [
    ('--spec_low_checkpoint',      '${SPEC_LOW_CKPT}'),
    ('--spec_high_checkpoint',     '${SPEC_HIGH_CKPT}'),
    ('--global_gat_checkpoint',    '${GLOBAL_GAT_CKPT}'),
    ('--spectral_lstm_checkpoint', '${SPECTRAL_LSTM_CKPT}'),
    ('--pa_lstm_checkpoint',       '${PA_LSTM_CKPT}'),
    ('--mlp_desc_checkpoint',      '${MLP_DESC_CKPT}'),
]
for flag, path in optional_flags:
    if path and os.path.isfile(path):
        argv += [flag, path]
    else:
        reason = "empty path" if not path else f"not found: {path}"
        print(f"  [argv] skipping {flag} ({reason})")

sys.argv = argv
print(f"  [argv] {' '.join(argv)}\n")

runpy.run_path('${WORK_DIR}/6_compare_models.py', run_name='__main__')
PYEOF

EXIT_CODE=$?
echo ""
echo "======================================================"
echo " Finished: $(date) | Exit: ${EXIT_CODE}"
echo " Results:"
ls -lh result/ 2>/dev/null || echo "  no files"
echo "======================================================"
exit ${EXIT_CODE}