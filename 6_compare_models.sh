#!/bin/bash
# =============================================================================
#  LUMI Supercomputer — SLURM Batch Script (GPU / ROCm)
#  Job: Model Comparison Evaluation
#  Author: Vahid Moeinifar (AGH University of Science and Technology)
#
#  Uses the same pyg-wrapper + pyg-packages setup from training.
#  No re-installation needed.
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
# 1. Modules — same as training job
# -----------------------------------------------------------------------------
module --force purge
module load LUMI/25.03
module load partition/G
module load lumi-container-wrapper/0.4.2-cray-python-3.11.7
echo "[modules] Loaded:"
module list 2>&1 || true
# -----------------------------------------------------------------------------
# 2. Paths — reuse everything built during training setup
# -----------------------------------------------------------------------------
WORK_DIR="/scratch/project_465002989/GNN_Project"
WRAPPER_DIR="${WORK_DIR}/pyg-wrapper"
PYG_PKGS="${WORK_DIR}/pyg-packages"
export PATH="${WRAPPER_DIR}/bin:${PATH}"
export SINGULARITYENV_PYTHONPATH="${PYG_PKGS}"
export PYTHONPATH="${PYG_PKGS}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
# MIOpen cache in /tmp
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
#    Update these if your filenames differ.  All paths except ZT are optional:
#    the comparison script skips any model whose checkpoint is not found.
# -----------------------------------------------------------------------------
SAVED="${WORK_DIR}/saved_models"

ZT_CKPT="${SAVED}/zealot_transformer.pt"          # ZealotTransformer (required)
SPEC_LOW_CKPT="${SAVED}/specialist_low_z2.pt"         # Specialist-Low  (GAT, Z=2)
#SPEC_HIGH_CKPT="${SAVED}/specialist_high.pt"       # Specialist-High (GAT, Z=32)
GLOBAL_GAT_CKPT="${SAVED}/Global-GAT.pt"           # Global-GAT      (all Z)
SPECTRAL_LSTM_CKPT="${SAVED}/SpectralLSTM.pt"     # SpectralLSTM
PA_LSTM_CKPT="${SAVED}/pa-lstm.pt"                 # PA-LSTM
MLP_DESC_CKPT="${SAVED}/mlp_descriptor.pt"         # MLP-Descriptor  (optional)

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
for CKPT in "${ZT_CKPT}" \
            "${SPEC_LOW_CKPT}" "${SPEC_HIGH_CKPT}" "${GLOBAL_GAT_CKPT}" \
            "${SPECTRAL_LSTM_CKPT}" "${PA_LSTM_CKPT}" "${MLP_DESC_CKPT}"; do
    if [ -f "${CKPT}" ]; then
        echo "  FOUND   ${CKPT}  ($(du -sh "${CKPT}" | cut -f1))"
    else
        echo "  MISSING ${CKPT}"
    fi
done

# Abort early if the required ZealotTransformer checkpoint is missing
if [ ! -f "${ZT_CKPT}" ]; then
    echo ""
    echo "ERROR: ZealotTransformer checkpoint not found at ${ZT_CKPT}"
    echo "       This checkpoint is required.  Aborting."
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

# Build argv for compare_models.py's argparse
# Optional checkpoints are included only when the file actually exists,
# so the script's own "skipping" logic handles any gaps gracefully.

argv = [
    '4-compare_models.py',
    '--zt_checkpoint',         '${ZT_CKPT}',
    '--out_dir',               '${WORK_DIR}/result',
    '--n',                     '1024',
    '--mc_runs',               '128',
    '--val_graphs',            '10',
    '--seed',                  '42',
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
    if os.path.isfile(path):
        argv += [flag, path]
    else:
        print(f"  [argv] skipping {flag} (file not found: {path})")

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