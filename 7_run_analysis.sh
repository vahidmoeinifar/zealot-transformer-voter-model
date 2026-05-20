#!/bin/bash
# =============================================================================
#  LUMI — Analysis Jobs (GPU)
#  Runs scripts in sequence:
#    7_attention_analysis.py    (ZealotTransformer attention)
#    8_convergence_speed.py     (convergence time vs Z)
#    9_speedup_table.py         (MC vs ZealotTransformer timing)
# =============================================================================

#SBATCH --job-name=analysis
#SBATCH --account=project_465002989
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=7_output.txt
#SBATCH --error=7_error.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.moeinifar@agh.edu.pl

# Stop on error; do NOT use nounset (-u) — optional checkpoint vars may be empty
set -eo pipefail

echo "======================================================"
echo " Job:     ${SLURM_JOB_NAME}  (${SLURM_JOB_ID})"
echo " Node:    ${SLURM_NODELIST}"
echo " Started: $(date)"
echo "======================================================"

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
module --force purge
module load LUMI/25.03
module load partition/G
module load lumi-container-wrapper/0.4.2-cray-python-3.11.7

# -----------------------------------------------------------------------------
# Paths
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
mkdir -p result

# -----------------------------------------------------------------------------
# Checkpoint path (single ZealotTransformer checkpoint used by all scripts)
# -----------------------------------------------------------------------------
SAVED="${WORK_DIR}/saved_models"
ZT_CKPT="${SAVED}/zealot_transformer.pt"

echo ""
echo "[check] Checkpoint:"
if [ -f "${ZT_CKPT}" ]; then
    echo "  FOUND  ${ZT_CKPT}  ($(du -sh "${ZT_CKPT}" | cut -f1))"
else
    echo "  MISSING ${ZT_CKPT}"
    echo "  ERROR: ZealotTransformer checkpoint is required. Aborting."
    exit 1
fi

# -----------------------------------------------------------------------------
# Helper: run a script with its argv
# -----------------------------------------------------------------------------
run_script() {
    local script="$1"
    shift               # remaining args become the argv list for the script
    local script_args=("$@")

    echo ""
    echo "────────────────────────────────────────"
    echo " Running: ${script}"
    echo " $(date)"
    echo "────────────────────────────────────────"

    python - << PYEOF
import sys, os
sys.path.insert(0, '${PYG_PKGS}')

argv = ['${WORK_DIR}/${script}']
extra = """${script_args[*]:-}""".split()
argv += [a for a in extra if a]   # skip empty strings
sys.argv = argv
print(f"  argv: {' '.join(argv)}\n", flush=True)

import runpy
runpy.run_path('${WORK_DIR}/${script}', run_name='__main__')
PYEOF

    local exit_code=$?
    echo " Done: ${script}  (exit ${exit_code})"
    if [ ${exit_code} -ne 0 ]; then
        echo " WARNING: ${script} exited with code ${exit_code} — continuing."
    fi
}

# -----------------------------------------------------------------------------
# 7 — Attention analysis
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 8 — Convergence speed
# -----------------------------------------------------------------------------
run_script "8_convergence_speed.py" \
    --zt_checkpoint "${ZT_CKPT}" \
    --log_path      "${SAVED}/training_log.json" \
    --out_dir       "${WORK_DIR}/result/convergence" \
    --n             1024 \
    --mc_runs       128 \
    --val_graphs    20 \
    --threshold     0.80 \
    --seed          42

# -----------------------------------------------------------------------------
# 9 — Speedup table
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "======================================================"
echo " All analysis done: $(date)"
echo " Results:"
ls -lh result/
echo "======================================================"