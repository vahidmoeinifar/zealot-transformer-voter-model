#!/bin/bash
# =============================================================================
#  LUMI — Analysis Jobs (GPU)
#  Runs scripts 5-8 in sequence:
#    5-attention_analysis.py    (Global-GAT attention)
#    6-convergence_speed.py     (convergence time vs rho_Z)
#    7-generalization_N.py      (unseen network sizes)
#    8-speedup_table.py         (MC vs SpectralLSTM timing)
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
#SBATCH --output=analysis-output.txt
#SBATCH --error=analysis-error.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.moeinifar@agh.edu.pl

set -euo pipefail

echo "======================================================"
echo " Job:     ${SLURM_JOB_NAME}  (${SLURM_JOB_ID})"
echo " Node:    ${SLURM_NODELIST}"
echo " Started: $(date)"
echo "======================================================"

module --force purge
module load LUMI/25.03
module load partition/G
module load lumi-container-wrapper/0.4.2-cray-python-3.11.7

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

run_script() {
    local script="$1"
    echo ""
    echo "────────────────────────────────────────"
    echo " Running: ${script}"
    echo " $(date)"
    echo "────────────────────────────────────────"
    python - << PYEOF
import sys
sys.path.insert(0, '${PYG_PKGS}')
import runpy
runpy.run_path('${WORK_DIR}/${script}', run_name='__main__')
PYEOF
    echo " Done: ${script}  (exit $?)"
}

run_script "7_attention_analysis.py"
run_script "8_convergence_speed.py"
run_script "9_speedup_table.py"
	

echo ""
echo "======================================================"
echo " All analysis done: $(date)"
echo " Results:"
ls -lh result/
echo "======================================================"
