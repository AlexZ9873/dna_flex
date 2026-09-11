#!/usr/bin/env bash
# B4a creation PROCEDURE ONLY. Running it requires a separately authorized
# real CARC CPU compute allocation. B4a implementation/testing does not run it.
#
# Usage (all paths absolute and fresh; expected commit includes accepted B4a):
#   bash scripts/carc/create_cnn_rc_environment.sh \
#     --spec /checkout/environments/carc_cnn_rc_v1.json \
#     --prefix /project/environments/carc_cnn_rc_v1 \
#     --cache-root /project/acquisition/carc_cnn_rc_v1 \
#     --exports-root /project/evidence/carc_cnn_rc_v1_exports \
#     --inventory-output /project/evidence/carc_cnn_rc_v1_inventory.json \
#     --expected-software-commit FULL_40_CHARACTER_COMMIT
#
# Missing binaries stop immediately; partial prefixes/caches/evidence remain.
# This script never accepts an environment. CPU and P100 evidence followed by
# explicit finalize are separate future authorized verification operations.
# No biological input, data transfer, stage assembly, or training occurs here.
#
# MANDATORY B4b HANDOFF (documentation only; not executed by this procedure):
# In the newly installed/activated CARC environment, a CPU verification job
# must first run the accepted B4a CPU verifier and this EXACT B1-B3b suite:
# PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" \
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
# NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
# python -B -m unittest -v \
#   tests.test_cnn_rc tests.test_downstream_metrics \
#   tests.test_downstream_fingerprints tests.test_exd_hox_dataset \
#   tests.test_exd_hox_staging tests.test_downstream_run \
#   tests.test_downstream_checkpoint tests.test_cnn_rc_training \
#   tests.test_cnn_rc_cli
# The accepted total is currently 202 tests. Any failure blocks both P100
# verification and the biological smoke run. B4b requests its GPU exactly as:
#   --gpus-per-task=p100:1
# B3b has no approved SIGUSR1/SIGTERM checkpoint-request interface. The first
# smoke uses no automatic requeue, no signal-to-checkpoint claim, no signal
# translation to SIGINT, and no automatic retry/resume. B3b publishes completion
# before CLI return: use durable project-storage result root from the beginning
# and validate independently afterward. Do not relocate cross-filesystem.

set -euo pipefail

spec=""
environment_prefix=""
cache_root=""
exports_root=""
inventory_output=""
expected_commit=""
declare -A seen_arguments=()
while (( $# > 0 )); do
    argument="$1"
    if [[ -n "${seen_arguments[$argument]+present}" ]]; then
        echo "Duplicate argument: $argument" >&2
        exit 2
    fi
    seen_arguments[$argument]=1
    if (( $# < 2 )); then
        echo "Every argument requires an explicit value." >&2
        exit 2
    fi
    case "$argument" in
        --spec) spec="$2" ;;
        --prefix) environment_prefix="$2" ;;
        --cache-root) cache_root="$2" ;;
        --exports-root) exports_root="$2" ;;
        --inventory-output) inventory_output="$2" ;;
        --expected-software-commit) expected_commit="$2" ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
    shift 2
done

for value in "$spec" "$environment_prefix" "$cache_root" "$exports_root" "$inventory_output"; do
    [[ "$value" == /* && "$value" != / ]] || { echo "Absolute explicit paths are required." >&2; exit 2; }
done
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "A full expected commit is required." >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A real CPU Slurm compute allocation is required." >&2; exit 2; }
[[ ! -e "$inventory_output" && ! -L "$inventory_output" ]] || { echo "Inventory output already exists; preserve it." >&2; exit 2; }
[[ -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" ]] || { echo "Python path overrides are forbidden." >&2; exit 2; }

module purge
module load conda/25.11.0
eval "$(conda shell.bash hook)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python -B "$script_directory/cnn_rc_environment.py" \
    --spec "$spec" --prefix "$environment_prefix" --cache-root "$cache_root" \
    --exports-root "$exports_root" --expected-software-commit "$expected_commit"

conda activate "$environment_prefix"
project_directory="$(cd -- "$script_directory/../.." && pwd -P)"
cd -- "$project_directory"
python -B -m scripts.carc.verify_cnn_rc_environment inventory \
    --spec "$spec" --prefix "$environment_prefix" \
    --expected-software-commit "$expected_commit" \
    --acquisition-lock "$cache_root/acquisition-lock.json" \
    --exports-root "$exports_root" --output "$inventory_output"
