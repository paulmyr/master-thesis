#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")" # run from script's directory

source ../.venv/bin/activate

# Force single-threaded execution per process (24 shards already give parallelism)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Adjust the range to match your physical core count
for SHARD in {0..3}; do
  SHARD=$SHARD python hparam_sweep.py >"shard_${SHARD}.log" 2>&1 &
done

wait
echo "All ${#} shards done"
