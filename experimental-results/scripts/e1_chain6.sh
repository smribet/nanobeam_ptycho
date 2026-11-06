#!/usr/bin/env bash
# Experimental re-runs for the second round of figures, one after another on one GPU:
#   e1_chain6.sh GPU NEED_MIB "args for reconstruct_quantem.py or run_ptyrad.py" ...
gpu=$1; need=$2; shift 2
E=~/data/nanobeam_ptycho/recons/e1; L=$E/logs6; mkdir -p $L
cd ~/git-repos/nanobeam_ptycho/experimental-results/scripts || exit 1
PY=~/git-repos/nanobeam_ptycho/simulated-results/.venv/bin/python
G=~/git-repos/nanobeam_ptycho/simulated-results/scripts/gpu_wait.sh
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 TQDM_DISABLE=1
for job in "$@"; do
  name=$(echo "$job" | tr -c "a-zA-Z0-9" "_" | cut -c1-60)
  echo "$(date +%T) start: $job"
  $G $need && $PY -u $job > $L/$name.log 2>&1
  echo "$(date +%T) exit $?: $job"
done
echo "chain done $(date +%T)"
