#!/usr/bin/env bash
# PtyRAD runs one after another on one GPU:  ptyrad_chain.sh GPU "SCAN:SEEDS:ITERS" ...
gpu=$1; shift
L=~/data/nanobeam_ptycho/recons/e1/ptyrad/logs; mkdir -p "$L"
cd ~/git-repos/nanobeam_ptycho/experimental-results/scripts || exit 1
PY=~/git-repos/nanobeam_ptycho/simulated-results/.venv/bin/python
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16
for job in "$@"; do
  IFS=: read -r scan seeds iters <<< "$job"
  echo "$(date +%T) scan $scan seeds $seeds iters $iters"
  $PY -u run_ptyrad.py --scan "$scan" --seeds $seeds --iters "$iters" --flipT 0 0 0 --rot 169.14 \
      > "$L/ptyrad_scan$scan.log" 2>&1
  echo "$(date +%T) scan $scan exit $?"
done
echo "chain done $(date +%T)"
