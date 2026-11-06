#!/usr/bin/env bash
# Wait until this worker's GPU has NEED_MIB free before a job starts, so a queue sharing a
# GPU with other users never pushes the card past its memory.
#
#   scripts/gpu_wait.sh 2000 && python ...
#
# Waiting workers pass one at a time: each holds the GPU's start lock for a further
# HOLD_S seconds after passing, long enough for its job to allocate, so the next one sees
# the memory that job took.
need=$1
gpu=${CUDA_VISIBLE_DEVICES:-0}
HOLD_S=${HOLD_S:-90}
exec 9> "/tmp/gpu_start_${USER}_${gpu}.lock"
flock 9
logged=
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")" \
        -ge "$need" ]; do
  [ -z "$logged" ] && echo "$(date +%T) gpu$gpu waiting for ${need} MiB free" && logged=1
  sleep 60
done
# the lock goes with fd 9 into this sleep, which outlives the script
sleep "$HOLD_S" &
