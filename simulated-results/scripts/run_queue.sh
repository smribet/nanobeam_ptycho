#!/usr/bin/env bash
# Run a job list from revision_jobs.py on this host's GPUs, several workers per GPU.
#
#   bash scripts/run_queue.sh outputs/rev/jobs_saturn.txt 6
#
# Workers claim the next line under flock, so jobs of very different cost balance
# themselves.  Every job skips itself once its DONE marker exists, so after an interruption
# reset the counter (rm outputs/rev/jobs_<host>.txt.next) and start the same command again.
#
# Before each job the worker checks the /home quota and waits while usage is above
# QUOTA_PAUSE_GB, so an unattended queue cannot run the account past its soft limit.
# GPUS="0" (say) keeps the queue off the other GPUs.
jobs=$1
per_gpu=${2:-6}
QUOTA_PAUSE_GB=${QUOTA_PAUSE_GB:-170}
cd "$(dirname "$0")/.." || exit 1
q=$(realpath "$jobs"); next="$q.next"; lock="$q.lock"
[ -f "$next" ] || echo 1 > "$next"
GPUS=${GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))}

used_gb() {
  quota -w 2>/dev/null | awk '/\/home|lv_home/{gsub(/\*/,"",$2); printf "%d", $2/1048576}'
}

worker() {
  export CUDA_VISIBLE_DEVICES=$1
  # a dozen workers share the host with other users; keep each one's CPU threads modest
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
  while :; do
    while [ "$(used_gb)" -ge "$QUOTA_PAUSE_GB" ] 2>/dev/null; do
      echo "$(date +%T) gpu$1 paused: /home at $(used_gb) GB"; sleep 300
    done
    n=$(flock "$lock" bash -c "n=\$(cat '$next'); echo \$((n+1)) > '$next'; echo \$n")
    cmd=$(sed -n "${n}p" "$q")
    [ -z "$cmd" ] && break
    echo "$(date +%T) gpu$1 job $n start"
    bash -c "$cmd"
    echo "$(date +%T) gpu$1 job $n exit $?"
  done
}

for g in $GPUS; do
  for _ in $(seq 1 "$per_gpu"); do worker "$g" & sleep 2; done
done
wait
echo QUEUE_DONE
