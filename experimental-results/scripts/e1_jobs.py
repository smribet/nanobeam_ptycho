"""Job lists for the experimental re-reconstructions, one command per line, run by
simulated-results/scripts/run_queue.sh like the simulation queues.

    python e1_jobs.py      # writes ~/data/nanobeam_ptycho/recons/e1/jobs_{saturn,jupiter}.txt

saturn, GPU 1: the two below-threshold scans (26 at 3 mrad, 27 at 1.5 mrad), four seeds
each to 200 iterations, whose 50-iteration snapshot is the published setting; then each
scan again from three new seeds with the probe of its seed-0 run frozen.
jupiter, after its simulation queue: the 24 mrad reference (23), the 6 mrad control (25),
and four more seeds of each below-threshold scan.
"""
from pathlib import Path

HOME = Path.home()
OUT = HOME / "data/nanobeam_ptycho/recons/e1"
PY = HOME / "git-repos/nanobeam_ptycho/simulated-results/.venv/bin/python"
SCRIPT = HOME / "git-repos/nanobeam_ptycho/experimental-results/scripts/reconstruct_quantem.py"
GUARD = HOME / "git-repos/nanobeam_ptycho/simulated-results/scripts/gpu_wait.sh"


def job(name, args, wait_for=None):
    log = OUT / "logs" / f"{name}.log"
    done = OUT / "logs" / f"{name}.DONE"
    cmd = (f"TQDM_DISABLE=1 OMP_NUM_THREADS=16 {PY} -u {SCRIPT} {args} --snapshot-every 50 "
           f"--out {OUT}")
    if wait_for:
        cmd = f"until [ -f {wait_for} ]; do sleep 120; done; {cmd}"
    # a full binned scan takes ~37 GB of the 46 GB card, so wait for nearly all of it
    return f"[ -f {done} ] || ({GUARD} 38000 && {cmd} > {log} 2>&1 && touch {done})"


def epie_jobs(rotation=169.14, sign=-1):
    """The independent pipeline (epie_independent.py) on jupiter: the 24 mrad control, the
    two below-threshold scans, then the 6 mrad one; the rotation is the one its misfit scan
    found on scan 23, used for every scan."""
    script = SCRIPT.with_name("epie_independent.py")
    out = OUT / "epie"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for scan, seeds, iters in ((23, "0 1", 100), (26, "0 1 2 3", 150), (27, "0 1 2 3", 150),
                               (25, "0 1", 150)):
        name = f"epie_scan{scan}"
        log, done = out / "logs" / f"{name}.log", out / "logs" / f"{name}.DONE"
        cmd = (f"OMP_NUM_THREADS=16 {PY} -u {script} --scan {scan} --seeds {seeds} "
               f"--iters {iters} --snapshot-every 25 --rotation {rotation} "
               f"--phase-sign {sign} --out {out}")
        # quantEM's footprint swings between ~28 and ~37 GB, so wait until it has left
        jobs.append(f"[ -f {done} ] || ({GUARD} 30000 && {cmd} > {log} 2>&1 && touch {done})")
    (out / "jobs_jupiter.txt").write_text("\n".join(jobs) + "\n")
    print(f"epie     {len(jobs)} jobs -> {out / 'jobs_jupiter.txt'}")


def main():
    import sys
    if "--epie" in sys.argv:
        return epie_jobs()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    saturn, jupiter = [], []
    for scan in (26, 27):
        saturn.append(job(f"scan{scan}_s0-3", f"--scan {scan} --seeds 0 1 2 3 --iters 200"))
    for scan in (26, 27):
        ref = OUT / f"scan{scan}_bin2_s000_it200.h5"
        saturn.append(job(f"scan{scan}_fixed_s10-12", f"--scan {scan} --seeds 10 11 12 "
                          f"--iters 200 --probe-from {ref}", wait_for=ref))
    jupiter.append(job("scan23_s0-1", "--scan 23 --seeds 0 1 --iters 100"))
    jupiter.append(job("scan25_s0-3", "--scan 25 --seeds 0 1 2 3 --iters 200"))
    for scan in (26, 27):
        jupiter.append(job(f"scan{scan}_s4-7", f"--scan {scan} --seeds 4 5 6 7 --iters 200"))
    for host, jobs in (("saturn", saturn), ("jupiter", jupiter)):
        (OUT / f"jobs_{host}.txt").write_text("\n".join(jobs) + "\n")
        print(f"{host:8s} {len(jobs)} jobs -> {OUT / f'jobs_{host}.txt'}")


if __name__ == "__main__":
    main()
