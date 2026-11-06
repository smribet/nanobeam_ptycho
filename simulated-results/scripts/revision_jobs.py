"""Job lists for the simulations added in revision, one command per line.

    python revision_jobs.py            # writes outputs/rev/jobs_{jupiter,saturn}.txt

Every sweep is split into jobs of one (angle, Poisson draw) -- ten initial guesses each --
written to outputs/rev/<sweep>/parts/a<angle>_d<draw>/ and merged afterwards by
merge_parts.py.  Sweeps that invert stored multislice data wait for that data to exist, so
the queues can be started together with the forward-model jobs.

  R0   every on-carbon condition of the paper, regenerated with a physical 2.8 nm film
  N2   initial guesses seeded from the ground truth and its gauge-transformed versions
  N3   randomly perturbed scan positions, with a known and with a refined probe
  N4   amorphous carbon thickness
  N5   particle size
  N6   detector sampling: binning, cropping, and a larger field (finer sampling)
  N9   the model mismatches an experiment cannot avoid, alone and together

    python revision_jobs.py --stage 2  # writes outputs/rev/jobs_{jupiter,saturn}_2.txt

Stage 2 follows up on the stage-1 results, and must be generated where those results are:

  film      three more realisations of the 2.8 nm film and two of the published one,
            matched single-slice on carbon at 1e5 and 1e4
  2e5       a matched baseline at the experimental dose, and twelve draws (up from four)
            for the N9 rows there
  pair      N5b: the [111] particle next to the [100] one at 6 mrad, edge gaps 5-20 A
  show      chosen stage-1 runs repeated one seed at a time with every snapshot, the
            complex object and any refined probe stored, for figures.  The seeds are the
            ones `figpanels.pick_seeds` would show, and each job reuses the exact arguments
            of the stage-1 job it repeats.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

import paths

REV = paths.OUTPUTS / "rev"
FD = REV / "forward_data"
# the project interpreter directly rather than `uv run`, so a running queue never depends
# on the uv cache (which may be cleaned to stay within the /home quota)
UV = f"{paths.ROOT}/.venv/bin/python -u scripts"
# the GPUs are shared: a job waits until its GPU has this much memory free (MiB)
GUARD = f"{paths.ROOT}/scripts/gpu_wait.sh"
COMMON = "--iters 200 --snapshot-every 20 --array-seeds 0 --seeds 10"
COMMON2 = COMMON.replace("--array-seeds 0", "--array-seeds 3")   # snapshots for 3 seeds
C28 = "--support --carbon-thickness 28"
SOURCE_SIGMA, FOCAL_SIGMA = 0.34, 25.0       # Angstrom; 0.8 A FWHM source, X-FEG focal spread


def draws(dose, n_draws=4, noiseless_runs=10):
    """(label, seed_start, n_seeds, extra args) for every dataset at one dose."""
    if dose == "inf":
        return [(f"d{k}", 10 * k, 10, "") for k in range(noiseless_runs // 10)]
    return [(f"d{k}", 100 * k, 10, f"--dose {dose} --noise-seed {k}") for k in range(n_draws)]


def sweep(name, args, angles=(3.0,), doses=("inf", "1e5"), n_draws=4, noiseless_runs=10,
          wait=None, common=COMMON, draw_start=0, need=2000):
    jobs = []
    for a in angles:
        for d in doses:
            for lab, s0, n, extra in draws(d, n_draws, noiseless_runs)[draw_start:]:
                out = REV / f"{name}_{d}" / "parts" / f"a{a:g}_{lab}"
                cmd = (f"{GUARD} {need} && "
                       f"{UV}/run_sweep.py --out {out} --angles {a:g} --seed-start {s0} "
                       f"{common.replace('--seeds 10', f'--seeds {n}')} {extra} {args}")
                if wait:
                    # stored multislice data: wait until this angle has been written
                    cmd = (f"until grep -q '\"{a:g}\"' {wait}/meta.json 2>/dev/null; "
                           f"do sleep 60; done; {cmd}")
                jobs.append(f"mkdir -p {out}; [ -f {out}/DONE ] || "
                            f"({cmd} > {out}/job.log 2>&1 && touch {out}/DONE)")
    return jobs


def search(name, args, dose="inf", draws_=(0,), per_job=2, wait=None):
    """Gauge-search jobs (gauge_search.py), `per_job` reconstructions each.

    A search costs ~8x an ordinary reconstruction, so jobs are kept small to spread them
    evenly over the workers.
    """
    jobs = []
    for k in draws_:
        for s in range(0, 10, per_job):
            seed0 = (0 if dose == "inf" else 100 * k) + s
            noise = "" if dose == "inf" else f"--dose {dose} --noise-seed {k}"
            out = REV / f"gs_{name}_{dose}" / "parts" / f"a3_d{k}_s{s}"
            cmd = (f"{UV}/gauge_search.py --out {out} --angles 3 --seed-start {seed0} "
                   f"--seeds {per_job} --n-shift 3 {noise} {args}")
            if wait:
                cmd = (f"until grep -q '\"3\"' {wait}/meta.json 2>/dev/null; "
                       f"do sleep 60; done; {cmd}")
            jobs.append(f"mkdir -p {out}; [ -f {out}/DONE ] || "
                        f"({cmd} > {out}/job.log 2>&1 && touch {out}/DONE)")
    return jobs


def forward(name, args, angles=(3.0,)):
    out = FD / name
    return [f"mkdir -p {out}; [ -f {out}/DONE ] || ({UV}/make_multislice_data.py "
            f"--out {out} --angles {' '.join(f'{a:g}' for a in angles)} --backend torch "
            f"--no-ss {args} > {out}/job.log 2>&1 && touch {out}/DONE)"]


def original(sweep_name, part):
    """(script, arguments) of the stage-1 job that wrote `sweep_name`/parts/`part`."""
    pat = re.compile(rf"scripts/(\w+\.py) --out \S+/rev/{re.escape(sweep_name)}/parts/"
                     rf"{re.escape(part)} (.*?) > ")
    for host in ("jupiter", "saturn"):
        for line in (REV / f"jobs_{host}.txt").read_text().splitlines():
            m = pat.search(line)
            if m:
                return m[1], m[2]
    raise KeyError(f"no stage-1 job for {sweep_name}/{part}")


def show(sweep_name, angle, n):
    """Repeat the `n` runs of one sweep that `figpanels.pick_seeds` would show, storing
    everything; one job per seed, written to show_<sweep>/parts/."""
    import figpanels
    df = pd.read_csv(REV / sweep_name / "summary.csv")
    df = df[df.angle_mrad == angle].assign(noise_seed=lambda d: d.noise_seed.fillna(0))
    _, seeds, _ = figpanels.pick_seeds(dict(df=df, QF=figpanels._default_QF()), n=n)
    noiseless = sweep_name.endswith("_inf")
    jobs = []
    for seed in seeds:
        script, args = original(sweep_name, f"a{angle:g}_d{seed // (10 if noiseless else 100)}")
        args = re.sub(r"--seed-start \d+", f"--seed-start {seed}", args)
        args = re.sub(r"--seeds \d+", "--seeds 1", args)
        args = re.sub(r"--array-seeds \d+", f"--array-seeds {seed + 1}", args)
        args = re.sub(r"--snapshot-every \d+", "--snapshot-every 10", args)
        out = REV / f"show_{sweep_name}" / "parts" / f"a{angle:g}_s{seed:03d}"
        need = 3600 if "--field 136" in args else 2000
        jobs.append(f"mkdir -p {out}; [ -f {out}/DONE ] || ({GUARD} {need} && {UV}/{script} "
                    f"--out {out} {args} --store-object > {out}/job.log 2>&1 "
                    f"&& touch {out}/DONE)")
    return jobs


def show_search(name, n_each=1, min_corr=0.5, q_tol=0.25):
    """Repeat gauge searches, now storing the phase before and after: the seeds where the
    search fixed the most regions, and where the most were left wrong."""
    s = pd.read_csv(REV / name / "search.csv")
    s = s[s.before_corr.abs() >= min_corr]
    wrong0 = s.before_reversed | (s.before_q > q_tol)
    wrong1 = s.after_reversed | (s.after_q > q_tol)
    per = pd.DataFrame(dict(seed=s.seed, fixed=wrong0 & ~wrong1, left=wrong1)
                       ).groupby("seed").sum()
    seeds = list(per.fixed.nlargest(n_each).index)
    seeds += [k for k in per.left.sort_values(ascending=False).index if k not in seeds][:n_each]
    jobs = []
    for seed in seeds:
        k, r = divmod(seed, 100)
        _, args = original(name, f"a3_d{k}_s{r - r % 2}")
        args = re.sub(r"--seed-start \d+", f"--seed-start {seed}", args)
        args = re.sub(r"--seeds \d+", "--seeds 1", args)
        out = REV / f"show_{name}" / "parts" / f"a3_s{seed:03d}"
        jobs.append(f"mkdir -p {out}; [ -f {out}/DONE ] || ({GUARD} 2000 && "
                    f"{UV}/gauge_search.py --out {out} {args} > {out}/job.log 2>&1 "
                    f"&& touch {out}/DONE)")
    return jobs


def stage2():
    """Follow-ups to the stage-1 results; see the module docstring."""
    C28 = "--support --carbon-thickness 28"
    blind = "--probe-update --probe-defocus-error 20"

    def ms_data(fw):
        return f"{C28} --data-dir {FD}/{fw}_carbon28 --data-kind ms"

    long, short = [], []
    # ---- film: is R0's weaker phasing the film or this one realisation of it?
    for cs in (11, 12, 13):
        long += sweep(f"film_new_s{cs}", f"{C28} --carbon-seed {cs}", doses=("1e5", "1e4"),
                      common=COMMON2)
    for cs in (11, 12):
        long += sweep(f"film_legacy_s{cs}", f"--support --carbon-seed {cs}",
                      doses=("1e5", "1e4"), common=COMMON2)
    # ---- 2e5: the matched baseline, and draws 4-11 for the N9 rows
    long += sweep("ss_carbon28_complex", C28, doses=("2e5",), n_draws=12, common=COMMON2)
    for name, fw, extra in (("mm_ms_known", "ms", ""), ("mm_msfp_known", "msfp", ""),
                            ("mm_msall_known", "msall", ""),
                            ("mm_msall_blind", "msall", blind)):
        long += sweep(name, f"{ms_data(fw)} {extra}".strip(), doses=("2e5",), n_draws=12,
                      draw_start=4)
    # ---- pair (N5b): [100] moved along the diagonal; centres from the measured gaps
    for gap, (x, y) in ((5, (39.6, 25.4)), (10, (43.1, 21.9)), (20, (50.2, 14.8))):
        long += sweep(f"pair_gap{gap}_bare", f"--particles au_111 au_100 --place au_100 {x} {y}",
                      angles=(6.0,), common=COMMON2)
    # ---- show
    for name, a, n in (
            [(f"{s}_1e4", 3.0, 3) for s in ("ss_carbon28_complex", "ms_carbon28_a3_complex")]
            + [(f"{s}_1e5", 3.0, 3) for s in ("ss_carbon28_complex", "ms_carbon28_a3_phase",
                                              "msfp_carbon28_a3_complex")]
            + [(f"init_{i}_{s}_{d}", 3.0, 2) for i in ("truth", "reversed", "shifted")
               for s in ("bare", "carbon28") for d in ("inf", "1e5")]
            + [(f"{s}_bare_1e5", 3.0, 3) for s in ("jit_known", "jit_blind", "grid_blind")]
            + [(f"thick_ss_c{t}_1e5", 3.0, 2) for t in (10, 50, 100, 200)]
            + [("thick_ms_c200_1e5", 3.0, 2)]
            + [(f"single111_scale{s}_bare_1e5", 6.0, 3) for s in ("1", "1.6", "2")]
            + [("scale1.25_bare_1e5", 6.0, 2)]
            + [(f"samp_{s}_bare_1e5", 3.0, 2) for s in ("bin4", "field136")]
            + [(f"mm_msall_{p}_2e5", 3.0, 3) for p in ("known", "blind")]
            + [(f"mixobj_{fw}_1e5", 3.0, 2) for fw in ("msfp", "ms")]):
        short += show(name, a, n)
    for name in ("gs_ss_bare_inf", "gs_ss_carbon28_1e4", "gs_msfp_bare_1e5"):
        long = show_search(name) + long
    return long + short


def stage3():
    """The thickness series scored without phase wrapping, and what phases the [111].

    dw_thick   N4 again, 1 to 20 nm, with every phase taken relative to the object's own
               smoothed copy (run_sweep.py --phase-smooth): from 10 nm up, np.angle of the
               film plus a particle wraps, which the stage-1 scoring read as failure.  The
               thinner films are repeated too, so the series is scored one way throughout
    pair_pub   the [111] with only one of its published neighbours, at the published
               positions, at 6 mrad
    """
    C = "--support --carbon-thickness"
    dw = "--phase-smooth 4"
    long = []
    for t in (10, 28, 50, 100, 200):
        long += sweep(f"dw_thick_ss_c{t}", f"{C} {t} {dw}", doses=("1e5", "1e4"),
                      common=COMMON2)
        long += sweep(f"dw_thick_ms_c{t}", f"{C} {t} --data-dir {FD}/ms_carbon{t} "
                      f"--data-kind ms {dw}", doses=("1e5",), common=COMMON2)
    for lab, other in (("100", "au_100"), ("deca", "decahedron")):
        long += sweep(f"pair_pub{lab}_bare", f"--particles au_111 {other}", angles=(6.0,),
                      common=COMMON2)
    return long


def stage4():
    """R0's multislice rows pooled over film realisations.

    One realisation of the 2.8 nm film decides every multislice and phonon row of the
    paper, and the matched rows show realisations differ (7/28 non-unique at 1e5 on R0's,
    0-2/28 on three others).  So the 3 mrad multislice conditions of Fig. 5, the phonon
    figure, and Table S1 are repeated on three more realisations, to be pooled with R0's.
    Returns (forward jobs, sweeps): the sweeps wait for their data.
    """
    C28 = "--support --carbon-thickness 28"
    ALL = ("inf", "1e6", "1e5", "1e4")
    fwd, jobs = [], []
    for cs in (11, 12, 13):
        for fw, extra in (("ms", ""), ("msfp", " --phonons 8")):
            fwd += forward(f"{fw}_carbon28_s{cs}", f"{C28} --carbon-seed {cs}{extra}")
        for fw, obj in (("ms", "complex"), ("ms", "phase"), ("msfp", "complex")):
            data = FD / f"{fw}_carbon28_s{cs}"
            jobs += sweep(f"pool_{fw}_carbon28_s{cs}_a3_{obj}",
                          f"{C28} --carbon-seed {cs} --data-dir {data} --data-kind ms "
                          f"--object {obj}", doses=ALL, wait=data, common=COMMON2)
    return fwd, jobs


def stage6():
    """Fixes and follow-ups to the reviewers' figures.

    mixfix      the mixed-state object again, after fixing each mode's free global phase
                (sim.orthogonalize), with the modes stored
    *_pe        blind runs from a meaningfully wrong probe: 1500 A defocus and 150 A
                astigmatism (0.63 rad rms across the 3 mrad aperture; the 20 A of the first
                runs was 0.01 rad), with the refined probe stored
    gsfull      W. de Kleijne's gauge search: each particle edited and the whole
                reconstruction re-converged from it (gauge_search_full.py), one job per
                (reconstruction, particle)
    """
    C28 = "--support --carbon-thickness 28"
    pe = "--probe-update --probe-defocus-error 1500 --probe-astig-error 150 --store-object"
    jit = "--scan-jitter 1.0 --jitter-seed 7"

    def ms_data(fw):
        return f"{C28} --data-dir {FD}/{fw}_carbon28 --data-kind ms"

    jobs = []
    for fw in ("msfp", "ms"):
        jobs += sweep(f"mixfix_{fw}", f"{ms_data(fw)} --object mixed --modes 4 --store-object",
                      doses=("1e5",), n_draws=1, common=COMMON2)
    for lab, extra in (("grid_blind_pe", pe), ("jit_blind_pe", f"{jit} {pe}")):
        jobs += sweep(f"{lab}_bare", extra, common=COMMON2)
        jobs += sweep(f"{lab}_carbon28", f"{C28} {extra}", doses=("1e4",), common=COMMON2)
    jobs += sweep("mm_ms_blind_pe", f"{ms_data('ms')} {pe}", doses=("1e5", "2e5"))
    jobs += sweep("mm_msall_blind_pe", f"{ms_data('msall')} {pe}", doses=("1e5",))
    jobs += sweep("mm_msall_blind_pe", f"{ms_data('msall')} {pe}", doses=("2e5",), n_draws=12)
    regions = ["au_111", "au_100"] + [f"decahedron_w{k}" for k in range(5)]
    for name, seeds, extra in (("ss_bare_inf", (4, 7), ""),
                               ("ss_bare_1e5", (2, 4), "--dose 1e5 --noise-seed 0"),
                               ("ss_carbon28_1e4", (7, 102), C28)):
        for sd in seeds:
            noise = (f"--dose 1e4 --noise-seed {sd // 100}" if "carbon" in name else "")
            for rg in regions:
                out = REV / f"gsfull_{name}" / "parts" / f"s{sd:03d}_{rg}"
                cmd = (f"{GUARD} 2000 && {UV}/gauge_search_full.py --out {out} --angles 3 "
                       f"--seed {sd} --region {rg} {extra} {noise}")
                jobs.append(f"mkdir -p {out}; [ -f {out}/DONE ] || ({cmd} > {out}/job.log "
                            f"2>&1 && touch {out}/DONE)")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 6])
    stage = ap.parse_args().stage
    if stage == 6:
        jobs = stage6()
        # the two long mixed-object jobs first, one per host; the rest dealt alternately
        split = {"jupiter": jobs[0:1] + jobs[2::2], "saturn": jobs[1:2] + jobs[3::2]}
        for name, js in split.items():
            f = REV / f"jobs_{name}_6.txt"
            f.write_text("\n".join(js) + "\n")
            print(f"{name:8s} {len(js):4d} jobs -> {f}")
        return
    if stage in (2, 3, 4):
        REV.mkdir(parents=True, exist_ok=True)
        if stage == 4:
            # jupiter runs on both GPUs and saturn on one (the other is for the experimental
            # reconstructions), so the sweeps are dealt 2:1, after the forward jobs
            fwd, jobs = stage4()
            split = {"jupiter": fwd + [j for k, j in enumerate(jobs) if k % 3 != 2],
                     "saturn": [j for k, j in enumerate(jobs) if k % 3 == 2]}
        else:
            # longest first, dealt alternately, so the two hosts finish together
            jobs = stage2() if stage == 2 else stage3()
            split = {"jupiter": jobs[0::2], "saturn": jobs[1::2]}
        for name, js in split.items():
            f = REV / f"jobs_{name}_{stage}.txt"
            f.write_text("\n".join(js) + "\n")
            print(f"{name:8s} {len(js):4d} jobs -> {f}")
        return
    A3 = (3.0, 6.0, 12.0)
    ALL = ("inf", "1e6", "1e5", "1e4")
    FWD, JUP, SAT = [], [], []

    # ---- forward data (GPU multislice), first in jupiter's queue, quickest first; the
    # sweeps below follow in the order their data become available, so a worker rarely
    # sits waiting on a forward job
    FWD += forward("ms_carbon28", C28, A3)
    for t in (10, 50, 100, 200):
        FWD += forward(f"ms_carbon{t}", f"--support --carbon-thickness {t}")
    FWD += forward("mssrc_carbon28", f"{C28} --source-sigma {SOURCE_SIGMA}")
    FWD += forward("msfoc_carbon28", f"{C28} --focal-sigma {FOCAL_SIGMA}")
    FWD += forward("msfp_carbon28", f"{C28} --phonons 8", A3)
    FWD += forward("msall_carbon28", f"{C28} --phonons 8 --source-sigma {SOURCE_SIGMA} "
                                     f"--focal-sigma {FOCAL_SIGMA} --n-source 3 --n-focal 3")

    def ms_data(fw):
        return f"{C28} --data-dir {FD}/{fw}_carbon28 --data-kind ms"

    # ---- R0: every on-carbon condition, physical 2.8 nm film
    SAT += sweep("ss_carbon28_complex", C28, A3, ALL, noiseless_runs=50)
    for obj in ("complex", "phase"):
        for a in A3:
            JUP += sweep(f"ms_carbon28_a{a:g}_{obj}", f"{ms_data('ms')} --object {obj}",
                         (a,), ALL, wait=FD / "ms_carbon28")

    # ---- N4 (multislice arm): carbon thickness
    for t in (10, 50, 100, 200):
        JUP += sweep(f"thick_ms_c{t}", f"--support --carbon-thickness {t} "
                     f"--data-dir {FD}/ms_carbon{t} --data-kind ms", doses=("1e5",),
                     wait=FD / f"ms_carbon{t}")

    # ---- N9: unavoidable mismatches at 1e5 and the experimental 2e5
    blind = "--probe-update --probe-defocus-error 20"
    JUP += sweep("mm_ms_known", ms_data("ms"), doses=("2e5",), wait=FD / "ms_carbon28")
    JUP += sweep("mm_ms_blind", f"{ms_data('ms')} {blind}", doses=("1e5", "2e5"),
                 wait=FD / "ms_carbon28")
    for fw in ("mssrc", "msfoc"):
        JUP += sweep(f"mm_{fw}_known", ms_data(fw), doses=("1e5", "2e5"),
                     wait=FD / f"{fw}_carbon28")
    for a in A3:
        JUP += sweep(f"msfp_carbon28_a{a:g}_complex", ms_data("msfp"), (a,), ALL,
                     wait=FD / "msfp_carbon28")
    JUP += sweep("mm_msfp_known", ms_data("msfp"), doses=("2e5",), wait=FD / "msfp_carbon28")
    JUP += sweep("mm_msall_known", ms_data("msall"), doses=("1e5", "2e5"),
                 wait=FD / "msall_carbon28")
    JUP += sweep("mm_msall_blind", f"{ms_data('msall')} {blind}", doses=("1e5", "2e5"),
                 wait=FD / "msall_carbon28")

    # ---- gauge search (W. de Kleijne): can the misfit pick each particle's registry?
    # Matched free-standing data, then multislice + phonon data reconstructed single-slice
    # (does it survive model mismatch?), then the low-dose on-carbon case.  Appended last.
    GS = forward("msfp_bare", "--no-support --phonons 8")
    GS += search("ss_bare", "", "inf")
    GS += search("ss_bare", "", "1e5", (0, 1))
    GS += search("msfp_bare", f"--data-dir {FD}/msfp_bare --data-kind ms", "1e5", (0, 1),
                 wait=FD / "msfp_bare")
    GS += search("ss_carbon28", C28, "1e4", (0, 1))
    JUP = FWD + JUP + GS

    # ---- N2: seeded initial guesses
    for init in ("truth", "reversed", "shifted"):
        SAT += sweep(f"init_{init}_bare", f"--init {init}", n_draws=2)
        SAT += sweep(f"init_{init}_carbon28", f"{C28} --init {init}", n_draws=2)

    # ---- N3: scan geometry, known and refined probe
    jit = "--scan-jitter 1.0 --jitter-seed 7"
    for lab, extra in (("jit_known", jit), ("jit_blind", f"{jit} {blind}"),
                       ("grid_blind", blind)):
        SAT += sweep(f"{lab}_bare", extra)
        SAT += sweep(f"{lab}_carbon28", f"{C28} {extra}", doses=("1e4",))

    # ---- N4 (matched arm): carbon thickness (0 is ss_bare, 28 is R0)
    for t in (10, 50, 100, 200):
        SAT += sweep(f"thick_ss_c{t}", f"--support --carbon-thickness {t}",
                     doses=("1e5", "1e4"))

    # ---- N5: particle size, all three particles, then the [111] particle alone
    for s in (0.6, 0.8, 1.25):
        SAT += sweep(f"scale{s:g}_bare", f"--scale {s}", (3.0, 6.0))
    for s in (1.0, 1.6, 2.0):
        SAT += sweep(f"single111_scale{s:g}_bare", f"--scale {s} --particles au_111",
                     (3.0, 6.0))

    # ---- N6: detector sampling
    for lab, extra in (("bin2", "--bin 2"), ("bin4", "--bin 4"), ("crop192", "--crop-px 192"),
                       ("field136", "--field 136"),
                       ("field136crop192", "--field 136 --crop-px 192")):
        SAT += sweep(f"samp_{lab}_bare", extra)
        if lab in ("bin2", "bin4", "field136"):
            SAT += sweep(f"samp_{lab}_carbon28", f"{C28} {extra}", doses=("1e4",))

    # ---- mixed-state object (reviewer 1, comment 2): four incoherent object modes on data
    # with and without thermal motion.  Appended last so a running queue can take it on
    # without renumbering the jobs it has already claimed.
    for fw in ("msfp", "ms"):
        SAT += sweep(f"mixobj_{fw}", f"{ms_data(fw)} --object mixed --modes 4",
                     doses=("1e5",), n_draws=1, wait=FD / f"{fw}_carbon28")

    REV.mkdir(parents=True, exist_ok=True)
    for name, jobs in (("jupiter", JUP), ("saturn", SAT)):
        (REV / f"jobs_{name}.txt").write_text("\n".join(jobs) + "\n")
        print(f"{name:8s} {len(jobs):4d} jobs -> {REV / f'jobs_{name}.txt'}")


if __name__ == "__main__":
    main()
