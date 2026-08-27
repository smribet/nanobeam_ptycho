# Simulated results

Multislice simulations and ptychographic reconstruction sweeps for the nanobeam
ptychography ambiguity study, plus the code that turns them into the paper's figures.

The question these sweeps answer: below the disk-overlap threshold, does iterative
ptychography converge to *one* object, or to several that fit the data equally well? Each
sweep reconstructs the same simulated dataset from many different random initial guesses
and asks whether they agree.

```
scripts/    the simulation, reconstruction, analysis, and figure code
outputs/    one directory per sweep, plus the forward-model metadata and DP caches
figures/    the five paper figures (PNG + PDF) and the summary table
```

## Reproducing the figures

```bash
uv sync
uv run python scripts/make_figures.py            # fig_sweep, fig_si_substrate, fig_si_phonons
uv run python scripts/make_fig_si_sparklines.py  # fig_si_sparklines
uv run python scripts/make_fig_stitching.py      # fig_stitching (re-runs the forward model)
uv run python scripts/summary_table.py           # figures/summary_table.{csv,md}
```

| figure | what it shows | made by |
|---|---|---|
| `fig_sweep` | dose, then object model — one variable per row | `make_figures.py fig_sweep` |
| `fig_stitching` | phase-free nanodiffraction stitching, the negative control | `make_fig_stitching.py` |
| `fig_si_substrate` | free-standing vs on amorphous carbon, everything else matched | `make_figures.py fig_si_substrate` |
| `fig_si_phonons` | with and without 8 frozen-phonon configurations | `make_figures.py fig_si_phonons` |
| `fig_si_sparklines` | every run, every checkpoint, every region | `make_fig_si_sparklines.py` |

Everything writes into `figures/`, which is committed, so a fresh run can be diffed against
what the paper used. All five figures and the summary table regenerate from what is in this
directory — no raw 4D-STEM data required.

Paths resolve against the project root via `scripts/paths.py`, so the scripts run from any
working directory.

## Re-running a sweep

`run_sweep.py` simulates and reconstructs in one go. A short run, to check the setup:

```bash
uv run python scripts/run_sweep.py --out outputs/smoke --support \
    --angles 12 3 --seeds 2 --iters 10 --snapshot-every 5 --dose 1e5 --noise-seed 0
```

That takes about 30 s and should show the effect the paper is about: at 12 mrad the disks
overlap and the reconstruction is clean (global correlation ≈ 0.92, no polarity flips); at
3 mrad they do not, and regions come back ambiguous or contrast-reversed. It writes the same
three CSVs and `runs/*.npz` as the shipped sweeps, and `sweep.log` is just its stdout
redirected.

`--noise-seed` is the important flag: it fixes the photon-noise realisation so every
reconstruction seed inverts the *same* dataset and only the initial guess differs. Without
it each seed draws fresh noise, which measures something else entirely — whether the
information survives the dose, rather than whether one measurement determines one answer.
The paper's sweeps all use it.

## Sweep naming

```
ms_carbon_a3_phase_1e4
│  │      │  │     └─ dose in e/Å², or `inf` for noiseless
│  │      │  └─────── object model used in the inverse: complex | phase
│  │      └────────── convergence semi-angle in mrad
│  └───────────────── sample:  carbon (Au on amorphous carbon) | bare (free-standing)
└──────────────────── forward model
```

| forward | |
|---|---|
| `ss` | single-slice — **matched** to the inverse model, so any failure is the reconstruction's own rather than a forward/inverse mismatch |
| `ms` | multislice |
| `msfp` | multislice, averaged over 8 frozen-phonon configurations |

The angle field appears only on the sweeps that ran one angle per directory (`ms_*`,
`msfp_*`); the `ss_*` sweeps carry all three angles in one directory, so their `runs/`
contain `a3_*`, `a6_*`, and `a12_*` side by side.

`ss_carbon_complex_inf` doubles as the reference sweep: its `runs/_reference.npz` supplies
the ground truth, the seven region masks, and each region's lattice basis to every figure.
`ss_bare_complex_1e5/runs/_reference.npz` does the same for the free-standing sample.

Two directories are not sweeps: `outputs/dp_cache/` holds pre-computed diffraction patterns
for the panel-a insets (one per sample), and `outputs/forward_data/` records the parameters
of the multislice runs whose exit waves were too large to commit.

## What's committed, and what isn't

The raw datasets are far too large to version: the multislice output alone is 6.4 GB, and
the reconstructions add another 900 MB. Committed instead is everything the figures and the
reported statistics are computed from.

| kept | |
|---|---|
| `summary.csv` | one row per (run, region) — final polarity, offset, correlation, error |
| `drift.csv` | the same per-region metrics at every 20th iteration, which is what the sparklines and offset trails plot |
| `boundaries.csv` | twin-boundary registry error and sign flips |
| `sweep.log` | the run's console output, captured by redirecting `run_sweep.py` — geometry, per-seed timings, and polarity flips |
| `runs/*.npz` | **only** the reconstructions the figures display — 54 of ~900 |
| `runs/_reference.npz` | ground truth, region masks, lattice bases; one per sample |

| left out | why |
|---|---|
| the multislice exit waves | 6.4 GB; `outputs/forward_data/*/meta.json` records the parameters that produced them |
| most `runs/*.npz` | ~900 reconstructions; every metric derived from them is in `summary.csv` and `drift.csv` |
| the superseded sweeps | an early free-standing pass on an older summary schema, a 30 kV comparison, a dose series replaced by the fixed-noise design, and a first-pass 300 keV sweep — none used by any figure or by the table |

Regenerating the omitted reconstructions means re-running the forward model
(`make_multislice_data.py`) and then the sweep (`run_sweep.py`); the `sweep.log` in each
directory records the exact invocation and geometry.

## Seed numbering

**The seed encodes two different things, and they matter for different reasons.**

```
seed = 100 * (Poisson draw) + (initial guess)
```

So `seed 207` is the 8th initial guess applied to the 3rd noise realisation. Runs sharing a
hundreds digit inverted the *same* photon-noise draw and differ only in where the optimiser
started — disagreement among them is non-uniqueness. Runs in different hundreds blocks
differ in the noise itself, so disagreement between blocks is just a different dataset.

The full design is 4 draws × 10 initial guesses = 40 runs per convergence angle. Noiseless
sweeps have only one dataset, so they carry 10 runs (seeds 0–9).

Because only the displayed reconstructions ship, `runs/` is sparse — the figures select
seeds via `pick_seeds`, which reads `summary.csv`, so it picks the same ones the paper used.
Ask for a seed that wasn't kept and the load fails loudly rather than silently substituting.

## Regions

Every sweep scores seven regions of the same field of view: two single-crystal particles
(`au_111`, `au_100`) and the five `<110>` wedges of a five-fold twin (`decahedron_w0`…`w4`,
labelled `[110] wedge 1`…`5` in the figures). The twin is split because it has no single
lattice — each wedge gets its own basis, with the cuts placed on the twin planes by
maximising lattice sharpness.

## Metrics

| column | meaning |
|---|---|
| `signed_corr` | correlation of the reconstructed lattice against truth; **negative means contrast-reversed** |
| `reversed` | `signed_corr < 0` |
| `ws_x`, `ws_y` | rigid offset of the reconstructed lattice, folded into the region's Wigner–Seitz cell |
| `peak_ratio` | weaker/stronger correlation extremum — how close the run came to the opposite polarity |
| `global_corr` | correlation over the whole field, i.e. reconstruction quality |

`occupancy` is present in the header but empty in every sweep here: it records the weights
of the object modes, and is only written by `--object mixed` runs, which none of these are.

The headline number, `nonuniq` in `summary_table.md`, is the fraction of (draw, region)
groups whose ten initial guesses fail to agree — either they disagree on polarity, or their
offsets spread by more than a quarter of the Wigner–Seitz cell. It is computed by
`figpanels.nonunanimity`, which both the table and the figures call, so the two cannot drift
apart.

## Environment

`pyproject.toml` and `uv.lock` pin the whole stack. Two notes:

- **py4DSTEM comes from git**, not PyPI: the released version still pins `numpy<2`, which
  abtem 1.x will not tolerate. Only `process.utils.get_maxima_2D` is used here.
- The figures were originally produced under NumPy 2.4. `analysis.cross2` replaces the 2-D
  `np.cross` overload, which NumPy deprecated in 2.0 and removed in 2.5; results are
  unchanged, and the summary table is bit-identical across both.
