# Simulated results

Multislice simulations and ptychographic reconstruction sweeps for the nanobeam
ptychography ambiguity study, plus the code that turns them into the paper's figures.

The question these sweeps answer: below the disk-overlap threshold, does iterative
ptychography converge to *one* object, or to several that fit the data equally well? Each
sweep reconstructs the same simulated dataset from many different random initial guesses
and asks whether they agree.

```
scripts/        the simulation, reconstruction, analysis, and figure code
outputs/        one directory per sweep of the submitted version, plus metadata and DP caches
outputs/rev/    the sweeps added in the revision (physical carbon film, new tests)
figures/        the paper figures (PNG + PDF), the SI tables, and the records they quote
figures/revision/, figures/response/   the SI and response figures added in the revision
```

## Reproducing the figures and tables

```bash
uv sync
uv run python scripts/make_figures.py            # Figs. 5, S6, S7
uv run python scripts/make_fig_si_sparklines.py  # Fig. S8
uv run python scripts/make_fig_stitching.py      # Fig. 4 (re-runs the forward model)
uv run python scripts/make_fig_revision.py seeded pair thickness mixed   # Figs. S12, S15, S16, S19
uv run python scripts/make_fig_response.py       # Figs. S13, S14, S17, S18, S20
uv run python scripts/table_s1.py                # Tables S1-S3
uv run python scripts/draw_level.py              # dataset-level counts quoted in Section 3.4
uv run python scripts/gsfull_results.py          # the gauge search of Section 4.2
```

| paper | what it shows | file in `figures/` | made by |
|---|---|---|---|
| Fig. 4 | phase-free nanodiffraction stitching, the negative control | `fig_stitching` | `make_fig_stitching.py` |
| Fig. 5 | dose, then object model, one variable per row | `fig_sweep` | `make_figures.py fig_sweep` |
| Fig. S6 | free-standing vs on the carbon film, everything else matched | `fig_si_substrate` | `make_figures.py fig_si_substrate` |
| Fig. S7 | with and without 8 frozen-phonon configurations | `fig_si_phonons` | `make_figures.py fig_si_phonons` |
| Fig. S8 | every run, every checkpoint, every region | `fig_si_sparklines` | `make_fig_si_sparklines.py` |
| Fig. S12 | four-mode mixed-state object against a single object | `revision/rev_mixed` | `make_fig_revision.py mixed` |
| Fig. S13 | thermal motion and combined model mismatches | `response/fig_resp_mismatch` | `make_fig_response.py mismatch` |
| Fig. S14 | particle size, and the [111] particle's neighbors | `response/fig_resp_size` | `make_fig_response.py size` |
| Fig. S15 | reconstructions behind Fig. S14b | `revision/rev_pair` | `make_fig_revision.py pair` |
| Fig. S16 | starts from the ground truth and its gauge transforms | `revision/rev_seeded` | `make_fig_revision.py seeded` |
| Fig. S17 | final misfit of random and seeded starts | `response/fig_resp_loss` | `make_fig_response.py loss` |
| Fig. S18 | the carbon support: thickness, folded vs physical film | `response/fig_resp_carbon` | `make_fig_response.py carbon` |
| Fig. S19 | one reconstruction per film thickness | `revision/rev_thickness` | `make_fig_revision.py thickness` |
| Fig. S20 | dynamical scattering in thick periodic slabs | `response/fig_resp_dynamical` | `make_fig_response.py dynamical` |
| Tables S1–S3 | every condition; detector sampling; scan geometry | `table_s1.{tex,csv}`, `table_s2.tex`, `table_s3.tex` | `table_s1.py` |

Figs. S9–S11 are experimental, in [`../experimental-results`](../experimental-results).
`make_fig_revision.py` also draws the figures that appear only in the response to the
reviewers (`sampling`, `scan`, `probe`, `search`, `mismatch`). Those figures are in
`figures/revision/`, but their inputs are not committed, so run without arguments the script
skips them.

The records in `figures/` hold the numbers the text quotes:

| file | what | made by |
|---|---|---|
| `draw_level.{md,csv}` | every condition counted by dataset, including the combined mismatches and the thickness series | `draw_level.py` |
| `gsfull_results.{md,csv}` | the full gauge search, one row per edited region | `gsfull_results.py` |
| `pe_results.{md,csv}` | how well blind runs recover the probe from a wrong start | `pe_results.py` (needs stored probes, not committed) |
| `dynamical_check.json` | pattern variation with probe position in thick slabs, read by Fig. S20 | `dynamical_check.py` |
| `summary_table.{md,csv}` | Table S1 of the submitted version, on the folded film | `summary_table.py` |

Everything writes into `figures/`, which is committed, so a fresh run can be diffed against
what the paper used. No raw 4D-STEM data are required.

Paths resolve against the project root via `scripts/paths.py`, so the scripts run from any
working directory.

## The carbon film

The submitted version built its "8 Å" film by folding the 12,800 atoms of a randomly
perturbed 14 Å diamond slab into an 8 Å layer. That film has the projected density of a
2.8 nm amorphous carbon film, but an unphysical three-dimensional structure. The revision
uses a physical film instead, built by random sequential addition at 2.0 g/cm³ with a
1.4 Å minimum separation: `sim.build_sample(support=True, carbon_thickness=28,
carbon_seed=10)`. Leaving `carbon_thickness` as `None` still builds the folded film, so the
submitted sweeps stay reproducible.

Every on-carbon result in the revised paper uses the physical film. The submitted sweeps in
`outputs/` (`ss_carbon_*`, `ms_carbon_*`, `msfp_carbon_*`) are kept as the record of the
submitted version and as the folded-film comparison of Fig. S18b.

## Re-running a sweep

`run_sweep.py` simulates and reconstructs in one go. A short run, to check the setup:

```bash
uv run python scripts/run_sweep.py --out outputs/smoke --support \
    --angles 12 3 --seeds 2 --iters 10 --snapshot-every 5 --dose 1e5 --noise-seed 0
```

That takes about 30 s and should show the effect the paper is about: at 12 mrad the disks
overlap and the reconstruction is clean (global correlation ≈ 0.92, no polarity flips); at
3 mrad they do not, and regions come back shifted or contrast-reversed. It writes the same
three CSVs and `runs/*.npz` as the shipped sweeps, and `sweep.log` is just its stdout
redirected.

`--noise-seed` is the important flag: it fixes the photon-noise realisation so every
reconstruction seed inverts the *same* dataset and only the initial guess differs. Without
it each seed draws fresh noise, which measures something else entirely: whether the
information survives the dose, rather than whether one measurement determines one answer.
The paper's sweeps all use it. Add `--carbon-thickness 28` to `--support` for the physical
film.

The revision's sweeps ran on CUDA hosts. `revision_jobs.py` writes them as job lists, one
command per line (the exact invocation of every sweep in `outputs/rev/`), and
`run_queue.sh` runs a list with several workers per GPU, waiting for free GPU memory
(`gpu_wait.sh`) and disk quota. Large sweeps run in parts that `merge_parts.py` combines.
Multislice forward data come from `make_multislice_data.py`, which has a torch/CUDA
backend (`--backend torch`). The gauge search of Section 4.2 is `gauge_search_full.py`, one
job per (reconstruction, region), and the thick-slab check of Fig. S20 is
`dynamical_check.py`.

## Sweep naming

The submitted sweeps in `outputs/`:

```
ms_carbon_a3_phase_1e4
│  │      │  │     └─ dose in e/Å², or `inf` for noiseless
│  │      │  └─────── object model used in the inverse: complex | phase
│  │      └────────── convergence semi-angle in mrad
│  └───────────────── sample:  carbon (Au on the folded film) | bare (free-standing)
└──────────────────── forward model
```

| forward | |
|---|---|
| `ss` | single-slice, **matched** to the inverse model, so any failure is the reconstruction's own rather than a forward/inverse mismatch |
| `ms` | multislice |
| `msfp` | multislice, averaged over 8 frozen-phonon configurations |

The angle field appears only on the sweeps that ran one angle per directory (`ms_*`,
`msfp_*`); the `ss_*` sweeps carry all three angles in one directory, so their `runs/`
contain `a3_*`, `a6_*`, and `a12_*` side by side.

The revision's sweeps in `outputs/rev/` follow the same scheme, with `carbon28` for the
physical 2.8 nm film (realisation 10, "R0"), plus:

| sweeps | what | used by |
|---|---|---|
| `ss_carbon28_*`, `ms_carbon28_*`, `msfp_carbon28_*` | the submitted conditions on the physical film | Figs. 5, S6–S8, Table S1 |
| `film_new_s1{1,2,3}_*`, `pool_{ms,msfp}_carbon28_s1{1,2,3}_a3_*` | the same 3 mrad conditions on three more realisations of the film, pooled with R0 | Table S1, Fig. S13 |
| `film_legacy_s1{1,2}_*` | two more realisations of the folded film | Fig. S18b |
| `mm_{ms,msfp,mssrc,msfoc,msall}_{known,blind_pe}_*` | combined model mismatches at 2×10⁵ e/Å²: phonons, source size, focal spread, all three, and a probe refined from a wrong start | Fig. S13c, Table S1 |
| `mixfix_{ms,msfp}_1e5` | four-mode mixed-state object | Fig. S12 |
| `single111_*`, `scale*_bare_*`, `pair_*` | the [111] particle alone at three sizes, the sample scaled, and the [111] particle with its neighbors added one at a time | Figs. S14, S15 |
| `init_{truth,reversed,shifted}_*`, `show_init_*` | starts from the ground truth and its gauge transforms | Figs. S16, S17 |
| `dw_thick_{ss,ms}_c{10,28,50,100,200}_*`, `show_thick_*` | film thickness series in Å, scored with the film's slowly varying phase divided out | Figs. S18a, S19 |
| `samp_*` | detector binning, cropping to 192 px, and a doubled field of view | Table S2 |
| `jit_known_*`, `jit_blind_pe_*`, `grid_blind_pe_*`, `grid_blind_bare_inf` | scan positions perturbed by ±1 Å, with a known or refined probe | Table S3 |
| `gsfull_*` | the full gauge search; its candidates are in `parts/*/candidates_*.csv` | Section 4.2 |

`_pe` marks blind runs started 1500 Å defocus and 150 Å astigmatism away from the true probe
(0.6 rad rms across the 3 mrad aperture). The one exception in Table S3 is the noiseless
regular-grid row, `grid_blind_bare_inf`, which starts from a 20 Å defocus error. `show_*`
sweeps re-run chosen seeds of a sweep with the reconstructed object, snapshots, or probe
stored, for the figures that display them. `forward_data/*/meta.json` records the
parameters of each multislice forward dataset.

`ss_carbon_complex_inf` doubles as the reference sweep: its `runs/_reference.npz` supplies
the ground truth, the seven region masks, and each region's lattice basis to every figure.
`ss_bare_complex_1e5/runs/_reference.npz` does the same for the free-standing sample, and
each revision sweep carries its own. Two directories in `outputs/` are not sweeps:
`dp_cache/` holds pre-computed diffraction patterns for the panel-a insets (one per sample,
`carbon28.npz` for the physical film, from `make_dp_cache.py --carbon-thickness 28`), and
`forward_data/` records the parameters of the submitted multislice runs.

## What's committed, and what isn't

The raw datasets are far too large to version: the multislice output alone is 6.4 GB for
the submitted version and 10 GB for the revision, and the reconstructions add several GB
more. Committed instead is everything the figures and the reported statistics are computed
from.

| kept | |
|---|---|
| `summary.csv` | one row per (run, region): final polarity, offset, correlation, error |
| `drift.csv` | the same per-region metrics at every 20th iteration, which is what the sparklines and offset trails plot |
| `boundaries.csv` | twin-boundary registry error and sign flips |
| `sweep.log` | the run's console output, captured by redirecting `run_sweep.py`: geometry, per-seed timings, and polarity flips |
| `runs/*.npz` | **only** the reconstructions the paper's figures display |
| `runs/_reference.npz` | ground truth, region masks, lattice bases |

| left out | why |
|---|---|
| the multislice exit waves | `forward_data/*/meta.json` records the parameters that produced them |
| most `runs/*.npz` | every metric derived from them is in `summary.csv` and `drift.csv` |
| the per-part outputs in `parts/` | merged into the sweep's CSVs, except the gauge search's candidate tables |
| the superseded sweeps | for the submitted version, an early free-standing pass, a 30 kV comparison, and two first-pass sweeps. For the revision, the thickness series scored without dividing out the film's phase, blind runs from a 20 Å defocus error, the first mixed-object runs before each mode's global phase was fixed, the noiseless repeats of the particle and seeded series, and the reconstructions shown only in the response to the reviewers |

The revision's run files are trimmed copies holding only the arrays the figures read, with
float arrays above 1 MB stored as float16 (`fig_inputs.py manifest` logs what each figure
reads, and `fig_inputs.py pack` writes the copies). Regenerating any omitted reconstruction
means re-running the forward model and then the sweep, whose exact invocation is in
`revision_jobs.py` or in the sweep's `sweep.log`.

## Seed numbering

**The seed encodes two different things, and they matter for different reasons.**

```
seed = 100 * (Poisson draw) + (initial guess)
```

So `seed 207` is the 8th initial guess applied to the 3rd noise realisation. Runs sharing a
hundreds digit inverted the *same* photon-noise draw and differ only in where the optimiser
started, so disagreement among them is non-uniqueness. Runs in different hundreds blocks
differ in the noise itself, so disagreement between blocks is just a different dataset.

The full design is 4 draws × 10 initial guesses = 40 runs per convergence angle (12 draws
for the 2×10⁵ e/Å² mismatch sweeps). Noiseless sweeps have only one dataset, so they carry
10 runs (seeds 0–9), or 50 for the matched on-carbon case.

Because only the displayed reconstructions ship, `runs/` is sparse. The figures select seeds
via `pick_seeds`, which reads `summary.csv`, so it picks the same ones the paper used. Ask
for a seed that wasn't kept and the load fails loudly rather than silently substituting.

## Regions

Every sweep scores seven regions of the same field of view: two single-crystal particles
(`au_111`, `au_100`) and the five `<110>` wedges of a five-fold twin (`decahedron_w0`…`w4`,
labelled `[110] wedge 1`…`5` in the figures). The twin is split because it has no single
lattice: each wedge gets its own basis, with the cuts placed on the twin planes by
maximising lattice sharpness.

## Metrics

| column | meaning |
|---|---|
| `signed_corr` | correlation of the reconstructed lattice against truth; **negative means contrast-reversed** |
| `reversed` | `signed_corr < 0` |
| `ws_x`, `ws_y` | rigid offset of the reconstructed lattice, folded into the region's Wigner–Seitz cell |
| `peak_ratio` | weaker/stronger correlation extremum, i.e. how close the run came to the opposite polarity |
| `global_corr` | correlation over the whole field, i.e. reconstruction quality |
| `occupancy` | weights of the object modes, written only by `--object mixed` runs (`mixfix_*`) |

The figures outline a region in red where it is reversed and in amber where it is shifted,
i.e. its offset exceeds a quarter of the Wigner–Seitz cell (`q > 0.25`, evaluated only where
`|signed_corr| ≥ 0.5`). The submitted figures also flagged a competing correlation peak
(`peak_ratio > 0.7`) as "ambiguous"; that fires on correctly placed lattices with few
reflections, so the revision dropped it.

The headline number is the fraction of (dataset, region) groups whose ten initial guesses
fail to agree: either they disagree on polarity, or their offsets spread by more than a
quarter of the Wigner–Seitz cell. It is computed by `figpanels.nonunanimity_groups`, which
the tables and the figures all call, so they cannot drift apart. The regions of one dataset
tend to fail together, so the revised Table S1 also counts non-unanimous datasets and takes
its intervals from a bootstrap over whole datasets (`draw_level.py`).

## Environment

`pyproject.toml` and `uv.lock` pin the whole stack. Two notes:

- **py4DSTEM comes from git**, not PyPI: the released version still pins `numpy<2`, which
  abtem 1.x will not tolerate. Only `process.utils.get_maxima_2D` is used here.
- The figures were originally produced under NumPy 2.4. `analysis.cross2` replaces the 2-D
  `np.cross` overload, which NumPy deprecated in 2.0 and removed in 2.5; results are
  unchanged, and the summary table is bit-identical across both.
