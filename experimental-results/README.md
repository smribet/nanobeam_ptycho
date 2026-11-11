# Experimental results

4D-STEM datasets of Au nanoparticles on an amorphous carbon film, recorded on TEAM I at
300 kV with a Dectris Arina detector (192 × 192 pixels) at five convergence angles over the
same field of view, and the code that reconstructs and compares them.

| scan | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|
| convergence semi-angle (mrad) | 24 | 12 | 6 | 3 | 1.5 |

The raw data (master and data HDF5 files, 512 × 512 probe positions each) are on
[Zenodo](https://doi.org/10.5281/zenodo.22132838). The scripts expect them in
`~/data/nanobeam_ptycho/raw/` and write their reconstructions under
`~/data/nanobeam_ptycho/recons/e1/`.

```
*.ipynb     the reconstructions of the submitted version (quantEM, 50 iterations)
scripts/    the revision's re-reconstructions with three codes, and their analysis
outputs/    compact exports of those reconstructions, which the figures read
figures/    the revision's figures, and the tables the text quotes
```

## The notebooks

`scan_NN_reconstruct.ipynb` reconstructs scan NN with quantEM, first by parallax imaging
(Fig. 2a) and then iteratively (Fig. 2b). `scan_23_optimize.ipynb` refines the reciprocal
sampling on the 24 mrad scan, which the other scans reuse, and
`scan_{23,27}_iterative_seeds.ipynb` reconstruct the 24 and 1.5 mrad scans from different
random seeds, the comparison of Fig. 3 in the submitted version.

## Reproducing the figures

The figures need only what is in `outputs/` and the environment of
[`../simulated-results`](../simulated-results):

```bash
cd scripts
uv run --project ../../simulated-results python make_fig_experiment.py ../outputs/renders \
    overlay fov engines unbinned
```

| paper | what it shows | file in `figures/response/` | made by |
|---|---|---|---|
| Fig. 3 | two seeds at 24 and 1.5 mrad, and the two boxed particles overlaid | `fig_resp_fig3_overlay` | `make_fig_experiment.py overlay` |
| Fig. S1 | the HAADF images with the scanned field of view and the two particles | `fig_resp_haadf_fov` | `make_fig_experiment.py fov` |
| Fig. S9 | the blue particle from every code at every angle | `fig_resp_engines` | `make_fig_experiment.py engines` |
| Fig. S10 | binned against unbinned data at 3 and 6 mrad | `fig_resp_unbinned` | `make_fig_experiment.py unbinned` |
| Fig. S11 | the same field of view at four camera lengths | `fig_resp_camera_length_noparallax.png` | made outside these scripts |

`fig_resp_fixed_probe` (`make_fig_experiment.py fixed`) appears only in the response to the
reviewers, and its input, `e1_fixed_crops.npz` from `fixed_crops.py`, is not committed.

| input in `outputs/` | what | made by |
|---|---|---|
| `e1_phases.npz` | the final phases of quantEM seeds 0 and 1 at 24 and 1.5 mrad, full field | `e1_export.py` |
| `e1_crops.npz` | the two particles of Fig. 3 cropped from every code's reconstructions of every scan, in one display frame | `e1_crops.py` |
| `e1_qcrop.npz` | the same crops from the quantEM reconstructions of binned and unbinned data | `e1_crops.py` |
| `renders/fig3_hi.png`, `renders/haadf_hi.png` | raster renders of the submitted Fig. 3 and Fig. S1, the only form in which the published panels and the HAADF images exist; used to locate the field of view | rendered from the manuscript's figure PDFs |

The numbers the text quotes are in:

| file | what | made by |
|---|---|---|
| `figures/summary.md` | for every code, variant, and scan, the fraction of the particle area in which each seed agrees with the first, over the lattice band and over {111}+{200} and {220}; the seed columns of Table 1 and the probe-frozen, single-mode, and unbinned tests | `e1_summary.py` |
| `figures/summary_PtyRAD_cosine.md` | the same for PtyRAD with a cosine-annealed step size | `e1_summary.py "PtyRAD, cosine"` |
| `figures/compare_engines.log` | the agreement between codes, after aligning each scan on its own lattice; the last column of Table 1 | `compare_engines.py --scans 23 24 25 26 27 --register-on 23 --per-scan` |

## Re-running the reconstructions

The reconstructions ran on CUDA GPUs. Each code is one script, with settings that follow the
notebooks: detector binned 2×, the notebooks' per-scan convergence angle, defocus, and
astigmatism, four mixed-state probe modes, and one slice.

```bash
# quantEM, as in the submitted version but for 200 iterations
python reconstruct_quantem.py --scan 26 --seeds 0 1 2 3 --iters 200 --out DIR
# ... with the probe of a previous reconstruction held fixed
python reconstruct_quantem.py --scan 26 --seeds 10 11 12 --iters 200 --probe-from RUN.h5 --out DIR
# ... unbinned, on the central 256 x 256 positions (Fig. S10)
python reconstruct_quantem.py --scan 26 --seeds 0 1 --bin 1 --iters 200 --snapshot-every 50 \
    --crop 128 384 128 384 --out DIR
# an independent mixed-state ePIE on the recorded counts
python epie_independent.py --scan 26 --seeds 0 1 2 3 --iters 200
# PtyRAD (add --cosine for the cosine-annealed step size)
python run_ptyrad.py --scan 26 --seeds 0 1 2 3 --iters 100 --flipT 0 0 0 --rot 169.14
```

`e1_jobs.py` writes the job lists that were run, one command per line, for
`../simulated-results/scripts/run_queue.sh`. `e1_chain6.sh` and `ptyrad_chain.sh` ran the
later quantEM and PtyRAD jobs one after another on one GPU. Then `e1_summary.py` and
`compare_engines.py` score the reconstructions, and `e1_export.py` and `e1_crops.py` write the
exports in `outputs/`.

Reconstructions of the same dataset are compared after removing any global shift
(`register_lattice.py`): two reconstructions are registered by cross-correlating their
low-pass-filtered (envelope) images, and their lattices are then compared locally, over an
8 Å window, within the particles. Reconstructions from different codes are first brought
into a common frame, refining the rotation and translation between them on each dataset's
own lattice (`compare_engines.py`).

## Environment

The scripts use the environment of `../simulated-results` (`uv sync` there). Only
`reconstruct_quantem.py` and `run_ptyrad.py` also need the two reconstruction codes, which
the revision installed into that environment:

```bash
cd ../simulated-results && uv pip install quantem==0.1.9 ptyrad==1.0.0
```
