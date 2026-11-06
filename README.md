# Nanobeam Ptychography paper

Data and analysis code for "Real-space overlap is not enough: ambiguity in nanobeam iterative ptychography" manuscript by S.M. Ribet, M. Danaie, W.P.M. de Kleijne, A.R.C. McCray, F. Allars, C.S. Allen, C. Ophus, and G. Varnavides.

## Contents

| directory | what's in it |
|---|---|
| [`experimental-results/`](experimental-results/) | the reconstruction notebooks, the revision's re-reconstructions with three codes, and the code that makes the experimental figures |
| [`simulated-results/`](simulated-results/) | multislice simulations, reconstruction sweeps, and the code that makes the simulated figures and the SI tables |

Each directory carries its own README with the details, including which script makes each
figure and table of the paper. The raw experimental data are on
[Zenodo](https://doi.org/10.5281/zenodo.22132838).

## Revision

The revised version (Ultramicroscopy, minor revision) adds:

- every on-carbon simulation regenerated with a physically built 2.8 nm amorphous carbon
  film, in place of the submitted folded film, pooled over four realisations of the film
  at 3 mrad;
- new simulations of combined model mismatches at the experimental fluence, of finite
  particles and their neighbors, of the film thickness, of the detector sampling and the
  scan geometry, of starts from the ground truth, and of dynamical scattering in thick
  crystals;
- the experimental datasets reconstructed again with quantEM, an independent ePIE, and
  PtyRAD, and compared after removing any global shift.

The submitted version's sweeps are kept, as its record and as the folded-film comparison of
Fig. S20b, and its figures are in the history of this repository (commit `667bbeb`).
