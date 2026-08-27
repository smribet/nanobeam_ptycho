"""Where things live, so a script does not care what directory it is run from.

Sweep directories are named

    {forward}_{sample}[_a{angle}]_{object}_{dose}

    forward   ss    single-slice -- matched to the inverse model, so any failure is the
                    reconstruction's own and not a forward/inverse mismatch
              ms    multislice
              msfp  multislice averaged over 8 frozen-phonon configurations
    sample    carbon | bare        Au on amorphous carbon, or free-standing
    angle     a3 | a6 | a12        convergence semi-angle in mrad; present only on the
                                   sweeps that ran one angle per directory
    object    complex | phase      the object model used in the inverse
    dose      inf | 1e6 | 1e5 | 1e4    e/A^2

Anything here that takes a sweep name accepts either the bare name or a path; bare names
are resolved against `outputs/`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
DP_CACHE = OUTPUTS / "dp_cache"
FORWARD_DATA = OUTPUTS / "forward_data"

# the noiseless matched sweep doubles as the reference: its runs/_reference.npz carries the
# ground truth, the seven region masks, and each region's lattice basis
REFERENCE = "ss_carbon_complex_inf"
REFERENCE_BARE = "ss_bare_complex_1e5"      # same, for the free-standing sample


def sweep(name):
    """Resolve a sweep directory (or a file inside outputs/), leaving absolute paths alone."""
    p = Path(name)
    return p if p.is_absolute() else OUTPUTS / p


def figures():
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES
