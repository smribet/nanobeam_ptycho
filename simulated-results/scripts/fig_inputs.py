"""The inputs of the revision figures: which run files, and which arrays in them, they read.

    python fig_inputs.py manifest [FIGURE ...]   # run the figures, log every np.load
    python fig_inputs.py pack OUT.tar            # trimmed copies of the logged files

A FIGURE is a make_fig_revision name, or module:function for another script, e.g.
make_figures:fig_sweep.  FIG_MANIFEST=path selects another manifest, so that a second set
of figures can be packed on its own.

`manifest` runs make_fig_revision's figures (all by default) with np.load wrapped, and
writes outputs/rev/fig_inputs.json: {file: [keys read]}.  `pack` writes, for each file, a
copy holding only those keys (float arrays above 1 MB as float16), plus every summary,
drift, and boundaries CSV of the same sweeps, into a tar with paths relative to outputs/,
so it can be unpacked over the local outputs/ and the figures rendered there unchanged.
"""
import importlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

import paths

OUTS = paths.OUTPUTS
MANIFEST = Path(os.environ.get("FIG_MANIFEST", OUTS / "rev" / "fig_inputs.json"))


class Logged:
    """An NpzFile that records the keys read from it."""

    def __init__(self, z, keys):
        self._z, self._keys = z, keys

    def __getitem__(self, k):
        self._keys.add(k)
        return self._z[k]

    def __getattr__(self, a):
        return getattr(self._z, a)

    def __contains__(self, k):
        return k in self._z

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._z.close()


def manifest(names):
    log = {}
    load = np.load

    missing = set()

    class Stub:
        """Stands in for a run file that exists only on another machine, so that a figure
        still runs to the end and every other file it reads is logged."""
        def __getitem__(self, k):
            return np.zeros((340, 340), np.float32)

        def __contains__(self, k):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def logged(f, *a, **k):
        p = Path(f).resolve()
        if not p.exists() and OUTS.resolve() in p.parents:
            missing.add(str(p.relative_to(OUTS.resolve())))
            return Stub()
        z = load(f, *a, **k)
        if isinstance(z, np.lib.npyio.NpzFile) and OUTS.resolve() in p.parents:
            return Logged(z, log.setdefault(str(p.relative_to(OUTS.resolve())), set()))
        return z

    np.load = logged
    import make_fig_revision as M
    for n in names or M.FIGURES:
        print(f"=== {n}", flush=True)
        if ":" in n:
            mod, fn = n.split(":")
            getattr(importlib.import_module(mod), fn)()
        else:
            M.FIGURES[n]()
    np.load = load
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old.update({f: sorted(k) for f, k in log.items()})
    MANIFEST.write_text(json.dumps(old, indent=1))
    print(f"{len(log)} files logged, manifest has {len(old)}")
    if missing:
        print(f"{len(missing)} files not on this machine (stubbed): "
              + ", ".join(sorted(missing)[:6]) + (" ..." if len(missing) > 6 else ""))


def pack(out):
    man = json.loads(MANIFEST.read_text())
    sweeps = {Path(f).parent.parent for f in man}
    total = 0
    with tarfile.open(out, "w") as tar:
        for f, keys in sorted(man.items()):
            z = np.load(OUTS / f, allow_pickle=True)
            arr = {}
            for k in keys:
                v = z[k]
                if v.dtype.kind == "f" and v.nbytes > 1e6:
                    v = v.astype(np.float16)
                elif v.dtype.kind == "c" and v.nbytes > 1e6:
                    v = v.astype(np.complex64)
                arr[k] = v
            buf = io.BytesIO()
            np.savez_compressed(buf, **arr)
            info = tarfile.TarInfo(f)
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)
            total += info.size
        for s in sorted(sweeps):
            for c in ("summary.csv", "drift.csv", "boundaries.csv", "search.csv"):
                if (OUTS / s / c).exists():
                    tar.add(OUTS / s / c, arcname=str(s / c))
                    total += (OUTS / s / c).stat().st_size
    print(f"{len(man)} run files from {len(sweeps)} sweeps, {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    {"manifest": lambda: manifest(sys.argv[2:]), "pack": lambda: pack(sys.argv[2])}[sys.argv[1]]()
