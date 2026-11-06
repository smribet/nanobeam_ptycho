"""Summary of the full gauge search (gauge_search_full.py): one line per (initial guess,
edited region), and one table per condition.

    python gsfull_results.py            # reads outputs/rev/gsfull_*/parts/*/candidates_*.csv

A region is wrong after re-convergence if it is reversed or its lattice offset is more than a
quarter of a cell (q > 0.25).  The reference is the unedited object re-converged for the same
number of iterations (polarity +1, no shift), so every candidate has had identical treatment.
Per candidate: whether all seven regions are correct (exact), the wrong-region count (an upper
bound: candidates.csv counts a region of the others that is both reversed and shifted twice),
and whether the edited region ended in the same state as without the edit ("undone").
Writes figures/gsfull_results.{md,csv}.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "outputs/rev"
CONDS = (("free-standing, noiseless", "gsfull_ss_bare_inf"),
         ("free-standing, 1e5", "gsfull_ss_bare_1e5"),
         ("2.8 nm a-C, 1e4", "gsfull_ss_carbon28_1e4"))


def state(r):
    return (r["region_reversed"] == "True", float(r["region_q"]) > 0.25)


def wrong(r):
    return int(r["region_reversed"] == "True" or float(r["region_q"]) > 0.25) \
        + int(r["others_reversed"]) + int(r["others_shifted"])


def load(sweep):
    out = []
    for f in sorted((REV / sweep / "parts").glob("*/candidates_*.csv")):
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 8:
            print(f"incomplete: {f} ({len(rows)} rows)")
            continue
        for r in rows:
            r["wrong"] = wrong(r)
            r["misfit"] = float(r["misfit"])
        start = next(r for r in rows if r["polarity"] == "1" and float(r["fu"]) == 0
                     and float(r["fv"]) == 0)
        best = min(rows, key=lambda r: r["misfit"])
        nw = np.array([r["wrong"] for r in rows])
        m = np.array([r["misfit"] for r in rows])
        ok = nw == 0
        undone = [state(r) == state(start) for r in rows if r is not start]
        # lowest misfit among all-correct candidates against the lowest among the others
        margin = ((m[~ok].min() - m[ok].min()) / start["misfit"] if ok.any() and (~ok).any()
                  else np.nan)
        out.append(dict(
            seed=int(rows[0]["seed"]), region=rows[0]["region"],
            start_wrong=start["wrong"], start_misfit=start["misfit"],
            min_wrong=int(nw.min()), n_correct=int((nw == 0).sum()),
            best_wrong=best["wrong"],
            best_is_min=best["wrong"] == nw.min(),
            spread=(m.max() - m.min()) / m.min(),
            # misfit of the most-correct candidate relative to the lowest misfit
            gap=(m[nw == nw.min()].min() - m.min()) / m.min(),
            wrongs=nw.tolist(), n_ok=int(ok.sum()), best_ok=bool(best["wrong"] == 0),
            undone=int(sum(undone)), margin=margin))
    return out


def main():
    lines, rows_csv = [], []
    for label, sweep in CONDS:
        res = load(sweep)
        if not res:
            continue
        lines += [f"\n### {label} ({len(res)} edits)\n",
                  "| guess | region | wrong (upper bound) after each candidate | all-correct "
                  "candidates | lowest misfit all-correct | edit undone (of 7) | margin | "
                  "spread |", "|---|---|---|---|---|---|---|---|"]
        for r in res:
            lines.append(f"| {r['seed']} | {r['region']} | "
                         f"{' '.join(map(str, r['wrongs']))} | {r['n_ok']} | "
                         f"{'yes' if r['best_ok'] else 'no'} | {r['undone']} | "
                         f"{100 * r['margin']:+.3f}% | {100 * r['spread']:.3f}% |")
            rows_csv.append(dict(condition=label, **{k: v for k, v in r.items()
                                                     if k != "wrongs"},
                                 wrongs=" ".join(map(str, r["wrongs"]))))
        any_correct = sum(r["min_wrong"] == 0 for r in res)
        best_ok = sum(r["best_is_min"] for r in res)
        best_zero = sum(r["best_wrong"] == 0 for r in res)
        improved = sum(r["min_wrong"] < r["start_wrong"] for r in res)
        start_bad = sum(r["start_wrong"] > 0 for r in res)
        for kind, sel in (("particle", lambda r: r["region"].startswith("au")),
                          ("wedge", lambda r: r["region"].startswith("deca"))):
            g = [r for r in res if sel(r)]
            mg = [r["margin"] for r in g if np.isfinite(r["margin"])]
            lines.append(f"- {kind} edits: an all-correct candidate in "
                         f"{sum(r['n_ok'] > 0 for r in g)}/{len(g)}, lowest misfit all-correct "
                         f"in {sum(r['best_ok'] for r in g)}/{len(g)}, edit undone in "
                         f"{sum(r['undone'] for r in g)}/{7 * len(g)} candidates"
                         + (f", margin {100 * min(mg):+.3f}% to {100 * max(mg):+.3f}%"
                            if mg else ""))
        lines += ["", f"- re-converged start wrong in {start_bad}/{len(res)}; some candidate "
                  f"reduces the wrong regions in {improved}/{start_bad}; some candidate "
                  f"reaches all-correct in {any_correct}/{len(res)}",
                  f"- lowest misfit picks the most-correct candidate in {best_ok}/{len(res)} "
                  f"and an all-correct one in {best_zero}/{len(res)}",
                  f"- misfit spread across candidates: median "
                  f"{100 * np.median([r['spread'] for r in res]):.2f}%, max "
                  f"{100 * max(r['spread'] for r in res):.2f}%"]
    txt = "\n".join(lines)
    print(txt)
    fig = ROOT / "figures"
    (fig / "gsfull_results.md").write_text(txt + "\n")
    with open(fig / "gsfull_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
