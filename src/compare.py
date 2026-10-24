"""Build our-results vs paper-results comparison tables.

Flag rule (assignment): a cell gets a cross when |ours - paper| > 3 * sd,
using the paper's reported standard deviation for that cell.  The paper only
reports SDs for Tables 1-3 (all methods) and Table 4 (ICA-FX only); cells
without a paper SD are compared on magnitude alone (no flag possible).

Outputs:
  results/comparison.json   machine-readable
  results/comparison.md     markdown tables (source for the Word document)
"""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results", "results.json")

from paper_tables import DATASETS  # noqa: E402


def main():
    with open(RES) as f:
        raw = json.load(f)

    out = {}
    for ds_name, paper in DATASETS.items():
        out[ds_name] = []
        raw_ds = raw.get(ds_name, {})
        for method, Ms in paper.items():
            for M, (p_avg, p_sd, p_best) in Ms.items():
                key = f"{method}|{M}"
                ours = raw_ds.get(key)
                if ours is None:
                    continue
                flag = False
                if p_sd:
                    flag = abs(ours["avg"] - p_avg) > 3 * p_sd
                elif p_avg > 0:
                    # No SD published (Table 4 baselines): 3-sigma is undefined;
                    # flag any cell whose values differ by more than a factor 2
                    # (in either direction).
                    flag = (ours["avg"] > 2.0 * p_avg) or (p_avg > 2.0 * ours["avg"])
                out[ds_name].append({
                    "method": method, "M": int(M),
                    "ours": round(ours["avg"], 2), "ours_sd": round(ours["std"], 2),
                    "ours_best": round(ours["best"], 2),
                    "paper": p_avg, "paper_sd": p_sd, "paper_best": p_best,
                    "cross": bool(flag),
                })

    with open(os.path.join(HERE, "results", "comparison.json"), "w") as f:
        json.dump(out, f, indent=2)

    # markdown tables
    md = []
    for ds_name, rows in out.items():
        Ms = sorted({r["M"] for r in rows})
        methods = []
        for r in rows:
            if r["method"] not in methods:
                methods.append(r["method"])
        md.append(f"### {ds_name}\n")
        md.append("| Method | " + " | ".join(f"M={m}" for m in Ms) + " |")
        md.append("|---" * (len(Ms) + 1) + "|")
        for meth in methods:
            cells = []
            for m in Ms:
                r = next((x for x in rows if x["method"] == meth and x["M"] == m), None)
                if r is None:
                    cells.append("–")
                else:
                    mark = " ✗" if r["cross"] else ""
                    cells.append(f"{r['ours']:.2f} / {r['paper']:.2f}{mark}")
            md.append(f"| {meth} | " + " | ".join(cells) + " |")
        md.append("")
    with open(os.path.join(HERE, "results", "comparison.md"), "w") as f:
        f.write("\n".join(md))

    # console summary of crosses
    n_cross = {ds: sum(1 for r in rows if r["cross"]) for ds, rows in out.items()}
    print("crosses per dataset:", n_cross)
    for ds, rows in out.items():
        for r in rows:
            if r["cross"]:
                print(f"  {ds:9s} {r['method']:8s} M={r['M']:3d}  ours={r['ours']:.2f}  paper={r['paper']:.2f}  sd={r['paper_sd']}")


if __name__ == "__main__":
    main()
