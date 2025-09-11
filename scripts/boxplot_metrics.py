import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORE_METRICS = [
    "peak_roll_deg",
    "peak_pitch_deg",
    "min_height",
    "t_end",
    "recovery_push1_s",
    "recovery_push2_s",
]

def load_metrics(csv_path: Path):
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def extract_numeric(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v.lower() == "nan":
            continue
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        except ValueError:
            pass
    return vals

def plot_box(metrics_dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # One figure per metric
    for key in CORE_METRICS:
        data = []
        labels = []
        n_text = []
        for cond in ["baseline", "adaptive"]:
            vals = metrics_dict.get(cond, {}).get(key, [])
            total = metrics_dict.get(cond, {}).get("_total_trials", 0)
            if vals:
                data.append(vals)
                labels.append(cond)
                n_text.append(f"n={len(vals)}/{total}")
            else:
                # Keep placeholder for alignment
                data.append([])
                labels.append(cond)
                n_text.append(f"n=0/{total}")
        fig, ax = plt.subplots(figsize=(4,4))
        # Filter out empty entirely? Keep to show missing.
        positions = np.arange(1, len(data)+1)
        plot_data = [d if d else [np.nan] for d in data]
        bp = ax.boxplot(plot_data, labels=labels, showmeans=True, meanline=True)
        # Scatter jitter
        for i, vals in enumerate(data):
            if not vals:
                continue
            x = np.random.normal(positions[i], 0.04, size=len(vals))
            ax.scatter(x, vals, s=18, alpha=0.55, color="k")
        ax.set_title(f"{key}")
        # Annotate counts
        for i, txt in enumerate(n_text):
            ax.text(positions[i], ax.get_ylim()[1], txt, ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()
        fig.savefig(outdir / f"box_{key}.png", dpi=150)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_root", required=True,
                    help="Root where batch plots placed (contains baseline/metrics.csv and/or adaptive/metrics.csv)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    root = Path(args.batch_root)
    outdir = Path(args.outdir) if args.outdir else root / "aggregates"

    metrics_dict = {}
    for cond in ["baseline", "adaptive"]:
        csv_path = root / cond / "metrics.csv"
        rows = load_metrics(csv_path)
        if not rows:
            continue
        metrics_dict[cond] = {"_total_trials": len(rows)}
        for key in CORE_METRICS:
            metrics_dict[cond][key] = extract_numeric(rows, key)

    if not metrics_dict:
        print("No metrics loaded.")
        return

    plot_box(metrics_dict, outdir)
    print(f"Saved box plots to {outdir}")

if __name__ == "__main__":
    main()