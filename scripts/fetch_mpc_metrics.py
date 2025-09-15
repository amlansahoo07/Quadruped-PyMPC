import argparse, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path: Path):
    if not path or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def basic_stats(series: pd.Series):
    s = series.dropna().astype(float)
    if s.empty:
        return dict(n=0)
    return dict(
        n=int(len(s)),
        mean=float(s.mean()),
        std=float(s.std(ddof=0)),
        median=float(s.median()),
        p95=float(s.quantile(0.95)),
        min=float(s.min()),
        max=float(s.max()),
        cov=float(s.std(ddof=0)/s.mean()) if s.mean() != 0 else math.nan
    )

def split_warmup(df: pd.DataFrame, warmup_steps: int | None = None, warmup_frac: float = 0.05):
    n = len(df)
    if n == 0:
        return df, df
    if warmup_steps is None:
        warmup_steps = max(20, int(n * warmup_frac))
    warmup_steps = min(warmup_steps, n//2)
    return df.iloc[:warmup_steps], df.iloc[warmup_steps:]

def summarize_nominal(df: pd.DataFrame, label: str):
    if df is None:
        return { "label": label, "available": False }
    warm_df, steady_df = split_warmup(df)
    out = {
        "label": label,
        "available": True,
        "total_steps": int(len(df)),
        "warmup_steps": int(len(warm_df)),
        "steady_steps": int(len(steady_df)),
        "cost_all": basic_stats(df["cost"]),
        "cost_steady": basic_stats(steady_df["cost"]) if len(steady_df)>0 else {},
        "solve_time_all_ms": basic_stats(df["solve_time_ms"]),
        "solve_time_steady_ms": basic_stats(steady_df["solve_time_ms"]) if len(steady_df)>0 else {},
    }
    # peak / steady mean ratio
    if out["cost_steady"].get("mean", 0) and df["cost"].max() and out["cost_steady"]["mean"]>0:
        out["cost_peak_to_steady_ratio"] = float(df["cost"].max()/out["cost_steady"]["mean"])
    return out

def summarize_adaptive_batch(df: pd.DataFrame):
    if df is None:
        return { "adaptive_batch_available": False }
    stats = {}
    stats["adaptive_batch_available"] = True
    stats["events"] = int(len(df))
    stats["best_cost"] = basic_stats(df["best_cost"])
    if "second_cost" in df.columns and df["second_cost"].notna().any():
        rel_margin = df["margin_rel"].replace([np.inf, -np.inf], np.nan)
        stats["margin_rel"] = basic_stats(rel_margin)
        low = (rel_margin < 0.10).sum()
        high = (rel_margin > 0.25).sum()
        stats["margin_low_count"] = int(low)
        stats["margin_high_count"] = int(high)
    if "best_idx" in df.columns:
        counts = df["best_idx"].value_counts().to_dict()
        stats["pattern_counts"] = { int(k): int(v) for k, v in counts.items() }
    return stats

def plot_cost_compare(base_df, adapt_nom_df, outdir: Path):
    if base_df is None and adapt_nom_df is None: return
    plt.figure(figsize=(8,3))
    if base_df is not None:
        plt.plot(base_df["step"], base_df["cost"], label="baseline", alpha=0.75)
    if adapt_nom_df is not None:
        plt.plot(adapt_nom_df["step"], adapt_nom_df["cost"], label="adaptive-nominal", alpha=0.75)
    plt.xlabel("step"); plt.ylabel("cost"); plt.title("MPC Cost"); plt.grid(alpha=0.3); plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outdir/"cost_compare.png", dpi=160); plt.close()

def plot_solve_time_hist(base_df, adapt_nom_df, outdir: Path):
    plt.figure(figsize=(8,3))
    bins = 40
    if base_df is not None and not base_df.empty:
        plt.hist(base_df["solve_time_ms"], bins=bins, alpha=0.5, label="baseline", density=True)
    if adapt_nom_df is not None and not adapt_nom_df.empty:
        plt.hist(adapt_nom_df["solve_time_ms"], bins=bins, alpha=0.5, label="adaptive-nominal", density=True)
    plt.xlabel("solve time [ms]"); plt.ylabel("density"); plt.title("Solve Time Distribution")
    plt.grid(alpha=0.3); plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outdir/"solve_time_hist.png", dpi=160); plt.close()

def plot_margin_hist(batch_df, outdir: Path):
    if batch_df is None or "margin_rel" not in batch_df.columns or batch_df.empty: return
    m = batch_df["margin_rel"].replace([np.inf,-np.inf], np.nan).dropna()
    if m.empty: return
    plt.figure(figsize=(6,3))
    plt.hist(m, bins=30, alpha=0.8, color="tab:blue")
    plt.xlabel("relative margin"); plt.ylabel("count"); plt.title("Adaptive Pattern Margin Distribution")
    plt.axvline(0.10, color="red", linestyle="--", linewidth=1, label="low conf 0.10")
    plt.axvline(0.25, color="green", linestyle="--", linewidth=1, label="high conf 0.25")
    plt.grid(alpha=0.3); plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outdir/"margin_hist.png", dpi=160); plt.close()

def plot_pattern_counts(batch_df, outdir: Path):
    if batch_df is None or "best_idx" not in batch_df.columns or batch_df.empty: return
    counts = batch_df["best_idx"].value_counts().sort_index()
    plt.figure(figsize=(4,3))
    counts.plot(kind="bar", color="tab:orange")
    plt.xlabel("pattern index"); plt.ylabel("count"); plt.title("Adaptive Pattern Usage")
    plt.grid(axis="y", alpha=0.3)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outdir/"pattern_counts.png", dpi=160); plt.close()

def run(baseline_dir: Path, adaptive_dir: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Locate files (assume single CSV each)
    base_nom = None
    if baseline_dir and baseline_dir.exists():
        for p in baseline_dir.glob("nominal_mpc_*.csv"):
            base_nom = p; break
    adapt_nom = adapt_batch = None
    if adaptive_dir and adaptive_dir.exists():
        for p in adaptive_dir.glob("nominal_mpc_*.csv"):
            adapt_nom = p; break
        for p in adaptive_dir.glob("adaptive_mpc_*.csv"):
            adapt_batch = p; break

    base_df = load_csv(base_nom) if base_nom else None
    adapt_nom_df = load_csv(adapt_nom) if adapt_nom else None
    adapt_batch_df = load_csv(adapt_batch) if adapt_batch else None

    summary = {
        "baseline_nominal": summarize_nominal(base_df, "baseline_nominal"),
        "adaptive_nominal": summarize_nominal(adapt_nom_df, "adaptive_nominal"),
        "adaptive_batch": summarize_adaptive_batch(adapt_batch_df),
    }

    # Plots
    plot_cost_compare(base_df, adapt_nom_df, outdir)
    plot_solve_time_hist(base_df, adapt_nom_df, outdir)
    plot_margin_hist(adapt_batch_df, outdir)
    plot_pattern_counts(adapt_batch_df, outdir)

    # Save summary JSON
    with open(outdir/"summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plain text quick view
    with open(outdir/"summary.txt", "w") as fh:
        for k, v in summary.items():
            fh.write(f"== {k} ==\n")
            fh.write(json.dumps(v, indent=2)+"\n\n")

    print(f"[OK] Wrote stats + plots to {outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, help="Directory containing baseline nominal_mpc_*.csv")
    ap.add_argument("--adaptive_dir", required=False, help="Directory containing adaptive nominal/adaptive_mpc_*.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for plots and summary")
    args = ap.parse_args()
    run(Path(args.baseline_dir), Path(args.adaptive_dir) if args.adaptive_dir else None, Path(args.outdir))

if __name__ == "__main__":
    main()