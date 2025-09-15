# import argparse
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# OBS_NAMES = [
#     "base_pos",
#     "base_ori_euler_xyz",
#     "base_lin_vel",
#     "base_ang_vel",
#     "feet_pos",
#     "feet_vel",
#     "contact_state",
#     "time",
# ]

# def get_dataset(h5, name):
#     # Prefer exact path under 'recordings/'
#     if f"recordings/{name}" in h5:
#         return h5[f"recordings/{name}"][()]
#     if name in h5:  # fallback (unlikely here)
#         return h5[name][()]
#     return None

# def walk_datasets(h5):
#     out = []
#     def _rec(g, p=""):
#         for k, v in g.items():
#             q = f"{p}/{k}" if p else k
#             if isinstance(v, h5py.Dataset):
#                 out.append(q)
#             elif isinstance(v, h5py.Group):
#                 _rec(v, q)
#     _rec(h5)
#     return out

# def find_any(h5, names):
#     all_ds = walk_datasets(h5)
#     for n in names:
#         if n in h5:  # absolute
#             return n, h5[n][()]
#         # match by basename
#         for ds in all_ds:
#             if ds.split("/")[-1] == n:
#                 return ds, h5[ds][()]
#     return None, None

# def squeeze(a): return np.squeeze(np.asarray(a))
# def to_deg(rad): return np.asarray(rad) * 180.0 / np.pi

# def shaded_push(ax, pushes):
#     for (t0, dur) in pushes:
#         ax.axvspan(t0, t0+dur, color="red", alpha=0.12, lw=0)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("h5", type=str, help="Path to H5 (ep=1_steps=*.h5)")
#     ap.add_argument("--push", nargs=2, action="append", metavar=("t0", "dur"),
#                     help="Add push window (start_s duration_s). Repeatable.")
#     ap.add_argument("--outdir", type=str, default=None, help="Output directory for PNGs")
#     args = ap.parse_args()

#     h5_path = Path(args.h5)
#     outdir = Path(args.outdir) if args.outdir else h5_path.parent
#     outdir.mkdir(parents=True, exist_ok=True)
#     pushes = [(float(t0), float(d)) for (t0, d) in (args.push or [])]

#     with h5py.File(h5_path, "r") as f:
#         # List datasets
#         print("Datasets:")
#         for ds in walk_datasets(f):
#             print(" -", ds, f[ds].shape)

#         # Time
#         T = get_dataset(f, "time")
#         if T is None:
#             print("Missing time dataset; abort.")
#             return
#         t = squeeze(T).astype(float).reshape(-1)

#         # Base
#         base_pos = get_dataset(f, "base_pos")
#         base_eul = get_dataset(f, "base_ori_euler_xyz")
#         feet_pos = get_dataset(f, "feet_pos")
#         contact_state = get_dataset(f, "contact_state")

#         # 1) Height (base z)
#         if base_pos is not None:
#             bp = squeeze(base_pos)
#             fig, ax = plt.subplots(figsize=(8,3))
#             ax.plot(t[:bp.shape[0]], bp[:len(t), 2], label="base z")
#             for (t0, dur) in pushes:
#                 ax.axvspan(t0, t0+dur, color="red", alpha=0.12, lw=0)
            
#             # Set fixed axis limits
#             ax.set_xlim([0, 12])  # x-axis from 0 to 12 seconds
#             ax.set_ylim([0.2, 0.6])   # Fixed height range, adjust as needed
            
#             ax.set_xlabel("time [s]"); ax.set_ylabel("height [m]")
#             ax.grid(True); ax.legend()
#             fig.tight_layout(); fig.savefig(outdir/"height_z.png", dpi=150); plt.close(fig)
#         else:
#             print("Warn: no base_pos; skipping height plot")

#         # 2) Roll/Pitch/Yaw
#         if base_eul is not None:
#             eul = squeeze(base_eul)
#             fig, ax = plt.subplots(figsize=(8,3))
#             ax.plot(t[:eul.shape[0]], to_deg(eul[:len(t), 0]), label="roll")
#             ax.plot(t[:eul.shape[0]], to_deg(eul[:len(t), 1]), label="pitch")
#             ax.plot(t[:eul.shape[0]], to_deg(eul[:len(t), 2]), label="yaw")
#             shaded_push(ax, pushes)

#             ax.set_xlim([0, 12])  # x-axis from 0 to 12 seconds
#             ax.set_ylim([-40, 40])  # Fixed yaw range, adjust as needed

#             ax.set_xlabel("time [s]"); ax.set_ylabel("deg")
#             ax.grid(True); ax.legend(ncol=3, fontsize=8)
#             fig.tight_layout(); fig.savefig(outdir/"base_rpy.png", dpi=150); plt.close(fig)
#         else:
#             print("Warn: no base orientation; skipping RPY plot")

#         # 3) Base XY path
#         if base_pos is not None:
#             bp = squeeze(base_pos)
#             fig, ax = plt.subplots(figsize=(4,4))
#             ax.plot(bp[:,0], bp[:,1], lw=1.0)
#             ax.plot([0],[0],"k+")
#             ax.set_aspect("equal")

#             # Set fixed axis limits
#             ax.set_xlim([-0.5, 0.5])  # Fixed x-axis range, adjust as needed
#             ax.set_ylim([-0.5, 0.5])  # Fixed y-axis range, adjust as needed

#             ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("Base XY")
#             ax.grid(True)
#             fig.tight_layout(); fig.savefig(outdir/"xy_path.png", dpi=150); plt.close(fig)
#         else:
#             print("Warn: no base_pos; skipping XY path plot")

#         # 4) Contacts
#         if contact_state is not None:
#             C = squeeze(contact_state).astype(float)  # (T,4)
#             if C.ndim == 2 and C.shape[1] == 4:
#                 fig, ax = plt.subplots(figsize=(8,3))
#                 labels = ["FL","FR","RL","RR"]
#                 # Reverse the plotting order
#                 for j in range(4):
#                     # Plot in reverse order (3-j instead of j)
#                     ax.step(t[:C.shape[0]], (3-j) + 0.9*C[:len(t), j], where="post", label=labels[j])
#                 for (t0, dur) in pushes:
#                     ax.axvspan(t0, t0+dur, color="red", alpha=0.12, lw=0)

#                 # Set fixed axis limits
#                 ax.set_xlim([-0.5, 12.5])  # Fixed x-axis range, adjust as needed
#                 ax.set_ylim([-0.5, 4.5])  # Fixed y-axis range, adjust as needed

#                 ax.set_xlabel("time [s]")
#                 ax.set_yticks(range(4)) 
#                 # Use reversed labels for y-ticks
#                 ax.set_yticklabels(labels[::-1])
#                 ax.grid(True, axis="x")
#                 fig.tight_layout(); fig.savefig(outdir/"contacts.png", dpi=150); plt.close(fig)
#         else:
#             print("Warn: no contact_state; skipping contacts plot")

#         print(f"Saved plots to: {outdir}")

# if __name__ == "__main__":
#     main()

import argparse
import h5py
import numpy as np
import matplotlib
# Use non-interactive backend to avoid Qt/Wayland plugin issues in batch mode
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Recovery definition parameters
RECOVERY_ROLL_THRESH_DEG = 5.0
RECOVERY_PITCH_THRESH_DEG = 5.0
RECOVERY_HOLD_SEC = 0.20

OBS_NAMES = [
    "base_pos",
    "base_ori_euler_xyz",
    "base_lin_vel",
    "base_ang_vel",
    "feet_pos",
    "feet_vel",
    "contact_state",
    "time",
]

def get_dataset(h5, name):
    if f"recordings/{name}" in h5:
        return h5[f"recordings/{name}"][()]
    if name in h5:
        return h5[name][()]
    return None

def walk_datasets(h5):
    out = []
    def _rec(g, p=""):
        for k, v in g.items():
            q = f"{p}/{k}" if p else k
            if isinstance(v, h5py.Dataset):
                out.append(q)
            elif isinstance(v, h5py.Group):
                _rec(v, q)
    _rec(h5)
    return out

def squeeze(a): return np.squeeze(np.asarray(a))
def to_deg(rad): return np.asarray(rad) * 180.0 / np.pi

def shaded_push(ax, pushes):
    for (t0, dur) in pushes:
        ax.axvspan(t0, t0+dur, color="red", alpha=0.12, lw=0)

def plot_single_file(h5_path: Path, pushes, outdir: Path,
                     fixed_time_xlim=(0, 12),
                     fixed_height_ylim=(0.2, 0.6),
                     fixed_rpy_ylim=(-40, 40),
                     fixed_xy_xlim=(-0.5, 0.5),
                     fixed_xy_ylim=(-0.5, 0.5),
                     fixed_contacts_xlim=(-0.5, 12.5)):
    outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        # Time
        T = get_dataset(f, "time")
        if T is None:
            print(f"[WARN] Missing time in {h5_path}")
            return None
        t = squeeze(T).astype(float).reshape(-1)
        if t.size == 0:
            print(f"[WARN] Empty time in {h5_path}")
            return None

        base_pos = get_dataset(f, "base_pos")
        base_eul = get_dataset(f, "base_ori_euler_xyz")
        contact_state = get_dataset(f, "contact_state")

        metrics = {
            "file": str(h5_path),
            "peak_roll_deg": np.nan,
            "peak_pitch_deg": np.nan,
            "min_height": np.nan,
            "final_height": np.nan,
            "t_end": float(t[-1]),
        }

        # 1) Height (base z)
        if base_pos is not None:
            bp = squeeze(base_pos)
            if bp.ndim == 1:  # single vector fallback
                bp = bp.reshape(1, -1)
            metrics["min_height"] = float(np.min(bp[:, 2]))
            metrics["final_height"] = float(bp[-1, 2])
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:bp.shape[0]], bp[:len(t), 2], label="base z")
            shaded_push(ax, pushes)
            if fixed_time_xlim: ax.set_xlim(fixed_time_xlim)
            if fixed_height_ylim: ax.set_ylim(fixed_height_ylim)
            ax.set_xlabel("time [s]"); ax.set_ylabel("height [m]")
            ax.grid(True); ax.legend()
            fig.tight_layout(); fig.savefig(outdir / "height_z.png", dpi=150); plt.close(fig)
        else:
            print(f"[INFO] No base_pos in {h5_path}")

        # 2) Roll/Pitch/Yaw
        if base_eul is not None:
            eul = squeeze(base_eul)
            if eul.ndim == 1:
                eul = eul.reshape(1, -1)
            roll_deg = to_deg(eul[:len(t), 0])
            pitch_deg = to_deg(eul[:len(t), 1])
            metrics["peak_roll_deg"] = float(np.max(np.abs(roll_deg)))
            metrics["peak_pitch_deg"] = float(np.max(np.abs(pitch_deg)))

            # --- Per-push recovery times ---
            if pushes:
                dt_est = (t[1]-t[0]) if len(t) > 1 else 0.002
                hold_steps = max(1, int(RECOVERY_HOLD_SEC / dt_est))
                for idx_push, (p_start, p_dur) in enumerate(pushes, start=1):
                    p_end = p_start + p_dur
                    start_i = int(np.searchsorted(t, p_end, side="left"))
                    rec_time = np.nan
                    if start_i < len(t):
                        for k in range(start_i, len(t)):
                            w_end = k + hold_steps
                            if w_end > len(t):
                                break
                            if (np.all(np.abs(roll_deg[k:w_end]) < RECOVERY_ROLL_THRESH_DEG) and
                                np.all(np.abs(pitch_deg[k:w_end]) < RECOVERY_PITCH_THRESH_DEG)):
                                rec_time = float(t[k] - p_end)
                                break
                    metrics[f"recovery_push{idx_push}_s"] = rec_time

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(t[:eul.shape[0]], roll_deg, label="roll")
            ax.plot(t[:eul.shape[0]], pitch_deg, label="pitch")
            ax.plot(t[:eul.shape[0]], to_deg(eul[:len(t), 2]), label="yaw")
            shaded_push(ax, pushes)
            if fixed_time_xlim: ax.set_xlim(fixed_time_xlim)
            if fixed_rpy_ylim: ax.set_ylim(fixed_rpy_ylim)
            ax.set_xlabel("time [s]"); ax.set_ylabel("deg")
            ax.grid(True); ax.legend(ncol=3, fontsize=8)
            fig.tight_layout(); fig.savefig(outdir / "base_rpy.png", dpi=150); plt.close(fig)
        else:
            print(f"[INFO] No base_ori_euler_xyz in {h5_path}")

        # 3) Base XY path
        if base_pos is not None:
            bp = squeeze(base_pos)
            if bp.ndim == 1:
                bp = bp.reshape(1, -1)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(bp[:, 0], bp[:, 1], lw=1.0)
            ax.plot([0], [0], "k+")
            ax.set_aspect("equal")
            if fixed_xy_xlim: ax.set_xlim(fixed_xy_xlim)
            if fixed_xy_ylim: ax.set_ylim(fixed_xy_ylim)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("Base XY")
            ax.grid(True)
            fig.tight_layout(); fig.savefig(outdir / "xy_path.png", dpi=150); plt.close(fig)

        # 4) Contacts
        if contact_state is not None:
            C = squeeze(contact_state).astype(float)
            if C.ndim == 1:
                C = C.reshape(-1, 1)
            if C.ndim == 2 and C.shape[1] == 4:
                fig, ax = plt.subplots(figsize=(8, 3))
                labels = ["FL", "FR", "RL", "RR"]
                for j in range(4):
                    ax.step(t[:C.shape[0]], (3 - j) + 0.9 * C[:len(t), j], where="post", label=labels[j])
                shaded_push(ax, pushes)
                if fixed_contacts_xlim: ax.set_xlim(fixed_contacts_xlim)
                ax.set_ylim([-0.5, 4.5])
                ax.set_xlabel("time [s]")
                ax.set_yticks(range(4))
                ax.set_yticklabels(labels[::-1])
                ax.grid(True, axis="x")
                fig.tight_layout(); fig.savefig(outdir / "contacts.png", dpi=150); plt.close(fig)

        return metrics

def write_metrics_csv(metrics_list, csv_path: Path):
    if not metrics_list:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics_list[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in metrics_list:
            w.writerow(row)

def aggregate_metrics(metrics_list):
    agg = {}
    if not metrics_list:
        return agg
    # Exclude non-numeric/meta keys
    candidate_keys = [k for k in metrics_list[0].keys() if k not in ("file", "trial")]
    for k in candidate_keys:
        vals_raw = [m[k] for m in metrics_list]
        # Keep only numeric finite values
        vals = [float(v) for v in vals_raw if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
        if not vals:
            continue
        vals_np = np.asarray(vals, dtype=float)
        agg[k] = {
            "mean": float(np.mean(vals_np)),
            "std": float(np.std(vals_np)),
            "min": float(np.min(vals_np)),
            "max": float(np.max(vals_np)),
            "n": len(vals),
        }
    agg["_total_trials"] = len(metrics_list)
    return agg

def read_outcomes(outcomes_csv: Path):
    if not outcomes_csv.exists():
        return None
    successes = 0
    total = 0
    with open(outcomes_csv, "r", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            total += 1
            successes += int(row.get("success", 0))
    return {"successes": successes, "total": total, "rate": successes / total if total else 0.0}

def save_summary(summary_path: Path, condition_results: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        for cond, info in condition_results.items():
            fh.write(f"== {cond} ==\n")
            if "outcomes" in info and info["outcomes"]:
                o = info["outcomes"]
                fh.write(f"Success: {o['successes']}/{o['total']} ({o['rate']*100:.1f}%)\n")
            if "agg" in info:
                 agg = info["agg"]
                 total_trials = agg.get("_total_trials", 0)
                 fh.write("Metrics (mean ± std) [min, max] (n / trials):\n")
                 # Print core k/v except internal markers
                 ordered = [k for k in agg.keys() if not k.startswith("_") and k not in ("t_end","peak_roll_deg","peak_pitch_deg","min_height","final_height") ]
                 # Keep core metrics first in a fixed order
                 core_order = ["peak_roll_deg","peak_pitch_deg","min_height","final_height","t_end"]
                 for k in core_order + sorted([x for x in ordered if x not in core_order]):
                     stats = agg.get(k)
                     if not stats or not isinstance(stats, dict):
                         continue
                     n = stats.get("n", 0)
                     fh.write(f"  {k}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                              f"[{stats['min']:.3f}, {stats['max']:.3f}]  ({n}/{total_trials})\n")
                 # Recovery rate summary grouped by pushes
                 rec_keys = [k for k in agg.keys() if k.startswith("recovery_push")]
                 if rec_keys:
                     fh.write("  Recovery rates:\n")
                     for rk in sorted(rec_keys):
                         stats = agg[rk]
                         n = stats.get("n", 0)
                         rate = (n / total_trials * 100.0) if total_trials else 0.0
                         fh.write(f"    {rk}: {n}/{total_trials} recovered ({rate:.1f}%) "
                                  f"mean={stats['mean']:.3f}s\n")
            fh.write("\n")

def batch_process(root: Path, pushes, outdir: Path):
    outdir = outdir or (root / "plots_batch")
    outdir.mkdir(parents=True, exist_ok=True)
    # Find condition directories
    condition_dirs = []
    for p in root.iterdir():
        if p.is_dir() and (p.name.startswith("push_stand_baseline_") or p.name.startswith("push_stand_adaptive_")):
            condition = "baseline" if "baseline" in p.name else "adaptive"
            condition_dirs.append((condition, p))
    condition_results = {}
    for condition, cdir in condition_dirs:
        print(f"[Batch] Condition={condition} dir={cdir}")
        metrics_all = []
        # outcomes
        outcomes = read_outcomes(cdir / "outcomes.csv")
        # trials
        for trial_dir in sorted(cdir.glob("trial_*")):
            if not trial_dir.is_dir():
                continue
            # find h5
            h5_files = list(trial_dir.rglob("ep=*_steps=*.h5"))
            if not h5_files:
                print(f"[WARN] No H5 in {trial_dir}")
                continue
            # usually one, process all just in case
            for h5f in h5_files:
                trial_name = trial_dir.name
                out_trial = outdir / condition / trial_name
                metrics = plot_single_file(h5f, pushes, out_trial)
                if metrics:
                    metrics["trial"] = trial_name
                    metrics_all.append(metrics)
        # write metrics CSV
        metrics_csv = outdir / condition / "metrics.csv"
        write_metrics_csv(metrics_all, metrics_csv)
        agg = aggregate_metrics(metrics_all)
        condition_results[condition] = {"agg": agg, "outcomes": outcomes}
    # summary
    save_summary(outdir / "summary.txt", condition_results)
    print(f"[Batch] Done. Summary: {outdir / 'summary.txt'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", nargs="?", type=str,
                    help="Single H5 file path (ep=*_steps=*.h5). Omit when using --root batch mode.")
    ap.add_argument("--root", type=str,
                    help="Experiment root containing push_stand_baseline_* and push_stand_adaptive_* directories.")
    ap.add_argument("--push", nargs=2, action="append", metavar=("t0", "dur"),
                    help="Push window (start_s duration_s). Repeatable.")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory root for plots.")
    args = ap.parse_args()

    pushes = [(float(t0), float(d)) for (t0, d) in (args.push or [])]

    if args.root:
        if args.h5:
            ap.error("Provide either a single H5 path or --root, not both.")
        root = Path(args.root)
        if not root.exists():
            ap.error(f"Root path does not exist: {root}")
        outdir = Path(args.outdir) if args.outdir else root / "plots_batch"
        batch_process(root, pushes, outdir)
        return

    # Single-file mode
    if not args.h5:
        ap.error("Must provide an H5 path in single-file mode or use --root for batch mode.")
    h5_path = Path(args.h5)
    if not h5_path.exists():
        ap.error(f"H5 path does not exist: {h5_path}")
    outdir = Path(args.outdir) if args.outdir else h5_path.parent
    metrics = plot_single_file(h5_path, pushes, outdir)
    if metrics:
        print("Metrics:", metrics)
    print(f"Saved plots to: {outdir}")

if __name__ == "__main__":
    main()