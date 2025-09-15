import h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_series(h5_path):
    with h5py.File(h5_path, "r") as f:
        t = np.squeeze(f["recordings/time"][()]).astype(float)
        pos = f["recordings/base_pos"][()] if "recordings/base_pos" in f else None
        eul = f["recordings/base_ori_euler_xyz"][()] if "recordings/base_ori_euler_xyz" in f else None
    
    def _normalize(arr):
        if arr is None:
            return None
        a = np.array(arr)
        # Remove leading episode axis if present
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 2 and a.shape[0] == 1:
            a = a[0]
        # If shape (3, T) transpose to (T,3)
        if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
            a = a.T
        return a

    pos = _normalize(pos)
    eul = _normalize(eul)

    # Align lengths (early termination → shorter pos/eul)
    if pos is not None:
        L = min(len(t), pos.shape[0])
        t = t[:L]
        pos = pos[:L]
        if eul is not None:
            eul = eul[:L]
    elif eul is not None:
        L = min(len(t), eul.shape[0])
        t = t[:L]
        eul = eul[:L]

    return t, pos, eul

def aggregate_condition(cond_dir: Path, max_time=12.0, dt=0.002):
    # Collect all trial H5s
    h5s = sorted(cond_dir.rglob("ep=*_steps=*.h5"))
    if not h5s:
        return None
    T_ref = np.arange(0.0, max_time+1e-9, dt)
    H = []; R = []; P = []
    Xs = []; Ys = []; Yaws = []
    for h5 in h5s:
        t, pos, eul = load_series(h5)
        if pos is None or eul is None: 
            continue
        # truncate to max_time
        mask = t <= max_time + 1e-9
        t = t[mask] 
        pos = pos[mask] 
        eul = eul[mask]
        # interp onto reference (stop at last valid with NaNs afterward)
        h = np.full_like(T_ref, np.nan)
        r = np.full_like(T_ref, np.nan)
        p = np.full_like(T_ref, np.nan)
        x = np.full_like(T_ref, np.nan)
        y = np.full_like(T_ref, np.nan)
        yaw = np.full_like(T_ref, np.nan)
        h[:len(t)] = pos[:,2]
        r[:len(t)] = eul[:,0]
        p[:len(t)] = eul[:,1]
        x[:len(t)] = pos[:,0]
        y[:len(t)] = pos[:,1]
        yaw[:len(t)] = eul[:,2]
        H.append(h); R.append(r); P.append(p); Xs.append(x); Ys.append(y); Yaws.append(yaw)
    return T_ref, np.vstack(H), np.vstack(R), np.vstack(P), np.vstack(Xs), np.vstack(Ys), np.vstack(Yaws)

def plot_median_iqr(T, M, label, ylabel, outpath, pushes, ylim=None, to_deg=False):
    if M.size == 0:
        return
    if to_deg:
        M = M * 180/np.pi
    with np.errstate(all="ignore"):
        med = np.nanmedian(M, axis=0)
        q1 = np.nanpercentile(M, 25, axis=0)
        q3 = np.nanpercentile(M, 75, axis=0)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(T, med, label=label, lw=2)
    ax.fill_between(T, q1, q3, alpha=0.25)
    for (t0,d) in pushes:
        ax.axvspan(t0, t0+d, color="red", alpha=0.12, lw=0)
    ax.set_xlabel("time [s]"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim)
    ax.set_xlim([0, T[-1]])
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="experiment root (contains push_stand_baseline_* and adaptive)")
    ap.add_argument("--push", nargs=2, action="append", metavar=("t0","dur"), help="push windows")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    pushes = [(float(a), float(b)) for (a,b) in (args.push or [])]

    root = Path(args.root)
    conds = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("push_stand_baseline_"):
            conds.append(("baseline", p))
        if p.is_dir() and p.name.startswith("push_stand_adaptive_"):
            conds.append(("adaptive", p))
    if not conds:
        print("No condition dirs found"); return
    outdir = Path(args.outdir) if args.outdir else root / "plots_batch" / "aggregates"
    outdir.mkdir(parents=True, exist_ok=True)

    agg_data = {}
    for cond, cdir in conds:
        res = aggregate_condition(cdir)
        if res is None:
            continue
        agg_data[cond] = res

    # Overlay baseline vs adaptive for roll/pitch/height
    # Build comparative plots if both present
    if "baseline" in agg_data and "adaptive" in agg_data:
        T_b, H_b, R_b, P_b, X_b, Y_b, Yaw_b = agg_data["baseline"]
        T_a, H_a, R_a, P_a, X_a, Y_a, Yaw_a = agg_data["adaptive"]
        # Ensure same time reference
        T = T_b
        for (name, Mb, Ma, ylabel, fname, todeg, ylim) in [
            ("Height", H_b, H_a, "base z [m]", "height_median_iqr.png", False, (0.2,0.6)),
            ("Roll", R_b, R_a, "roll [deg]", "roll_median_iqr.png", True, (-40,40)),
            ("Pitch", P_b, P_a, "pitch [deg]", "pitch_median_iqr.png", True, (-40,40)),
            ("Yaw", Yaw_b, Yaw_a, "yaw [deg]", "yaw_median_iqr.png", True, None),
        ]:
            # Plot both medians + IQR shaded separately
            fig, ax = plt.subplots(figsize=(8,3))
            for (label, M, color) in [("baseline", Mb, "tab:red"), ("adaptive", Ma, "tab:blue")]:
                X = M * (180/np.pi) if todeg else M
                # Suppress all-NaN slice warnings
                with np.errstate(all="ignore"):
                    med = np.nanmedian(X, axis=0)
                    q1 = np.nanpercentile(X, 25, axis=0)
                    q3 = np.nanpercentile(X, 75, axis=0)
                ax.plot(T, med, label=label)
                ax.fill_between(T, q1, q3, alpha=0.20)
            for (t0,d) in pushes:
                ax.axvspan(t0, t0+d, color="red", alpha=0.12, lw=0)
            ax.set_xlabel("time [s]"); ax.set_ylabel(ylabel)
            if ylim: ax.set_ylim(ylim)
            ax.set_xlim([0, T[-1]])
            ax.grid(True); ax.legend()
            fig.tight_layout(); fig.savefig(outdir / fname, dpi=150); plt.close(fig)

    # Survival curve (prob still standing)
    # Use final time arrays: treat NaN where episodes ended earlier
    for cond, data in agg_data.items():
        T, H, R, P, X, Y, Yaw = data
        # Derive per-trial end index (last non-NaN height)
        alive = np.isfinite(H)
        # For each time column, fraction of trials with finite height
        survival = np.sum(alive, axis=0) / alive.shape[0]
        fig, ax = plt.subplots(figsize=(6,3))
        ax.step(T, survival, where="post")
        for (t0,d) in pushes:
            ax.axvspan(t0, t0+d, color="red", alpha=0.12, lw=0)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("time [s]"); ax.set_ylabel("P(alive)")
        ax.set_title(f"Survival: {cond}")
        ax.grid(True)
        fig.tight_layout(); fig.savefig(outdir / f"survival_{cond}.png", dpi=150); plt.close(fig)

    # XY final drift scatter with 95% confidence ellipses
    if "baseline" in agg_data or "adaptive" in agg_data:
        import math
        def collect_final_xy(Xmat, Ymat):
            finals = []
            for xi, yi in zip(Xmat, Ymat):
                finite_idx = np.where(np.isfinite(xi) & np.isfinite(yi))[0]
                if finite_idx.size == 0:
                    continue
                k = finite_idx[-1]
                finals.append([xi[k], yi[k]])
            return np.array(finals) if finals else np.zeros((0,2))

        def ellipse_points(mean, cov, chi2_val=5.991, n=200):
            if cov.shape != (2,2):  # fallback
                return None
            vals, vecs = np.linalg.eigh(cov)
            if np.any(vals < 0):
                vals = np.clip(vals, 0, None)
            order = np.argsort(vals)[::-1]
            vals = vals[order]; vecs = vecs[:,order]
            axes = np.sqrt(vals * chi2_val)
            theta = np.linspace(0, 2*np.pi, n)
            circle = np.stack([axes[0]*np.cos(theta), axes[1]*np.sin(theta)], axis=0)
            Rm = vecs
            pts = (Rm @ circle).T + mean
            return pts

        fig, ax = plt.subplots(figsize=(5,5))
        colors = {"baseline":"tab:red", "adaptive":"tab:blue"}
        labels_done = set()
        for cond, data in agg_data.items():
            T, H, R, P, Xmat, Ymat, Yawmat = data
            finals = collect_final_xy(Xmat, Ymat)
            if finals.shape[0] == 0:
                continue
            ax.scatter(finals[:,0], finals[:,1], s=22, alpha=0.7, color=colors.get(cond,"gray"), label=cond)
            if finals.shape[0] >= 2:
                mean = finals.mean(axis=0)
                cov = np.cov(finals.T)
                pts = ellipse_points(mean, cov)
                if pts is not None:
                    ax.plot(pts[:,0], pts[:,1], color=colors.get(cond,"gray"), lw=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title("Final base XY (95% ellipses)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout(); fig.savefig(outdir / "xy_final_scatter.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    main()