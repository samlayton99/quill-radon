"""The learned profiles as one gif per case: the channel tensor C[M, N, K] (coefficient against center, an M x K grid)
and the readout's coefficients (one row, K panels), during training from zero profiles.

Frames: 60 evolving frames whose steps follow a smooth speedup, step(f) = steps * (f / 59)^2.5 (frame 10 is step 23,
frame 30 is step 360, frame 45 is step 1000), at 250 ms each = 15 s; then the final structure repeated for 40 frames
= 10 s. Uniform frame durations, so every viewer plays it the same way.

    uv run --extra dev python experiments/expI02_block_prior/profile_gif.py --target gauss_bump --name good
    uv run --extra dev python experiments/expI02_block_prior/profile_gif.py --rerender <snaps.npz> --name bad
"""
import argparse
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qi2      # noqa: E402
import tasks    # noqa: E402

OUT = tasks.REPO / "results" / "checkpoint_I_depth_theory" / "expI02_block_prior" / "figures"
DATA = tasks.REPO / "results" / "checkpoint_I_depth_theory" / "expI02_block_prior" / "data" / "profiles"
N_EVOLVE, N_STILL, FRAME_MS, POWER = 60, 40, 250, 2.5


def frame_steps(total):
    return sorted(set(int(round(total * (f / (N_EVOLVE - 1)) ** POWER)) for f in range(N_EVOLVE)))


def train_with_snapshots(target, d, M, N1, K, N2, steps, lr, beta, seed, n_train, head, rcond=1e-14, init="zero", coords="raw"):
    torch.set_default_dtype(torch.float64)
    D = tasks.analytic(target, d, n_train, seed=seed)
    net = qi2.make_block(d, M, N1, K, N2, coords=coords, seed=seed)
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        if init == "zero":
            net.layers[0].mixer.theta.zero_()                                   # profiles from zero
            net.layers[0].mixer.bias.copy_(0.1 * torch.randn(K, generator=g))   # distinct channel biases break the symmetry
            p = D["Xtr"] @ net.layers[0].V.T                                     # projection calibrated; the zero mixer left alone
            mu, sd = p.mean(0), p.std(0, unbiased=False).clamp_min(1e-12)
            net.layers[0].V.mul_((qi2.S_STAR / sd).view(-1, 1)); net.layers[0].bV.copy_(-mu * qi2.S_STAR / sd)
        else:
            qi2.calibrate(net, D["Xtr"])                                        # the A1 recipe: smooth profiles, calibrated
        net.layers[-1].mixer.theta.zero_(); net.layers[-1].mixer.bias.fill_(float(D["Ytr"].mean()))
    Y = D["Ytr"].view(-1, 1)
    nl = net.nonlinear_params()
    params = nl + (net.readout() if head == "trained" else [])
    if head == "varpro":
        net.solve_readout(D["Xtr"], Y, rcond)
    opt = torch.optim.Adam(params, lr=lr)
    want = set(frame_steps(steps))
    snaps = []

    def take(t):
        with torch.no_grad():
            err = qi2.rel_l2(net(D["Xte"])[:, 0], D["Yte"])
            try:
                err_solved = qi2.refit_eval(net, D["Xtr"], Y, D["Xte"], D["Yte"].view(-1, 1))
            except Exception:
                err_solved = float("nan")                                        # the QR+SVD refit can fail on a degenerate head
            C = net.layers[0].mixer.coeffs().detach().cpu().numpy().copy()            # [M, N1, K]
            H = net.layers[-1].mixer.coeffs().detach().cpu().numpy()[:, :, 0].copy()   # [K, N2]
        snaps.append(dict(step=t, err=err, err_solved=err_solved, C=C, head=H))
    take(0)
    t0 = time.time()
    for t in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, t / 50) * 0.5 * (1 + math.cos(math.pi * (t - 1) / steps))
        opt.zero_grad(set_to_none=True)
        out, pres = net.forward_pres(D["Xtr"], solve=((Y, rcond) if head == "varpro" else None))
        loss = ((out - Y) ** 2).mean() + beta * qi2.band_penalty(pres)
        loss.backward(); opt.step()
        if t in want:
            take(t)
            if t % 250 == 0 or t == steps:
                print(f"  step {t:5d}  test {snaps[-1]['err']:.2e}  (head solved {snaps[-1]['err_solved']:.2e})  [{time.time() - t0:.0f}s]", flush=True)
    return net, snaps


def render(snaps, c1, c2, M, K, dirs, out_gif, title, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Cmax = np.max(np.abs(np.stack([s["C"] for s in snaps])), axis=(0, 1, 2))       # per channel, over frames, directions, centers
    fig, axes = plt.subplots(M + 1, K, figsize=(2.05 * K, 1.55 * (M + 1)), sharex=True)
    frames = []
    # the final structure is held for N_STILL frames; each carries a visible countdown so no viewer merges them
    seq = list(snaps) + [dict(snaps[-1], hold=N_STILL - j) for j in range(N_STILL)]
    for i, s in enumerate(seq):
        for r in range(M):
            for k in range(K):
                ax = axes[r, k]; ax.cla()
                ax.axhline(0, color="0.85", lw=0.6)
                ax.plot(c1, s["C"][r, :, k], "-", color="#d62728", lw=1.0)
                lim = max(Cmax[k], 1e-12) * 1.1
                ax.set_ylim(-lim, lim); ax.set_xlim(-1, 1)
                ax.tick_params(labelsize=6, length=2)
                if r == 0:
                    ax.set_title(f"channel {k}", fontsize=8)
                if k == 0:
                    ax.set_ylabel(f"direction {r}\n({', '.join(f'{v:+.2f}' for v in dirs[r])})", fontsize=6.5)
                else:
                    ax.set_yticklabels([])
        hmax = max(float(np.abs(s["head"]).max()), 1e-12)
        for k in range(K):
            ax = axes[M, k]; ax.cla()
            ax.axhline(0, color="0.85", lw=0.6)
            ax.plot(c2, s["head"][k], "-", color="#1f77b4", lw=1.0)
            ax.set_ylim(-1.1 * hmax, 1.1 * hmax); ax.set_xlim(-1, 1); ax.tick_params(labelsize=6, length=2)
            ax.set_xlabel("center", fontsize=7)
            if k == 0:
                ax.set_ylabel(f"readout coefficients\nmax |w| = {hmax:.1e}", fontsize=6.5)
            else:
                ax.set_yticklabels([])
        hold = f"    FINAL STRUCTURE, holding {s['hold'] * FRAME_MS / 1000:.2f} s" if "hold" in s else ""
        fig.suptitle(f"{title}    step {s['step']:4d} of {snaps[-1]['step']}    test rel L2 {s['err']:.1e}"
                     f"    (head re-solved: {s['err_solved']:.1e}){hold}", fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        if i == len(snaps) - 1:
            fig.savefig(str(out_gif).replace(".gif", "_final.png"), dpi=150)     # the final structure as a still
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    plt.close(fig)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    return len(frames)


def pick_frames(snaps, steps):
    """From an arbitrary snapshot set, the frames nearest the speedup schedule (for re-rendering older runs)."""
    avail = np.array([s["step"] for s in snaps])
    idx = sorted(set(int(np.argmin(np.abs(avail - t))) for t in frame_steps(steps)))
    return [snaps[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gauss_bump"); ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--M", type=int, default=8); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N1", type=int, default=128); ap.add_argument("--N2", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--beta", type=float, default=1e-2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=4096); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--head", default="trained", choices=["trained", "varpro"])
    ap.add_argument("--rcond", type=float, default=1e-14, help="truncation of the VarPro head during training")
    ap.add_argument("--init", default="zero", choices=["zero", "smooth"]); ap.add_argument("--coords", default="raw")
    ap.add_argument("--name", required=True, help="output name: figures/profiles_<name>.gif")
    ap.add_argument("--rerender", default=None, help="a saved snaps .npz to render instead of training")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    if a.rerender:
        z = np.load(a.rerender)
        es = z["err_solved"] if "err_solved" in z else z["err"]
        snaps = [dict(step=int(t), err=float(e), err_solved=float(s), C=C, head=H) for t, e, s, C, H in zip(z["steps"], z["err"], es, z["C"], z["head"])]
        snaps = pick_frames(snaps, int(z["steps"][-1]))
        c1, c2, dirs = z["c1"], z["c2"], z["dirs"]
        title = str(z["title"]) if "title" in z else a.name
    else:
        print(f"{a.name}: {a.target} d={a.d} M={a.M} N1={a.N1} K={a.K} N2={a.N2} steps={a.steps} lr={a.lr} head={a.head}", flush=True)
        net, snaps = train_with_snapshots(a.target, a.d, a.M, a.N1, a.K, a.N2, a.steps, a.lr, a.beta, a.seed, a.n_train, a.head, a.rcond, a.init, a.coords)
        c1 = net.layers[0].bank.c.cpu().numpy(); c2 = net.layers[-1].bank.c.cpu().numpy()
        dirs = (net.layers[0].V / net.layers[0].V.norm(dim=1, keepdim=True)).detach().cpu().numpy()
        title = f"{a.target.replace('_', ' ')}, d = {a.d}, M = {a.M}, K = {a.K}, N = {a.N1}, profiles from {a.init}" + (f", {a.coords} coords" if a.coords != "raw" else "") + f", {a.head} head" + (f" (rcond {a.rcond:g})" if a.head == "varpro" else "") + f", Adam lr {a.lr:g}, {a.steps} steps"
        np.savez(DATA / f"profiles_{a.name}.npz", steps=[s["step"] for s in snaps], err=[s["err"] for s in snaps],
                 err_solved=[s["err_solved"] for s in snaps], C=np.stack([s["C"] for s in snaps]),
                 head=np.stack([s["head"] for s in snaps]), dirs=dirs, c1=c1, c2=c2, title=title)
    n = render(snaps, c1, c2, a.M, a.K, dirs, OUT / f"profiles_{a.name}.gif", title)
    print(f"{a.name}: final test rel L2 {snaps[-1]['err']:.2e}; {n} frames at {FRAME_MS} ms -> {OUT / f'profiles_{a.name}.gif'}", flush=True)


if __name__ == "__main__":
    main()
