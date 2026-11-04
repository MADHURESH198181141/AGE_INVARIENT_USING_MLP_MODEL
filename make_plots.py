"""Makes the 5 result graphs (+ one overview sheet) from a TEST folder of people the models/adapter never saw.
Photos stay on your computer.   Outputs (PNG, in --out_dir):
   roc_curve.png  det_curve.png  eer_comparison.png  fmr10000_comparison.png  inference_time.png  overview.png  plot_data.json

  python make_plots.py --test_dir DATA2/test --adapter ../adapter_trainonly/mlp_adapter.pth
  optional extra models trained from scratch (put compare.py / fno_face.py in the same folder or the parent folder):
  python make_plots.py --test_dir DATA2/test --adapter ... --fno ../compare_out/fno_face.pth --iresnet ../compare_out/iresnet50.pth

IMPORTANT: --adapter must be an adapter trained WITHOUT the test people (e.g. adapter_trainonly), otherwise the
"+ adapter" curve is invalid. FNMR at FMR 1/10,000 is only plotted when the test set has >= 100,000 impostor pairs
(about 320 people with first/last pairing); otherwise the graph says "not measurable" instead of inventing a number.
"""
import argparse, json, os, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(HERE, "archive"), os.getcwd()):
    if p not in sys.path: sys.path.insert(0, p)
from face_engine import Engine, Adapter
from eval_service import scan

# ---- palette (reference palette, fixed slot order; colour follows the model, not its rank)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"buffalo_l": "#2a78d6", "buffalo_l + MLP adapter": "#eb6834", "FNO (from scratch)": "#1baf7a", "IResNet50 (from scratch)": "#eda100"}
FMR_LEVELS = ((1e-1, "1/10"), (1e-2, "1/100"), (1e-3, "1/1,000"), (1e-4, "1/10,000"))


def style(plt):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
                         "axes.axisbelow": True, "legend.frameon": False, "axes.titleweight": "bold",
                         "axes.titlesize": 12, "axes.titlelocation": "left"})


# ---------------------------------------------------------------- metrics
def curves(gen, imp):
    gs, is_ = np.sort(gen), np.sort(imp)
    ths = np.unique(np.concatenate([gs, is_]))
    fmr = 1 - np.searchsorted(is_, ths, "left") / len(is_)
    fnmr = np.searchsorted(gs, ths, "left") / len(gs)
    return fmr, fnmr


def metrics(gen, imp):
    fmr, fnmr = curves(gen, imp)
    k = int(np.argmin(np.abs(fmr - fnmr)))
    out = {"EER": float((fmr[k] + fnmr[k]) / 2 * 100)}
    for r, nm in FMR_LEVELS:
        ok = fmr <= r
        out[nm] = float(fnmr[ok].min() * 100) if (ok.any() and len(imp) >= 10 / r) else float("nan")
    return out


def split(S, idx):
    M = S[np.ix_(idx, idx)]
    same = idx[:, None] == idx[None, :]
    return np.diag(M), M[~same]


def boot_ci(S, n_boot, seed=0):
    rng = np.random.default_rng(seed); n = S.shape[0]
    v = [metrics(*split(S, rng.integers(0, n, n)))["EER"] for _ in range(n_boot)]
    return np.percentile(v, [2.5, 97.5])


# ---------------------------------------------------------------- embeddings / timing
def load_images(paths):
    return [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]


def scratch_embed(model, paths, dev):
    import torch, torch.nn.functional as F
    out = []
    model.eval().to(dev)
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            xs = []
            for p in paths[i:i + 64]:
                im = cv2.cvtColor(cv2.resize(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR), (112, 112)), cv2.COLOR_BGR2RGB)
                xs.append(((im.astype(np.float32) / 255 - .5) / .5).transpose(2, 0, 1))
            out.append(F.normalize(model(torch.from_numpy(np.stack(xs)).to(dev)).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def time_scratch(model, n=30):
    import torch
    torch.set_num_threads(1); model.eval().cpu()
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        for _ in range(3): model(x)
        t = []
        for _ in range(n):
            s = time.perf_counter(); model(x); t.append(time.perf_counter() - s)
    return float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True); ap.add_argument("--out_dir", default="plots_out")
    ap.add_argument("--adapter", default=None, help="MLP adapter trained WITHOUT the test people")
    ap.add_argument("--fno", default=None); ap.add_argument("--iresnet", default=None)
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--time_images", type=int, default=20)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    style(plt); os.makedirs(a.out_dir, exist_ok=True)

    people = scan(a.test_dir); ids = list(people)
    pa = [people[p][0][0] for p in ids]; pb = [people[p][-1][0] for p in ids]
    n = len(ids); n_imp = n * (n - 1)
    print(f"{n} test people -> {n} genuine pairs, {n_imp:,} impostor pairs")

    eng = Engine(adapter_path="")
    ad = Adapter(a.adapter) if a.adapter else None
    if ad is None: print("no --adapter given: the adapter curve will be left out")
    plain, stages = [], []
    for k, p in enumerate(pa + pb, 1):
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        e, st = eng.embed_plain(img) if img is not None else (np.zeros(512, np.float32), 9)
        plain.append(e); stages.append(st)
        if k % 50 == 0: print(f"  embedded {k}/{2 * n}", flush=True)
    P = np.stack(plain); PA, PB = P[:n], P[n:]
    sets = {"buffalo_l": (PA, PB)}
    if ad is not None:
        sets["buffalo_l + MLP adapter"] = (np.stack([ad(e) for e in PA]), np.stack([ad(e) for e in PB]))
    models = {}
    if a.fno or a.iresnet:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if a.fno:
            from fno_face import load_pth; m = load_pth(a.fno); models["FNO (from scratch)"] = m
            sets["FNO (from scratch)"] = (scratch_embed(m, pa, dev), scratch_embed(m, pb, dev))
        if a.iresnet:
            from compare import IResNet50; m = IResNet50(); m.load_state_dict(torch.load(a.iresnet, map_location="cpu"))
            models["IResNet50 (from scratch)"] = m
            sets["IResNet50 (from scratch)"] = (scratch_embed(m, pa, dev), scratch_embed(m, pb, dev))

    S, M, CI, CUR = {}, {}, {}, {}
    idx = np.arange(n)
    for name, (A, B) in sets.items():
        S[name] = A @ B.T
        gen, imp = split(S[name], idx)
        M[name] = metrics(gen, imp); CUR[name] = curves(gen, imp); CI[name] = boot_ci(S[name], a.boot)
        print(f"{name:28s} EER {M[name]['EER']:.2f}%  CI [{CI[name][0]:.2f}, {CI[name][1]:.2f}]")
    names = list(sets); many = len(names) > 1

    # ---- timing (1 thread, this computer)
    T = {}
    imgs = load_images(pa[:a.time_images]); imgs = [i for i in imgs if i is not None]
    for im in imgs[:2]: eng.embed_plain(im)
    t_full, t_net = [], []
    for im in imgs:
        s = time.perf_counter(); eng.embed_plain(im); t_full.append(time.perf_counter() - s)
    al = np.zeros((112, 112, 3), np.uint8)
    for _ in range(3): eng._feat(al)
    for _ in range(a.time_images):
        s = time.perf_counter(); eng._feat(al); t_net.append(time.perf_counter() - s)
    T["buffalo_l  (detect + align + embed)"] = float(np.median(t_full))
    T["buffalo_l  (embedding network only)"] = float(np.median(t_net))
    if ad is not None:
        e = PA[0]; t = []
        for _ in range(200):
            s = time.perf_counter(); ad(e); t.append(time.perf_counter() - s)
        T["MLP adapter (extra cost)"] = float(np.median(t))
    for nm, m in models.items(): T[nm + "  (network only)"] = time_scratch(m)
    print("median seconds per image, 1 thread:", {k: round(v, 4) for k, v in T.items()})

    def legend(ax, **kw):
        if many: ax.legend(loc=kw.pop("loc", "best"), fontsize=9, labelcolor=INK2, **kw)

    def note(fig, text):
        fig.text(0.01, 0.01, text, fontsize=8, color=MUTED, ha="left", va="bottom")

    foot = f"{n} test people unseen in training | {n} genuine / {n_imp:,} impostor pairs (youngest vs oldest photo)"

    # ---- drawing functions (take an axis so they also fill the overview)
    def roc(ax):
        for nm in names:
            f, fn = CUR[nm]; ax.plot(f, 1 - fn, color=COL[nm], lw=2, label=nm)
        ax.set(xlabel="False match rate (FMR)", ylabel="True accept rate (1 - FNMR)", xlim=(0, 1), ylim=(0, 1.01))
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1); ax.set_title("ROC curve"); legend(ax, loc="lower right")

    def det(ax):
        lo = 0.5 / max(n_imp, 1)
        for nm in names:
            f, fn = CUR[nm]; k = (f > 0) & (fn > 0)
            ax.plot(f[k], fn[k], color=COL[nm], lw=2, label=nm)
        ax.set(xscale="log", yscale="log", xlabel="False match rate (FMR)", ylabel="False non-match rate (FNMR)")
        ax.set_xlim(max(lo, 1e-5), 1); ax.set_ylim(1e-2, 1)
        ax.axvspan(max(lo, 1e-5), 10.0 / max(n_imp, 1), color=GRID, alpha=0.6, lw=0)
        ax.text(0.02, 0.04, "shaded: fewer than 10 impostor pairs per step -\nnot reliable", transform=ax.transAxes, fontsize=8, color=MUTED)
        ax.set_title("DET curve (lower-left is better)"); legend(ax, loc="upper right")

    def eer(ax):
        x = np.arange(len(names)); v = np.array([M[nm]["EER"] for nm in names])
        lo = np.array([M[nm]["EER"] - CI[nm][0] for nm in names]).clip(0); hi = np.array([CI[nm][1] - M[nm]["EER"] for nm in names]).clip(0)
        ax.bar(x, v, 0.5, color=[COL[nm] for nm in names], zorder=3)
        ax.errorbar(x, v, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4, zorder=4)
        for xi, vi, h in zip(x, v, hi): ax.text(xi, vi + h + 0.01 * max(v.max(), 1), f"{vi:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([nm.replace(" + ", "\n+ ").replace(" (from", "\n(from") for nm in names], fontsize=9)
        ax.set(ylabel="Equal error rate, % (lower is better)"); ax.set_ylim(0, (v + hi).max() * 1.2); ax.grid(axis="x", visible=False)
        ax.set_title("EER comparison (bars: 95% CI over people)")

    def fmr_panels(axs):
        for ax, (r, lab) in zip(axs, FMR_LEVELS):
            vals = [M[nm][lab] for nm in names]
            ax.grid(axis="x", visible=False)
            if all(np.isnan(vals)):
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                ax.text(0.5, 0.5, f"not measurable\nneeds >= {int(10 / r):,} impostor pairs\n(this test set: {n_imp:,})", ha="center", va="center", transform=ax.transAxes, fontsize=9, color=MUTED)
            else:
                x = np.arange(len(names)); ax.bar(x, vals, 0.6, color=[COL[nm] for nm in names], zorder=3)
                for xi, vi in zip(x, vals): ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
                ax.set_xticks([]); ax.set_ylim(0, max(vals) * 1.25 + 1e-9)
            ax.set_title(f"FMR {lab}", fontsize=10)
        axs[0].set_ylabel("FNMR % (lower is better)")

    def timing(ax):
        keys = list(T); v = [T[k] * 1000 for k in keys]
        cols = [COL["buffalo_l"] if k.startswith("buffalo") else COL["buffalo_l + MLP adapter"] if k.startswith("MLP") else COL[k.split("  ")[0]] for k in keys]
        y = np.arange(len(keys))[::-1]; ax.barh(y, v, 0.5, color=cols, zorder=3)
        for yi, vi in zip(y, v): ax.text(vi * 1.15, yi, f"{vi:.1f} ms" if vi < 1000 else f"{vi / 1000:.2f} s", va="center", fontsize=9, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=9); ax.set_xscale("log")
        ax.set_xlim(min(v) * 0.5, max(v) * 6); ax.set(xlabel="Median time per image, milliseconds (log scale, 1 CPU thread)")
        ax.axvline(10000, color=AXIS, lw=1); ax.text(10000 * 0.9, y.max() + 0.45, "enroll limit 10 s", ha="right", fontsize=8, color=MUTED)
        ax.grid(axis="y", visible=False); ax.set_title("Inference time per image")

    # ---- single-chart files
    def save(fn, drawer, size=(7.2, 5.4)):
        fig, ax = plt.subplots(figsize=size); drawer(ax); note(fig, foot); fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(a.out_dir, fn), dpi=160); plt.close(fig)
    save("roc_curve.png", roc); save("det_curve.png", det)
    save("eer_comparison.png", eer, (max(6.2, 2.2 * len(names) + 2.5), 5.4))
    fig, axs = plt.subplots(1, 4, figsize=(11, 4.4)); fmr_panels(axs)
    if many: fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[nm]) for nm in names], labels=names, loc="upper right", ncol=len(names), fontsize=9, frameon=False, labelcolor=INK2)
    fig.suptitle("FNMR at fixed false-match rates", x=0.01, ha="left", fontweight="bold", fontsize=12)
    note(fig, foot); fig.tight_layout(rect=(0, 0.04, 1, 0.9)); fig.savefig(os.path.join(a.out_dir, "fmr10000_comparison.png"), dpi=160); plt.close(fig)
    save("inference_time.png", timing, (9.5, 4.2))

    # ---- overview
    fig = plt.figure(figsize=(14, 11)); gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 0.8])
    roc(fig.add_subplot(gs[0, 0:2])); det(fig.add_subplot(gs[0, 2:4])); eer(fig.add_subplot(gs[1, 0:2]))
    sub = gs[1, 2:4].subgridspec(2, 2, hspace=0.5, wspace=0.35)
    fmr_panels([fig.add_subplot(sub[i // 2, i % 2]) for i in range(4)])
    timing(fig.add_subplot(gs[2, :]))
    note(fig, foot); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(a.out_dir, "overview.png"), dpi=130); plt.close(fig)

    json.dump({"n_people": n, "impostor_pairs": n_imp, "metrics": M, "eer_ci95": {k: [float(x) for x in v] for k, v in CI.items()},
               "seconds_per_image_1thread": T}, open(os.path.join(a.out_dir, "plot_data.json"), "w"), indent=2)
    print("saved to", os.path.abspath(a.out_dir), ":", sorted(os.listdir(a.out_dir)))


if __name__ == "__main__":
    main()

