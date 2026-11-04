"""STEP 2 - FNO adapter on top of FROZEN buffalo_l embeddings, tested with person-level cross-validation.

Idea: buffalo_l (pretrained on millions of faces) stays frozen. A small Fourier Neural Operator learns a correction
        z = normalize( e + alpha * FNO(e) )         (alpha starts at 0  ->  starts EXACTLY as buffalo_l)
from aging pairs. It is only kept if it beats buffalo_l on people it never saw.

Honest-by-construction protocol
  * K-fold split BY PERSON: every test person is unseen by the adapter in that fold.
  * Epoch count is chosen on an inner validation set of training people (epoch 0 = "no adaptation" is allowed).
  * Control arm: a plain MLP adapter of similar size, to show whether the FNO structure matters at all.
  * Pooled out-of-fold EER with 95% confidence intervals (people are resampled, not pairs) and a paired verdict.

  python adapter_cv.py --emb fgnet_emb.npz --out_dir adapter_out
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


# ============================================================ model
class SpectralConv1d(nn.Module):
    def __init__(self, ch, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(ch, ch, modes, 2) / ch)

    def forward(self, x):                                   # x: (B, C, L)
        B, C, L = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft(x.float())
            m = min(self.modes, xf.shape[-1])
            out = torch.zeros(B, C, xf.shape[-1], dtype=torch.cfloat, device=x.device)
            out[:, :, :m] = torch.einsum("bix,iox->box", xf[:, :, :m],
                                         torch.view_as_complex(self.w[:, :, :m].contiguous()))
            return torch.fft.irfft(out, n=L)


class FNOAdapter(nn.Module):
    def __init__(self, dim=512, width=16, modes=32, depth=3):
        super().__init__()
        self.cfg = dict(kind="fno", dim=dim, width=width, modes=modes, depth=depth)
        self.lift = nn.Conv1d(1, width, 1)
        self.spec = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(depth)])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Conv1d(width, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(1))          # zero-init: starts identical to buffalo_l

    def forward(self, e):                                   # e: (B, dim), L2-normalised
        x = self.lift(e.unsqueeze(1))
        for sp, sk in zip(self.spec, self.skip):
            x = F.gelu(sp(x) + sk(x))
        return F.normalize(e + self.alpha * self.proj(x).squeeze(1), dim=1)


class MLPAdapter(nn.Module):
    def __init__(self, dim=512, hidden=64):
        super().__init__()
        self.cfg = dict(kind="mlp", dim=dim, hidden=hidden)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, e):
        return F.normalize(e + self.alpha * self.net(e), dim=1)


def build(kind, a):
    return FNOAdapter(512, a.width, a.modes, a.depth) if kind == "fno" else MLPAdapter(512, a.mlp_hidden)


def save_adapter(model, path, extra=None):
    torch.save({"cfg": model.cfg, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                **(extra or {})}, path)


def load_adapter(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    c = dict(b["cfg"]); kind = c.pop("kind")
    m = FNOAdapter(**c) if kind == "fno" else MLPAdapter(**c)
    m.load_state_dict(b["state_dict"]); return m.eval()


class ArcFace(nn.Module):
    def __init__(self, n, s=30.0, m=0.3, dim=512):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, dim) * 0.01); self.s, self.m = s, m

    def forward(self, e, y):
        cos = F.linear(e, F.normalize(self.W)).clamp(-1 + 1e-6, 1 - 1e-6)
        tgt = torch.cos(torch.acos(cos) + self.m)
        oh = F.one_hot(y, cos.size(1)).bool()
        return F.cross_entropy(self.s * torch.where(oh, tgt, cos), y)


# ============================================================ pairs + metrics
def pair_scores(Z, person, age, min_gap):
    """All genuine pairs (same person, age gap >= min_gap when ages known) and ALL impostor pairs."""
    S = Z @ Z.T
    iu = np.triu_indices(len(Z), 1)
    s, pi, pj = S[iu], person[iu[0]], person[iu[1]]
    same = pi == pj
    ok = np.ones_like(same) if np.isnan(age).all() else (np.abs(age[iu[0]] - age[iu[1]]) >= min_gap)
    g = same & ok
    return dict(gs=s[g], gp=pi[g], is_=s[~same], ia=pi[~same], ib=pj[~same])


def weighted_curve(gs, gw, is_, iw):
    og, oi = np.argsort(gs), np.argsort(is_)
    gs, gw, is_, iw = gs[og], gw[og], is_[oi], iw[oi]
    cg, ci = np.concatenate([[0], np.cumsum(gw)]), np.concatenate([[0], np.cumsum(iw)])
    ths = np.concatenate([gs, is_])
    fnmr = cg[np.searchsorted(gs, ths, "left")] / cg[-1]
    fmr = 1 - ci[np.searchsorted(is_, ths, "left")] / ci[-1]
    return fmr, fnmr


def metrics(P, cw=None):
    """EER % and TAR % at FAR 1e-2 / 1e-3 / 1e-4 (TAR = 1 - FNMR). cw = per-person bootstrap counts."""
    if len(P["gs"]) == 0 or len(P["is_"]) == 0:
        return dict(EER=float("nan"), TAR_1e2=float("nan"), TAR_1e3=float("nan"), TAR_1e4=float("nan"))
    gw = np.ones(len(P["gs"])) if cw is None else cw[P["gp"]].astype(float)
    iw = np.ones(len(P["is_"])) if cw is None else (cw[P["ia"]] * cw[P["ib"]]).astype(float)
    if gw.sum() == 0 or iw.sum() == 0:
        return dict(EER=float("nan"), TAR_1e2=float("nan"), TAR_1e3=float("nan"), TAR_1e4=float("nan"))
    fmr, fnmr = weighted_curve(P["gs"], gw, P["is_"], iw)
    k = int(np.argmin(np.abs(fmr - fnmr)))
    out = {"EER": float((fmr[k] + fnmr[k]) / 2 * 100)}
    n_imp = iw.sum() if cw is None else len(P["is_"])
    for far, nm in ((1e-2, "TAR_1e2"), (1e-3, "TAR_1e3"), (1e-4, "TAR_1e4")):
        ok = fmr <= far
        out[nm] = float((1 - fnmr[ok].min()) * 100) if (ok.any() and n_imp >= 10 / far) else float("nan")
    return out


@torch.no_grad()
def transform(model, E):
    if model is None:
        return E
    model.eval()
    return model(torch.from_numpy(E)).numpy()


# ============================================================ training
def train_adapter(kind, E, y, a, val=None, seed=0):
    """Full-batch-ish training. val=(E, person, age): pick the best epoch on it (epoch 0 = no adaptation)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model, head = build(kind, a), ArcFace(int(y.max()) + 1, a.scale, a.margin)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=a.wd)
    X, Y = torch.from_numpy(E), torch.from_numpy(y)
    best_eer, best_ep, best_state = None, 0, {k: v.clone() for k, v in model.state_dict().items()}
    if val is not None:
        best_eer = metrics(pair_scores(val[0], val[1], val[2], a.min_gap))["EER"]
    for ep in range(1, a.epochs + 1):
        model.train(); head.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), a.bs):
            idx = perm[i:i + a.bs]
            x = F.normalize(X[idx] + a.noise * torch.randn_like(X[idx]), dim=1)   # embedding-noise augmentation
            loss = head(model(x), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if val is not None and ep % a.eval_every == 0:
            eer = metrics(pair_scores(transform(model, val[0]), val[1], val[2], a.min_gap))["EER"]
            if eer < best_eer - 1e-9:
                best_eer, best_ep = eer, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if val is not None:
        model.load_state_dict(best_state)
    return model.eval(), best_ep


def relabel(person):
    u = {p: i for i, p in enumerate(sorted(set(person.tolist())))}
    return np.array([u[p] for p in person.tolist()], np.int64)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True); ap.add_argument("--out_dir", default="adapter_out")
    ap.add_argument("--arms", default="fno,mlp"); ap.add_argument("--final", default="fno")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--min_gap", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--scale", type=float, default=30.0); ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=16); ap.add_argument("--modes", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--max_stage", type=int, default=4,
                    help="use only images found by the detector (stages 0-4). Stage 5 = unaligned fallback with "
                         "no face check; include it only as a separate stress test with --max_stage 5")
    ap.add_argument("--boot", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d = np.load(a.emb)
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    person_str, age = d["person"], d["age"].astype(np.float32)
    if "stage" in d.files:
        st = d["stage"]; msk = st <= a.max_stage
        print("images per stage:", {int(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))},
              f"| using stage <= {a.max_stage}: {int(msk.sum())} images")
        E, person_str, age = E[msk], person_str[msk], age[msk]
    keep_p = [p for p, c in zip(*np.unique(person_str, return_counts=True)) if c >= 2]
    m = np.isin(person_str, keep_p)
    E, person_str, age = E[m], person_str[m], age[m]
    person = relabel(person_str); P = int(person.max()) + 1
    ages_known = not np.isnan(age).all()
    print(f"{len(E)} images | {P} people | ages {'known' if ages_known else 'UNKNOWN (all same-person pairs used)'}"
          f" | genuine pairs need age gap >= {a.min_gap if ages_known else 0}")

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(P); folds = np.array_split(order, a.folds)
    arms = a.arms.split(",")
    names = ["buffalo_l (baseline)"] + [f"{k.upper()} adapter" for k in arms]
    oof = {n: {k: [] for k in ("gs", "gp", "is_", "ia", "ib")} for n in names}
    per_fold = {n: [] for n in names}; best_eps = {k: [] for k in arms}

    for fi, test_p in enumerate(folds):
        tr_p = np.setdiff1d(np.arange(P), test_p)
        nval = max(3, int(round(0.2 * len(tr_p))))
        val_p = rng.permutation(tr_p)[:nval]; fit_p = np.setdiff1d(tr_p, val_p)
        te, va, fi_m = np.isin(person, test_p), np.isin(person, val_p), np.isin(person, fit_p)
        y_fit = relabel(person[fi_m])
        row = f"fold {fi+1}/{a.folds}: test {len(test_p)} people"
        res = {"buffalo_l (baseline)": metrics(pair_scores(E[te], person[te], age[te], a.min_gap))}
        pairs_by = {"buffalo_l (baseline)": pair_scores(E[te], person[te], age[te], a.min_gap)}
        for k in arms:
            model, bep = train_adapter(k, E[fi_m], y_fit, a, val=(E[va], person[va], age[va]), seed=a.seed + fi)
            best_eps[k].append(bep)
            Pt = pair_scores(transform(model, E[te]), person[te], age[te], a.min_gap)
            nm = f"{k.upper()} adapter"; res[nm] = metrics(Pt); pairs_by[nm] = Pt
            row += f" | {k} best epoch {bep}"
        for n in names:
            per_fold[n].append(res[n])
            for key in oof[n]:
                oof[n][key].append(pairs_by[n][key])
        print(row + "   EER: " + "  ".join(f"{n.split()[0]} {res[n]['EER']:.2f}%" for n in names), flush=True)
    for n in names:
        oof[n] = {k: np.concatenate(v) for k, v in oof[n].items()}

    # ------------- report
    print("\n" + "=" * 86)
    print(f"RESULTS - out-of-fold, every test person unseen | genuine pairs {len(oof[names[0]]['gs'])} | "
          f"impostor pairs {len(oof[names[0]]['is_'])}")
    pooled = {n: metrics(oof[n]) for n in names}
    print(f"{'':26s}{'EER %':>10s}{'TAR@FAR 1%':>14s}{'TAR@FAR 0.1%':>15s}{'TAR@FAR 0.01%':>16s}")
    for n in names:
        r = pooled[n]; f = lambda v: "n/a" if np.isnan(v) else f"{v:.2f}"
        print(f"{n:26s}{f(r['EER']):>10s}{f(r['TAR_1e2']):>14s}{f(r['TAR_1e3']):>15s}{f(r['TAR_1e4']):>16s}")
    print("\nPer-fold EER % (mean +- std over folds):")
    for n in names:
        v = np.array([x["EER"] for x in per_fold[n]]); print(f"  {n:26s} {v.mean():.2f} +- {v.std():.2f}   {np.round(v, 2).tolist()}")

    brng = np.random.default_rng(a.seed + 99); boots = {n: [] for n in names}
    for _ in range(a.boot):
        cw = np.bincount(brng.integers(0, P, P), minlength=P)
        for n in names:
            boots[n].append(metrics(oof[n], cw)["EER"])
    boots = {n: np.array(v) for n, v in boots.items()}
    print("\nEER 95% confidence interval (resampling PEOPLE):")
    for n in names:
        lo, hi = np.nanpercentile(boots[n], [2.5, 97.5]); print(f"  {n:26s} {pooled[n]['EER']:.2f}%  [{lo:.2f}, {hi:.2f}]")
    verdicts = {}
    print("\nPaired difference vs buffalo_l  (positive = adapter has LOWER error):")
    for n in names[1:]:
        dlt = boots[names[0]] - boots[n]; lo, hi = np.nanpercentile(dlt, [2.5, 97.5])
        v = ("BETTER than buffalo_l (clear)" if lo > 0 else "WORSE than buffalo_l (clear)" if hi < 0
             else "no clear difference from buffalo_l")
        verdicts[n] = v; print(f"  {n:26s} {np.nanmean(dlt):+.2f} points  95% CI [{lo:+.2f}, {hi:+.2f}]  ->  {v}")
    if len(arms) == 2:
        dlt = boots[names[2]] - boots[names[1]]; lo, hi = np.nanpercentile(dlt, [2.5, 97.5])
        v = ("FNO clearly better than the plain MLP" if lo > 0 else "MLP clearly better than FNO" if hi < 0
             else "FNO and MLP are indistinguishable (the Fourier structure is not what helps)")
        print(f"  {'MLP minus FNO':26s} {np.nanmean(dlt):+.2f} points  95% CI [{lo:+.2f}, {hi:+.2f}]  ->  {v}")
        verdicts["fno_vs_mlp"] = v

    # ------------- final model on ALL people
    k = a.final
    ep_final = int(np.median(best_eps[k])) if k in best_eps else 0
    out = {"pooled": pooled, "per_fold": per_fold, "best_epochs": best_eps, "verdicts": verdicts,
           "n_people": P, "n_images": int(len(E))}
    if ep_final == 0:
        print(f"\nNo epochs of adaptation helped on validation people (median best epoch 0) -> no adapter saved.")
    else:
        a2 = argparse.Namespace(**{**vars(a), "epochs": ep_final})
        model, _ = train_adapter(k, E, person, a2, val=None, seed=a.seed)
        path = os.path.join(a.out_dir, f"{k}_adapter.pth")
        save_adapter(model, path, {"epochs": ep_final, "min_gap": a.min_gap, "base": "buffalo_l"})
        print(f"\nFinal {k.upper()} adapter trained on ALL {P} people for {ep_final} epochs -> {path}")
        out["saved"] = path
    json.dump(out, open(os.path.join(a.out_dir, "adapter_results.json"), "w"), indent=2)
    print("saved", os.path.join(a.out_dir, "adapter_results.json"))


if __name__ == "__main__":
    main()
