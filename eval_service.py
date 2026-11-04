"""Score the DEPLOYED pipeline (face detect + buffalo_l + adapter, from face_engine.py) the way the portal will:
FNMR at FMR 1/10, 1/100, 1/1,000, 1/10,000, plus EER, ROC and DET curves.
Photos stay on your computer.

TEST FOLDER: people the model/adapter never saw.
    TEST/<person>/<images>          (or flat FG-NET names 001A02.JPG)
  --pairs first_last (default): per person, first & last file by name = (child, adult) pair. One genuine pair per
                                person; impostors = every child vs every OTHER person's adult. With N people you get
                                N genuine and N*(N-1) impostor pairs -> need N >= ~320 for FMR 1/10,000 to be measurable.
  --pairs all:  every same-person pair (with age gap >= --min_gap when ages are in the names); impostors sampled.

    python eval_service.py --test_dir DATA3/test --out_dir eval_service_out
    python eval_service.py --test_dir DATA3/test --no_adapter          (plain buffalo_l, to see what the adapter adds)
Embeddings are cached in <out_dir>/emb_cache.npz so a re-run (or a crash) does not start from scratch.
"""
import argparse, os, re, sys, time, json
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_engine import Engine

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scan(root):
    people = {}
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() not in EXT: continue
            if os.path.abspath(dp) == os.path.abspath(root):
                m = re.match(r"^(\d+)[A-Za-z]", f)
                if not m: continue
                key = m.group(1)
            else:
                key = os.path.relpath(dp, root).split(os.sep)[0]
            ma = re.match(r"^\d+[A-Za-z](\d+)", f)
            people.setdefault(key, []).append((os.path.join(dp, f), float(ma.group(1)) if ma else float("nan")))
    people = {k: sorted(v) for k, v in sorted(people.items()) if len(v) >= 2}
    if not people:
        raise SystemExit(f"no people with >=2 images in {os.path.abspath(root)}")
    return people


def curves(gen, imp):
    gs, is_ = np.sort(gen), np.sort(imp)
    ths = np.unique(np.concatenate([gs, is_]))
    fmr = 1 - np.searchsorted(is_, ths, "left") / len(is_)
    fnmr = np.searchsorted(gs, ths, "left") / len(gs)
    return fmr, fnmr


def report(gen, imp):
    fmr, fnmr = curves(gen, imp)
    k = int(np.argmin(np.abs(fmr - fnmr)))
    out = {"EER_%": round(float((fmr[k] + fnmr[k]) / 2 * 100), 3), "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp))}
    for r, nm in ((1e-1, "FMR10"), (1e-2, "FMR100"), (1e-3, "FMR1000"), (1e-4, "FMR10000")):
        ok = fmr <= r
        v = float(fnmr[ok].min() * 100) if ok.any() else float("nan")
        out[f"FNMR_%@{nm}"] = round(v, 3) if len(imp) >= 10 / r else f"unreliable({round(v, 2)}; need >= {int(10/r):,} impostor pairs)"
    return out, fmr, fnmr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True); ap.add_argument("--out_dir", default="eval_service_out")
    ap.add_argument("--pairs", default="first_last", choices=["first_last", "all"])
    ap.add_argument("--min_gap", type=float, default=5.0); ap.add_argument("--max_imp", type=int, default=2_000_000)
    ap.add_argument("--no_adapter", action="store_true"); ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    eng = Engine(adapter_path="" if a.no_adapter else a.adapter)
    print("adapter:", "ON" if eng.adapter is not None else "OFF (plain buffalo_l)")
    people = scan(a.test_dir)
    items = []                                    # (person, age, path)
    for p, v in people.items():
        sel = [v[0], v[-1]] if a.pairs == "first_last" else v
        items += [(p, ag, pa) for pa, ag in sel]
    cache_path = os.path.join(a.out_dir, "emb_cache.npz" if not a.no_adapter else "emb_cache_noadapter.npz")
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True); cache = dict(zip(z["path"].tolist(), zip(z["emb"], z["stage"])))
    t0, done = time.time(), 0
    for i, (p, ag, path) in enumerate(items, 1):
        if path in cache: continue
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        cache[path] = eng.embed(img) if img is not None else (np.zeros(512, np.float32), 9)
        done += 1
        if done % 50 == 0 or i == len(items):
            ks = list(cache); np.savez(cache_path, path=np.array(ks), emb=np.stack([cache[k][0] for k in ks]),
                                       stage=np.array([cache[k][1] for k in ks]))
            print(f"  {i}/{len(items)} images | {(time.time()-t0)/done:.2f}s each", flush=True)
    ks = list(cache); np.savez(cache_path, path=np.array(ks), emb=np.stack([cache[k][0] for k in ks]), stage=np.array([cache[k][1] for k in ks]))
    E = np.stack([cache[p][0] for _, _, p in items]); stg = np.array([cache[p][1] for _, _, p in items])
    person = np.array([p for p, _, _ in items]); age = np.array([g for _, g, _ in items], np.float32)
    print("images per detection stage:", {int(k): int(v) for k, v in zip(*np.unique(stg, return_counts=True))},
          "(0-1 normal, 2-4 rescued, 5 no face found, 9 unreadable file)")

    rng = np.random.default_rng(0)
    if a.pairs == "first_last":
        n = len(people); A, B = E[0::2], E[1::2]
        S = A @ B.T; gen = np.diag(S); imp = S[~np.eye(n, dtype=bool)]
    else:
        gen_l = []
        pidx = {p: np.where(person == p)[0] for p in people}
        for p, ix in pidx.items():
            for x in range(len(ix)):
                for y in range(x + 1, len(ix)):
                    i, j = ix[x], ix[y]
                    if np.isnan(age[i]) or abs(age[i] - age[j]) >= a.min_gap:
                        gen_l.append(float(E[i] @ E[j]))
        gen = np.array(gen_l)
        ii = rng.integers(0, len(E), a.max_imp); jj = rng.integers(0, len(E), a.max_imp); ok = person[ii] != person[jj]
        imp = np.einsum("ij,ij->i", E[ii[ok]], E[jj[ok]])
    if len(gen) == 0: raise SystemExit("no genuine pairs found (age gap too strict?)")
    res, fmr, fnmr = report(gen, imp)
    print("\n" + "=" * 70); print(f"PORTAL-STYLE RESULTS  ({len(people)} people, pairs={a.pairs}, adapter={'on' if eng.adapter else 'off'})")
    for k, v in res.items(): print(f"  {k:22s} {v}")
    print("  (FNMR = % of genuine pairs wrongly rejected at that false-match rate; lower is better)")
    json.dump(res, open(os.path.join(a.out_dir, "results.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(fmr, 1 - fnmr); ax[0].set(xlabel="FMR", ylabel="TAR (1 - FNMR)", title="ROC"); ax[0].grid(True)
        ax[1].plot(fmr, fnmr); ax[1].set(xscale="log", yscale="log", xlabel="FMR", ylabel="FNMR", title="DET"); ax[1].grid(True, which="both", ls="--")
        plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "roc_det.png"), dpi=140); print("saved roc_det.png")
    except ImportError:
        print("(pip install matplotlib to also save ROC/DET plots)")
    print("saved", os.path.join(a.out_dir, "results.json"))


if __name__ == "__main__":
    main()

