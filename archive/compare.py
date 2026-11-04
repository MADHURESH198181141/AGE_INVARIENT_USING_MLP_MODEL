"""Compare FNO vs IResNet50 (both + ArcFace) on the SAME person-level train/test split.

DATA LAYOUT (split by person, no person in both folders):
  TRAIN/<person>/<images...>          (>=2 images per person)
  TEST/<person>/<child>.jpg, <adult>.jpg   (first & last file by name = age-gap pair)

USAGE:
  python compare.py --train_dir DATA/train --test_dir DATA/test --epochs 30 --out_dir compare_out
Optional: --models fno,iresnet50   --bs 32   --lr 1e-3   --eval_every 5
Quick smoke test: add --max_train_people 40 --epochs 1
"""
import argparse, os, re, time, json, random
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from fno_face import FNOFace, save_pkl, save_pth

IMG = 112
POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))  # cv2 releases the GIL


# ------------------------------------------------------------------ data
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def index_people(root, min_imgs=2, limit=None):
    """Return {person: [image paths sorted by name]}.
    Accepts  root/<person>/<images>   (images may be in deeper subfolders)
    or a FLAT folder of FG-NET style names (001A02.JPG -> person 001)."""
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: folder not found: {os.path.abspath(root)}")
    people = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            if os.path.abspath(dp) == os.path.abspath(root):
                m = re.match(r"^(\d+)[A-Za-z]", f)
                if not m:
                    continue
                key = m.group(1)
            else:
                key = os.path.relpath(dp, root).split(os.sep)[0]
            people.setdefault(key, []).append(os.path.join(dp, f))
    people = {k: sorted(v) for k, v in sorted(people.items()) if len(v) >= min_imgs}
    if limit and len(people) > limit:
        keys = sorted(random.Random(0).sample(sorted(people), limit))
        people = {k: people[k] for k in keys}
    if not people:
        seen = os.listdir(root)[:8]
        raise SystemExit(
            f"ERROR: no people with >=2 images found in {os.path.abspath(root)}\n"
            f"First items in that folder: {seen}\n"
            "Expected  <folder>/<person>/<image files>  (>=2 images per person)\n"
            "or flat files named like 001A02.JPG.")
    return people


def load(path, aug=False):
    im = cv2.imread(path)
    if im is None:
        raise IOError(path)
    im = cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB).astype(np.float32)
    if aug:  # augmentation is applied to TRAIN images only
        if random.random() < 0.5: im = im[:, ::-1]
        im = np.clip(im * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
    x = (im / 255. - .5) / .5
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# ------------------------------------------------------------------ models
class IBlock(nn.Module):
    def __init__(s, cin, cout, stride, down):
        super().__init__()
        s.bn1 = nn.BatchNorm2d(cin); s.c1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        s.bn2 = nn.BatchNorm2d(cout); s.act = nn.PReLU(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, stride, 1, bias=False); s.bn3 = nn.BatchNorm2d(cout); s.down = down

    def forward(s, x):
        o = s.bn3(s.c2(s.act(s.bn2(s.c1(s.bn1(x))))))
        return o + (x if s.down is None else s.down(x))


class IResNet50(nn.Module):
    def __init__(s, emb=512, layers=(3, 4, 14, 3)):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); s.bn1 = nn.BatchNorm2d(64); s.prelu = nn.PReLU(64)
        blocks, cin = [], 64
        for cout, n in zip((64, 128, 256, 512), layers):
            for k in range(n):
                st = 2 if k == 0 else 1
                down = (nn.Sequential(nn.Conv2d(cin, cout, 1, st, bias=False), nn.BatchNorm2d(cout))
                        if (st != 1 or cin != cout) else None)
                blocks.append(IBlock(cin, cout, st, down)); cin = cout
        s.layers = nn.Sequential(*blocks)
        s.bn2 = nn.BatchNorm2d(512); s.drop = nn.Dropout(0.4)
        s.fc = nn.Linear(512 * 7 * 7, emb); s.features = nn.BatchNorm1d(emb)

    def forward(s, x):
        x = s.layers(s.prelu(s.bn1(s.conv1(x))))
        x = s.drop(torch.flatten(s.bn2(x), 1))
        return F.normalize(s.features(s.fc(x)), dim=1)


class ArcFace(nn.Module):
    def __init__(s, emb, n, scale, margin):
        super().__init__()
        s.W = nn.Parameter(torch.randn(n, emb) * 0.01); s.s, s.m = scale, margin

    def forward(s, e, y):
        cos = F.linear(e, F.normalize(s.W)).clamp(-1 + 1e-6, 1 - 1e-6)
        tgt = torch.cos(torch.acos(cos) + s.m)
        oh = F.one_hot(y, cos.size(1)).bool()
        return F.cross_entropy(s.s * torch.where(oh, tgt, cos), y)


def build(name):
    return FNOFace() if name == "fno" else IResNet50()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def embed_paths(model, paths, dev, bs=64):
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        x = torch.stack(list(POOL.map(load, paths[i:i + bs]))).to(dev)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, test_dir, dev):
    people = index_people(test_dir)
    ppl = list(people)
    # first and last image by filename = the age-gap pair
    pa = [people[p][0] for p in ppl]; pb = [people[p][-1] for p in ppl]
    A, B = embed_paths(model, pa, dev), embed_paths(model, pb, dev)
    S = A @ B.T
    gen = np.diag(S); imp = S[~np.eye(len(ppl), dtype=bool)]
    ths = np.unique(np.concatenate([gen, imp]))
    fmr = np.array([(imp >= t).mean() for t in ths]); fnmr = np.array([(gen < t).mean() for t in ths])
    k = int(np.argmin(np.abs(fmr - fnmr)))
    res = {"people": len(ppl), "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)),
           "EER_%": round(float((fmr[k] + fnmr[k]) / 2 * 100), 3)}
    for r, nm in ((1e-1, "FMR10"), (1e-2, "FMR100"), (1e-3, "FMR1000"), (1e-4, "FMR10000")):
        ok = fmr <= r
        v = float(fnmr[ok].min() * 100) if ok.any() else float("nan")
        # a 1-in-N false match rate is only meaningful with >= 10*N impostor pairs
        res[f"FNMR_%@{nm}"] = round(v, 3) if len(imp) >= 10 / r else f"unreliable({round(v,2)})"
    return res


# ------------------------------------------------------------------ training
def run(name, a, items, n_classes, dev):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    model = build(name).to(dev)
    head = ArcFace(512, n_classes, a.scale, a.margin).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), a.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
    amp = dev == "cuda"; scaler = torch.amp.GradScaler("cuda") if amp else None
    print(f"\n===== {name}  params={params/1e6:.1f}M  device={dev} =====", flush=True)
    t0 = time.time(); history = []
    for ep in range(1, a.epochs + 1):
        model.train(); head.train(); random.shuffle(items); tot, nb = 0., 0
        for i in range(0, len(items) - a.bs + 1, a.bs):
            b = items[i:i + a.bs]
            x = torch.stack(list(POOL.map(lambda t: load(t[0], aug=True), b))).to(dev)
            y = torch.tensor([l for _, l in b]).to(dev)
            opt.zero_grad()
            if amp:
                with torch.amp.autocast("cuda"):
                    loss = head(model(x), y)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss = head(model(x), y); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        line = f"[{name}] epoch {ep}/{a.epochs} loss {tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)"
        if ep % a.eval_every == 0 and ep != a.epochs:
            r = evaluate(model, a.test_dir, dev); history.append((ep, r["EER_%"])); line += f" | test EER {r['EER_%']}%"
        print(line, flush=True)
    train_min = (time.time() - t0) / 60
    res = evaluate(model, a.test_dir, dev)
    model.cpu().eval(); torch.set_num_threads(1)
    x = load(next(iter(index_people(a.test_dir).values()))[0])[None]
    t = time.time()
    with torch.no_grad():
        for _ in range(5): model(x)
    res.update({"params_M": round(params / 1e6, 1), "train_min": round(train_min, 1),
                "sec_per_image_1thread": round((time.time() - t) / 5, 3), "eer_history": history})
    os.makedirs(a.out_dir, exist_ok=True)
    if name == "fno":
        save_pth(model, f"{a.out_dir}/fno_face.pth")   # full precision checkpoint
        save_pkl(model, f"{a.out_dir}/fno_face.pkl")
    else: torch.save(model.state_dict(), f"{a.out_dir}/iresnet50.pth")
    torch.set_num_threads(max(1, os.cpu_count() or 1)); model.to(dev)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True); ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_dir", default="compare_out"); ap.add_argument("--models", default="fno,iresnet50")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--margin", type=float, default=0.4); ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--max_train_people", type=int, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = index_people(a.train_dir, limit=a.max_train_people)
    test = index_people(a.test_dir)
    ppl = list(train)
    items = [(f, i) for i, p in enumerate(ppl) for f in train[p]]
    overlap = set(train) & set(test)
    if overlap:
        print(f"WARNING: {len(overlap)} folder names appear in BOTH train and test (e.g. {sorted(overlap)[:3]}). "
              "Test results will be too optimistic.")
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB | mixed precision on")
    else:
        print("WARNING: running on CPU - install the CUDA build of PyTorch (see instructions)")
    print(f"train: {len(ppl)} people, {len(items)} images | test: {len(test)} people | device {dev}")
    results = {m: run(m, a, list(items), len(ppl), dev) for m in a.models.split(",")}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(results, open(f"{a.out_dir}/results.json", "w"), indent=2)
    keys = [k for k in next(iter(results.values())) if k != "eer_history"]
    print("\n================ RESULTS (same split, same settings) ================")
    print(f"{'metric':28s}" + "".join(f"{m:>18s}" for m in results))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{str(results[m][k]):>18s}" for m in results))
    print(f"\nsaved to {a.out_dir}/results.json   (lower EER / FNMR is better)")


if __name__ == "__main__":
    main()