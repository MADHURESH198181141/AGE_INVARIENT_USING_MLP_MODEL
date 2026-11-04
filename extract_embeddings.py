"""STEP 1 - run on YOUR computer. Turns photos into buffalo_l (ArcFace ResNet-100) embeddings.
Photos never leave your machine; only the small .npz of numbers is used by step 2.

Accepts  FOLDER/<person>/<images>   or flat FG-NET names (001A02.JPG -> person 001, age 02).

  pip install insightface onnxruntime        (GPU: onnxruntime-gpu instead of onnxruntime)
  python extract_embeddings.py --images FGNET/images --out fgnet_emb.npz

Rescue cascade (an image is tried at each stage until buffalo_l finds a face):
  0 as-is | 1 + grey border (faces that fill the frame) | 2 + 2x upscale | 3 + low detector threshold
  4 + contrast/sharpen (old, faded or blurry scans) | 5 NO detector: whole image treated as the face crop
The stage that rescued each image is saved, so results can be reported with and without the weaker stages.
"""
import argparse, os, re, sys
import cv2, numpy as np

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scan(root):
    """-> list of (person, age, path)."""
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: folder not found: {os.path.abspath(root)}")
    rows = []
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() not in EXT:
                continue
            flat = os.path.abspath(dp) == os.path.abspath(root)
            m = re.match(r"^(\d+)[A-Za-z]", f)
            if flat:
                if not m:
                    continue
                person = m.group(1)
            else:
                person = os.path.relpath(dp, root).split(os.sep)[0]
            ma = re.match(r"^\d+[A-Za-z](\d+)", f)       # FG-NET: 001A02 -> age 02
            rows.append((person, float(ma.group(1)) if ma else float("nan"), os.path.join(dp, f)))
    if not rows:
        raise SystemExit(f"ERROR: no images found in {os.path.abspath(root)}  (first items: {os.listdir(root)[:8]})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="embeddings.npz")
    ap.add_argument("--model", default="buffalo_l")
    ap.add_argument("--det_size", type=int, default=640)
    ap.add_argument("--no_fallback", action="store_true", help="skip stage 5 (unaligned whole-image embedding)")
    a = ap.parse_args()

    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=a.model, providers=prov)
    print(f"model {a.model} | {'GPU' if cuda else 'CPU'}")

    rows = scan(a.images)
    rec = app.models["recognition"]
    STAGES = [  # (name, pad fraction, upscale, detector threshold, enhance)
        ("0 as-is", 0.0, 1, 0.5, False), ("1 +border", 0.5, 1, 0.5, False),
        ("2 +upscale", 0.5, 2, 0.5, False), ("3 +low thresh", 0.5, 2, 0.2, False),
        ("4 +enhance", 0.5, 2, 0.2, True)]

    def enhance(im):
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(im, 1.5, cv2.GaussianBlur(im, (0, 0), 2.0), -0.5, 0)   # unsharp mask

    cur = [None]

    def detect(img, pad, up, thr, enh):
        im = img
        if up != 1:
            im = cv2.resize(im, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        if enh:
            im = enhance(im)
        if pad > 0:
            p = int(pad * max(im.shape[:2]))
            im = cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        if cur[0] != thr:                                # re-prepare only when the threshold changes
            app.prepare(ctx_id=0 if cuda else -1, det_thresh=thr, det_size=(a.det_size, a.det_size)); cur[0] = thr
        faces = app.get(im)
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None

    embs, persons, ages, paths, stages, failed = [], [], [], [], [], []
    for k, (person, age, path) in enumerate(rows, 1):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)   # works with any Windows path
        e, st = None, -1
        if img is not None:
            for si, (nm, pad, up, thr, enh) in enumerate(STAGES):
                f = detect(img, pad, up, thr, enh)
                if f is not None:
                    e, st = f.normed_embedding, si; break
            if e is None and not a.no_fallback:                              # stage 5: no detector, no alignment
                x = cv2.resize(enhance(img), (112, 112), interpolation=cv2.INTER_AREA)
                v = np.asarray(rec.get_feat(x)).reshape(-1)
                e, st = v / (np.linalg.norm(v) + 1e-9), 5
        if e is None:
            failed.append(path); continue
        embs.append(np.asarray(e, np.float32)); persons.append(person); ages.append(age); paths.append(path); stages.append(st)
        if k % 100 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)} done, {len(failed)} still without a face", flush=True)
    if not embs:
        raise SystemExit("No faces detected in any image.")
    np.savez(a.out, emb=np.stack(embs), person=np.array(persons), age=np.array(ages, np.float32),
             path=np.array(paths), stage=np.array(stages, np.int8))
    open(a.out + ".failed.txt", "w").write("\n".join(failed))
    names = [x[0] for x in STAGES] + ["5 no-detector (unaligned)"]
    print(f"\nsaved {a.out}: {len(embs)} of {len(rows)} images, {len(set(persons))} people | no embedding for {len(failed)}")
    for i, nm in enumerate(names):
        print(f"  stage {nm:28s} {int(np.sum(np.array(stages) == i)):5d} images")
    print("  (stages 0-1 = normal detections; 2-4 = rescued by upscale/enhance; 5 = unaligned fallback, least reliable)")


if __name__ == "__main__":
    main()
