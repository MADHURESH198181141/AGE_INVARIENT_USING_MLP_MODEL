"""Split a dataset by PERSON into train/test (no person appears in both). Originals untouched (files are copied).
Accepts  DATA/<person>/<images>   or a FLAT folder of FG-NET names (001A02.JPG -> person 001).
Output:  OUT/train/<person>/..., OUT/test/<person>/...
Usage:   python split_data.py DATA OUT [test_fraction=0.3]"""
import sys, os, re, shutil, random

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
src, out = sys.argv[1], sys.argv[2]
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
if not os.path.isdir(src):
    raise SystemExit(f"ERROR: folder not found: {os.path.abspath(src)}")
people = {}
for dp, _, fs in os.walk(src):
    for f in fs:
        if os.path.splitext(f)[1].lower() not in EXT:
            continue
        if os.path.abspath(dp) == os.path.abspath(src):
            m = re.match(r"^(\d+)[A-Za-z]", f)
            if not m: continue
            key = m.group(1)
        else:
            key = os.path.relpath(dp, src).split(os.sep)[0]
        people.setdefault(key, []).append(os.path.join(dp, f))
people = {k: sorted(v) for k, v in people.items() if len(v) >= 2}
if not people:
    raise SystemExit(f"No people with >=2 images found in {os.path.abspath(src)}\n"
                     f"First items in that folder: {os.listdir(src)[:8]}\n"
                     "Expected <folder>/<person>/<images> or flat files named like 001A02.JPG")
if os.path.exists(out) and os.listdir(out):
    raise SystemExit(f"{out} already exists and is not empty - choose a new folder name")
keys = sorted(people); random.Random(0).shuffle(keys)
nt = max(1, round(len(keys) * frac))
for i, k in enumerate(keys):
    d = os.path.join(out, "test" if i < nt else "train", k)
    os.makedirs(d, exist_ok=True)
    for f in people[k]:
        shutil.copy(f, os.path.join(d, os.path.basename(f)))
print(f"{len(keys)-nt} train people, {nt} test people -> {out}  (no person is in both)")