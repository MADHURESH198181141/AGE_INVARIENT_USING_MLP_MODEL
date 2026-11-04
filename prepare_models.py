"""Copy the two buffalo_l files the service needs from the insightface cache into ./models/buffalo_l.
Run once on your computer (the cache exists because you already ran insightface):  python prepare_models.py"""
import os, shutil, sys
src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.insightface/models/buffalo_l")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "buffalo_l"); os.makedirs(dst, exist_ok=True)
for f in ("det_10g.onnx", "w600k_r50.onnx"):
    if not os.path.exists(os.path.join(src, f)): raise SystemExit(f"missing {os.path.join(src, f)}")
    shutil.copy(os.path.join(src, f), os.path.join(dst, f)); print("copied", f)
print("done ->", dst)
