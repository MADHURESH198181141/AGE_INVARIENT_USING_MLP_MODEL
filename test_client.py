"""Test the running service with two photos.   Open a SECOND PowerShell window (the first one runs the server) and run:
    python test_client.py photo1.jpg photo2.jpg
    python test_client.py photo1.jpg photo2.jpg --url http://localhost:8000      (if you started the server on another port)"""
import argparse, sys, time
try:
    import requests
except ImportError:
    sys.exit("run:  pip install requests")

ap = argparse.ArgumentParser()
ap.add_argument("image1"); ap.add_argument("image2"); ap.add_argument("--url", default="http://localhost")
a = ap.parse_args()
try:
    emb = []
    for p in (a.image1, a.image2):
        t = time.time()
        r = requests.post(a.url + "/enroll", files={"file": open(p, "rb")}, timeout=60).json()
        print(f"/enroll {p}: status={r.get('status')} stage={r.get('stage')} time={time.time()-t:.2f}s {r.get('error', '')}")
        emb.append(r["embedding"])
    t = time.time()
    m = requests.post(a.url + "/match", json={"embedding1": emb[0], "embedding2": emb[1]}, timeout=60).json()
    print(f"/match: {m}  time={time.time()-t:.3f}s")
    if "score" in m:
        print("=> SAME person" if m["score"] >= 0.5 else "=> DIFFERENT person", "(demo threshold 0.50; the portal ranks by score)")
except requests.exceptions.ConnectionError:
    sys.exit(f"Cannot reach {a.url}. Is the server window still running? Try --url http://localhost:8000")

