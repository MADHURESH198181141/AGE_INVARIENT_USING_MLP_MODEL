# UIDAI age-variation face matching service (buffalo_l + optional MLP adapter)

Single-threaded, offline, CPU. ~0.8 s per /enroll, ~0.005 s per /match on one thread (limits: 10 s / 1 s).

## Set up (once, on your computer)
1. `pip install -r requirements.txt`
2. `python prepare_models.py`          copies det_10g.onnx + w600k_r50.onnx from your insightface cache into models/buffalo_l/
3. (optional) copy `adapter_out2/mlp_adapter.pth` into `adapter/mlp_adapter.pth`.
   Present -> used automatically.  Absent  -> plain buffalo_l.  Set ADAPTER_PATH to use another location.

## Run
    python -m uvicorn app:app --host 0.0.0.0 --port 80 --workers 1

## Before submitting
- Open the README.md in the portal's SAMPLE SUBMISSION and edit the constants at the top of app.py
  (field names, response keys, status codes). That file is the authority, not this one.
- The portal needs a Docker image served on port 80. The model files in models/ must be inside the image
  (nothing may be downloaded at evaluation time).

## Behaviour
- /enroll: detects the face (retries with border/upscale/enhance for tight or poor scans), aligns, returns a base64
  float32[512] embedding and the rescue `stage` (0-1 normal, 2-4 rescued, 5 = no face found, unaligned fallback).
- /match: cosine similarity of two embeddings.
- Every error returns a JSON status code instead of crashing.
"# AGE_INVARIENT_FACE_RECOGNITION_USING_MLP" 
