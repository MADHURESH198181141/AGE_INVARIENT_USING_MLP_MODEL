"""UIDAI face-matching service.  POST /enroll (image -> embedding)   POST /match (two embeddings -> score)
>>> The README.md inside the portal's SAMPLE SUBMISSION is the authority on field names and response format.
>>> Change ONLY the constants below to match it.  Everything else stays the same."""
import base64, os, traceback
import cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from face_engine import Engine

# ---------------- portal-format settings (EDIT to match the sample submission's README) ----------------
ENROLL_FILE_FIELD = "file"          # multipart field holding the image
MATCH_FIELD_1, MATCH_FIELD_2 = "embedding1", "embedding2"
RESP_EMBEDDING, RESP_SCORE, RESP_STATUS, RESP_ERROR = "embedding", "score", "status", "error"
STATUS_OK, STATUS_FAIL = 0, 1
# --------------------------------------------------------------------------------------------------------
ENGINE = Engine()
app = FastAPI()


def _fail(msg, code=200):
    return JSONResponse({RESP_STATUS: STATUS_FAIL, RESP_ERROR: msg}, status_code=code)


@app.post("/enroll")
async def enroll(request: Request, file: UploadFile = File(None)):
    try:
        data = await file.read() if file is not None else await request.body()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return _fail("invalid image")
        emb, stage = ENGINE.embed(img)
        return {RESP_STATUS: STATUS_OK, RESP_EMBEDDING: base64.b64encode(emb.tobytes()).decode(), "stage": stage}
    except Exception as ex:                                 # always return a code, never crash
        traceback.print_exc(); return _fail(str(ex))


@app.post("/match")
async def match(body: dict):
    try:
        a, b = (np.frombuffer(base64.b64decode(body[k]), np.float32) for k in (MATCH_FIELD_1, MATCH_FIELD_2))
        if a.shape != b.shape or a.size != 512:
            return _fail("bad embedding size")
        return {RESP_STATUS: STATUS_OK, RESP_SCORE: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))}
    except Exception as ex:
        traceback.print_exc(); return _fail(str(ex))
