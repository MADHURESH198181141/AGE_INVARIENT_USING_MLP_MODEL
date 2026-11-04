"""Face engine: buffalo_l (SCRFD-10G detector + w600k_r50 ArcFace recogniser) + optional adapter.
SINGLE-THREADED, fully OFFLINE (loads model files from MODEL_DIR; nothing is downloaded).
Never raises on a hard image: it falls back through detection rescue stages (see extract_embeddings.py)."""
import glob, math, os, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # insightface's harmless 'estimate is deprecated' notice
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2, numpy as np
import onnxruntime as ort
cv2.setNumThreads(1)
from insightface import model_zoo
from insightface.utils import face_align

EMB_DIM = 512
_ERF = np.vectorize(math.erf)


def _enhance(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(im, 1.5, cv2.GaussianBlur(im, (0, 0), 2.0), -0.5, 0)


class Adapter:
    """Residual MLP adapter trained by adapter_cv.py:  z = normalize(e + alpha * MLP(e)).  numpy only (no torch)."""
    def __init__(self, path):
        import torch                                   # only used to read the .pth file
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b["cfg"].get("kind") != "mlp":
            raise ValueError("only the MLP adapter is supported in the service")
        sd = {k: v.numpy().astype(np.float32) for k, v in b["state_dict"].items()}
        self.w1, self.b1, self.w2, self.b2 = sd["net.0.weight"], sd["net.0.bias"], sd["net.2.weight"], sd["net.2.bias"]
        self.alpha = float(sd["alpha"].reshape(-1)[0])

    def __call__(self, e):
        h = e @ self.w1.T + self.b1
        h = 0.5 * h * (1.0 + _ERF(h / math.sqrt(2.0)))      # exact GELU
        z = e + self.alpha * (h @ self.w2.T + self.b2)
        return z / (np.linalg.norm(z) + 1e-9)


class Engine:
    def __init__(self, model_dir=None, adapter_path=None):
        model_dir = model_dir or os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models", "buffalo_l"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.det = self.rec = None
        self.last_aligned = None
        for f in sorted(glob.glob(os.path.join(model_dir, "*.onnx"))):
            m = model_zoo.get_model(f, sess_options=so, providers=["CPUExecutionProvider"])
            if getattr(m, "taskname", None) == "detection": self.det = m
            elif getattr(m, "taskname", None) == "recognition": self.rec = m
        if self.det is None or self.rec is None:
            raise RuntimeError(f"need a detection and a recognition .onnx in {model_dir}")
        self.det.prepare(-1, input_size=(640, 640), det_thresh=0.5)
        default_ad = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter", "mlp_adapter.pth")
        adapter_path = adapter_path if adapter_path is not None else os.environ.get("ADAPTER_PATH", default_ad)
        self.adapter = Adapter(adapter_path) if adapter_path and os.path.exists(adapter_path) else None

    def _largest(self, im, thr):
        self.det.det_thresh = thr
        bboxes, kpss = self.det.detect(im, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            return None
        i = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
        return kpss[i]

    def _feat(self, aligned112):
        v = np.asarray(self.rec.get_feat(aligned112), np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_plain(self, img):
        """BGR uint8 image -> (512-d buffalo_l embedding WITHOUT adapter, stage 0-5). Stage 5 = no face found.
        Also stores the aligned 112x112 crop in self.last_aligned."""
        stages = [(0.0, 1, 0.5, False), (0.5, 1, 0.5, False), (0.5, 2, 0.5, False), (0.5, 2, 0.2, False), (0.5, 2, 0.2, True)]
        for si, (pad, up, thr, enh) in enumerate(stages):
            im = img
            if up != 1: im = cv2.resize(im, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            if enh: im = _enhance(im)
            if pad > 0:
                p = int(pad * max(im.shape[:2]))
                im = cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=(127, 127, 127))
            kps = self._largest(im, thr)
            if kps is not None:
                self.last_aligned = face_align.norm_crop(im, landmark=kps, image_size=112)
                e, st = self._feat(self.last_aligned), si
                break
        else:
            self.last_aligned = cv2.resize(_enhance(img), (112, 112), interpolation=cv2.INTER_AREA)
            e, st = self._feat(self.last_aligned), 5
        return e.astype(np.float32), st

    def embed(self, img):
        """BGR uint8 image -> (512 float32 L2-normalised embedding incl. adapter if loaded, stage 0-5)."""
        e, st = self.embed_plain(img)
        if self.adapter is not None:
            e = self.adapter(e).astype(np.float32)
        return e.astype(np.float32), st

    def compare(self, img1, img2):
        """Everything the demo page needs for one pair of photos."""
        items = []
        for img in (img1, img2):
            t = time.time(); e, st = self.embed_plain(img)
            items.append(dict(e=e, stage=st, aligned=self.last_aligned.copy(), seconds=time.time() - t))
        out = dict(plain=float(items[0]["e"] @ items[1]["e"]), adapted=None, items=items)
        if self.adapter is not None:
            a = [self.adapter(i["e"]) for i in items]
            out["adapted"] = float(a[0] @ a[1])
        return out