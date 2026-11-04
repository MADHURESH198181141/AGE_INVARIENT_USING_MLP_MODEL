"""Face verification page: upload two photos, get a similarity score from the SAME pipeline the service uses
(buffalo_l + optional MLP adapter, offline, single-threaded).  Photos never leave your computer.

    pip install streamlit
    streamlit run streamlit_app.py
Run it inside the service folder (needs models/buffalo_l/*.onnx; adapter/mlp_adapter.pth is optional)."""
import os, time
import cv2, numpy as np
import streamlit as st
from face_engine import Engine

st.set_page_config(page_title="Age-gap face verification", page_icon="🧑", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_engine():
    return Engine()


def read(f):
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read {f.name} as an image")
    return img


STAGE_TEXT = {0: "face found directly", 1: "face found after adding a border (tight crop)",
              2: "found after 2x upscale", 3: "found with a relaxed detector", 4: "found after contrast enhancement",
              5: "NO FACE FOUND - whole image was used, result is unreliable"}

eng = load_engine()
has_adapter = eng.adapter is not None

st.title("Age-gap face verification")
st.caption("buffalo_l (SCRFD detector + ArcFace ResNet-50) with an optional MLP adapter trained on aging pairs. "
           "Runs entirely on this computer.")

with st.sidebar:
    st.header("Settings")
    use_adapter = st.toggle("Use MLP adapter", value=has_adapter, disabled=not has_adapter,
                            help="adapter/mlp_adapter.pth found" if has_adapter else "No adapter file found: plain buffalo_l")
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    st.caption("A demo cut-off only. The UIDAI portal ranks the raw scores over many pairs, it does not use a fixed "
               "threshold. Large age gaps often give genuine scores of 0.3 to 0.5, still far above strangers (about 0.0 to 0.2).")
    if st.button("Clear history"):
        st.session_state.pop("history", None)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Photo 1 (e.g. old / child)")
    f1 = st.file_uploader("Upload photo 1", type=["jpg", "jpeg", "png", "bmp", "webp"], key="f1")
with c2:
    st.subheader("Photo 2 (e.g. recent)")
    src = st.radio("Source", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
    f2 = (st.file_uploader("Upload photo 2", type=["jpg", "jpeg", "png", "bmp", "webp"], key="f2")
          if src == "Upload" else st.camera_input("Take a photo"))

if f1 is not None and f2 is not None:
    try:
        img1, img2 = read(f1), read(f2)
        t0 = time.time()
        with st.spinner("Comparing..."):
            r = eng.compare(img1, img2)
        score = r["adapted"] if (use_adapter and r["adapted"] is not None) else r["plain"]
        st.divider()
        m1, m2, m3 = st.columns([1, 1, 2])
        m1.metric("Similarity score", f"{score:.3f}")
        m2.metric("Threshold", f"{thr:.2f}")
        with m3:
            if any(i["stage"] == 5 for i in r["items"]):
                st.warning("No face was detected in at least one photo. The score below is not reliable.")
            elif score >= thr:
                st.success("MATCH: same person (at this threshold)")
            else:
                st.error("NO MATCH: different people (at this threshold)")
            st.progress(float(min(max(score, 0.0), 1.0)))
        a1, a2 = st.columns(2)
        for col, name, img, item in ((a1, f1.name, img1, r["items"][0]), (a2, f2.name, img2, r["items"][1])):
            with col:
                s1, s2 = st.columns(2)
                s1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{name}", width="stretch")
                s2.image(cv2.cvtColor(item["aligned"], cv2.COLOR_BGR2RGB), caption="aligned 112x112 face the model sees",
                         width="stretch")
                st.caption(f"Detection: {STAGE_TEXT[item['stage']]} | {item['seconds']:.2f}s")
        with st.expander("Details"):
            st.write(f"- Plain buffalo_l score: **{r['plain']:.4f}**")
            if r["adapted"] is not None:
                st.write(f"- With MLP adapter: **{r['adapted']:.4f}**   (difference {r['adapted'] - r['plain']:+.4f})")
            st.write(f"- Total time: {time.time() - t0:.2f}s for both photos (limit: enroll 10 s per photo, match 1 s)")
        hist = st.session_state.setdefault("history", [])
        key = (f1.name, f2.name, round(score, 4), use_adapter)
        if not hist or hist[-1].get("key") != key:
            hist.append(dict(key=key, photo1=f1.name, photo2=f2.name, score=round(score, 4),
                             adapter="on" if (use_adapter and r["adapted"] is not None) else "off",
                             result="match" if score >= thr else "no match"))
    except Exception as ex:
        st.error(f"Could not process these photos: {ex}")
else:
    st.info("Upload both photos to see the similarity score.")

if st.session_state.get("history"):
    st.subheader("This session")
    st.dataframe([{k: v for k, v in h.items() if k != "key"} for h in st.session_state["history"]],
                 width="stretch", hide_index=True)