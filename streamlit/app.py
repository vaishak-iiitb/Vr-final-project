"""
Visual Product Search — Streamlit Demo
Run: streamlit run app.py

Requirements: pip install streamlit requests pillow
"""

import streamlit as st
import requests
import base64
import io
import json
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — paste your ngrok URL here after running the Kaggle backend notebook
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BACKEND_URL = "https://YOUR-NGROK-URL.ngrok-free.app"   # ← update this
TOP_K               = 10

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main {
    background-color: #0d0d0d;
    color: #f0ede8;
}

/* Header */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f0ede8;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Step badges */
.step-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #c8f060;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
}

/* Result cards */
.result-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #c8f060;
}
.result-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-id {
    font-size: 0.78rem;
    font-weight: 600;
    color: #f0ede8;
    margin: 2px 0;
    word-break: break-all;
}
.result-caption {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #666;
    margin-top: 4px;
    line-height: 1.4;
}
.score-pill {
    display: inline-block;
    background: #1e2a0e;
    color: #c8f060;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
    margin-top: 6px;
}
.score-pill-dim {
    display: inline-block;
    background: #1a1a1a;
    color: #555;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
    margin-top: 6px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #1a1a1a;
}

/* Divider */
.vr-divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1.5rem 0;
}

/* Status bar */
.status-bar {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    padding: 6px 0;
    border-top: 1px solid #1a1a1a;
    margin-top: 2rem;
}
.status-ok   { color: #c8f060; }
.status-err  { color: #ff5f5f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_pil(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")

def call_crop(backend_url: str, pil_img: Image.Image):
    payload  = {"image": pil_to_b64(pil_img)}
    response = requests.post(f"{backend_url}/crop", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def call_search(backend_url: str, pil_img: Image.Image, top_k: int):
    payload  = {"image": pil_to_b64(pil_img), "top_k": top_k}
    response = requests.post(f"{backend_url}/search", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()

def check_health(backend_url: str):
    try:
        r = requests.get(f"{backend_url}/health", timeout=8)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "stage": "upload",          # upload → cropped → results
    "original_img": None,
    "cropped_img": None,
    "search_results": None,
    "query_caption": "",
    "backend_ok": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">⚙ Settings</div>', unsafe_allow_html=True)
    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)

    backend_url = st.text_input(
        "Backend URL (ngrok)",
        value=DEFAULT_BACKEND_URL,
        help="Paste the ngrok URL from your Kaggle backend notebook here.",
    ).rstrip("/")

    top_k = st.slider("Top-K results", min_value=5, max_value=20, value=TOP_K, step=1)

    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)

    # Health check
    if st.button("🔌 Check Backend Connection", use_container_width=True):
        with st.spinner("Checking..."):
            ok, info = check_health(backend_url)
            st.session_state.backend_ok = ok
        if ok:
            st.success(f"Connected ✓ | Gallery: {info.get('gallery_size', '?'):,} vectors")
        else:
            st.error(f"Failed: {info.get('error', 'Unknown error')}")

    if st.session_state.backend_ok is True:
        st.markdown('<span class="status-ok">● Backend online</span>', unsafe_allow_html=True)
    elif st.session_state.backend_ok is False:
        st.markdown('<span class="status-err">● Backend offline</span>', unsafe_allow_html=True)

    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#333;line-height:1.8">
PIPELINE<br>
YOLO → crop<br>
CLIP ViT-L/14 → embed<br>
FAISS HNSW → ANN search<br>
BLIP-2 ITM → re-rank
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Visual Product Search</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">DeepFashion In-Shop · CLIP + BLIP-2 · FAISS HNSW</div>', unsafe_allow_html=True)

# ── STAGE 1: Upload ───────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 01 — Upload Query Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a clothing image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    pil_uploaded = Image.open(uploaded).convert("RGB")

    # Reset state if a new image is uploaded
    if st.session_state.original_img is None or \
       pil_to_b64(pil_uploaded) != pil_to_b64(st.session_state.original_img):
        st.session_state.original_img  = pil_uploaded
        st.session_state.cropped_img   = None
        st.session_state.search_results = None
        st.session_state.query_caption  = ""
        st.session_state.stage          = "upload"

col_orig, col_crop, col_pad = st.columns([1, 1, 2])

with col_orig:
    if st.session_state.original_img:
        st.image(st.session_state.original_img, caption="Original", use_container_width=True)

# ── STAGE 2: Crop ─────────────────────────────────────────────────────────────
st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-badge">Step 02 — YOLO Crop & Confirm</div>', unsafe_allow_html=True)

if st.session_state.original_img is not None and st.session_state.stage == "upload":
    if st.button("🔍 Detect & Crop Product", type="primary", use_container_width=False):
        with st.spinner("Running YOLO detection on Kaggle backend..."):
            try:
                resp = call_crop(backend_url, st.session_state.original_img)
                if "error" in resp:
                    st.error(f"Backend error: {resp['error']}")
                else:
                    crop_pil = b64_to_pil(resp["crop"])
                    st.session_state.cropped_img = crop_pil
                    st.session_state.stage = "cropped"
                    had_det = resp.get("had_detection", False)
                    if had_det:
                        st.success(f"Detection found ✓  |  Crop size: {resp['crop_size'][0]}×{resp['crop_size'][1]}px")
                    else:
                        st.warning("No detection above threshold — using full image as crop.")
                    st.rerun()
            except Exception as e:
                st.error(f"Connection error: {e}")

if st.session_state.cropped_img is not None:
    with col_crop:
        st.image(st.session_state.cropped_img, caption="YOLO Crop", use_container_width=True)

    col_confirm, col_recrop = st.columns([1, 1])
    with col_confirm:
        if st.button("✅ Confirm Crop — Search", type="primary", use_container_width=True,
                     disabled=(st.session_state.stage == "results")):
            st.session_state.stage = "searching"
            st.rerun()
    with col_recrop:
        if st.button("🔄 Re-crop / Use Full Image", use_container_width=True):
            st.session_state.cropped_img    = st.session_state.original_img.copy()
            st.session_state.search_results = None
            st.session_state.stage          = "cropped"
            st.rerun()

# ── STAGE 3: Search ───────────────────────────────────────────────────────────
if st.session_state.stage == "searching":
    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">Step 03 — Retrieval + BLIP-2 ITM Re-ranking</div>', unsafe_allow_html=True)

    with st.spinner(
        f"Embedding query → FAISS search (gallery: all vectors) → BLIP-2 ITM re-ranking top {top_k}…"
        "\n\n*This may take 30–90 seconds on Kaggle.*"
    ):
        try:
            resp = call_search(backend_url, st.session_state.cropped_img, top_k)
            if "error" in resp:
                st.error(f"Search error: {resp['error']}")
                st.code(resp.get("trace", ""), language="python")
                st.session_state.stage = "cropped"
            else:
                st.session_state.search_results = resp["results"]
                st.session_state.query_caption  = resp.get("query_caption", "")
                st.session_state.stage          = "results"
                st.rerun()
        except Exception as e:
            st.error(f"Connection error: {e}")
            st.session_state.stage = "cropped"

# ── STAGE 4: Results ──────────────────────────────────────────────────────────
if st.session_state.stage == "results" and st.session_state.search_results:
    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">Step 04 — Results</div>', unsafe_allow_html=True)

    results = st.session_state.search_results

    # Query caption
    if st.session_state.query_caption:
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
            f'color:#888;margin-bottom:1rem">Query caption: '
            f'<span style="color:#c8f060">{st.session_state.query_caption}</span></div>',
            unsafe_allow_html=True,
        )

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Results returned", len(results))
    if results:
        m2.metric("Top cosine score", f"{results[0]['cosine_score']:.4f}")
        m3.metric("Top combined score", f"{results[0]['combined_score']:.4f}")

    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)

    # Grid: 5 columns
    COLS_PER_ROW = 5
    for row_start in range(0, len(results), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col_i, result in enumerate(results[row_start: row_start + COLS_PER_ROW]):
            with cols[col_i]:
                # Gallery image
                if result.get("image"):
                    img_pil = b64_to_pil(result["image"])
                    st.image(img_pil, use_container_width=True)
                else:
                    st.markdown("*(no image)*")

                # Metadata card
                st.markdown(f"""
<div class="result-card">
  <div class="result-rank">Rank #{result['rank']}</div>
  <div class="result-id">{result['item_id']}</div>
  <span class="score-pill">cos {result['cosine_score']:.3f}</span>
  <span class="score-pill-dim">itm {result['itm_score']:.3f}</span>
  <span class="score-pill">⭐ {result['combined_score']:.3f}</span>
  <div class="result-caption">{result['caption'][:80]}{'…' if len(result['caption']) > 80 else ''}</div>
</div>
""", unsafe_allow_html=True)

    # Download results as JSON
    st.markdown('<hr class="vr-divider">', unsafe_allow_html=True)
    col_dl, col_reset = st.columns([1, 1])
    with col_dl:
        json_str = json.dumps({
            "query_caption": st.session_state.query_caption,
            "results": [
                {k: v for k, v in r.items() if k != "image"}
                for r in results
            ]
        }, indent=2)
        st.download_button(
            label="⬇ Download Results JSON",
            data=json_str,
            file_name="search_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_reset:
        if st.button("🔁 New Search", use_container_width=True):
            for key in ["original_img", "cropped_img", "search_results", "query_caption"]:
                st.session_state[key] = None
            st.session_state.query_caption = ""
            st.session_state.stage = "upload"
            st.rerun()

# ── Status bar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="status-bar">
  Visual Recognition Project · Config C · CLIP ViT-L/14 (fine-tuned) · BLIP-2 2.7B · α=0.7
  &nbsp;|&nbsp; Stage: {st.session_state.stage}
</div>
""", unsafe_allow_html=True)
