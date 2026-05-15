# Visual Product Search — Demo Setup Guide

## Architecture

```
[Local Machine]                    [Kaggle Kernel]
  Streamlit app  ←── HTTP/JSON ──→  Flask API + ngrok tunnel
     app.py                          kaggle_backend_server.ipynb
```

---

## Step 1 — Set up the Kaggle Backend

1. Open `kaggle_backend_server.ipynb` in your Kaggle kernel.

2. **Edit Section 1 (Configuration):**
   - Verify all paths match your Kaggle dataset slugs:
     - `GALLERY_INDEX_PATH` → your saved FAISS index from Config C
     - `CLIP_CKPT_PATH`     → your best checkpoint `.pt` file
     - `CAPTION_CACHE_PATH` → your `caption_cache_C.json`
     - `YOLO_WEIGHTS`       → your YOLO `.pt` file
   - Set `SEED` to whichever seed produced your best results (default: 2023085)

3. **Get a free ngrok token:**
   - Go to https://dashboard.ngrok.com/get-started/your-authtoken
   - Sign up (free), copy your auth token
   - Paste it into `NGROK_AUTH_TOKEN = 'YOUR_TOKEN_HERE'`

4. **Run all cells top to bottom.**
   - Cells 0–8 load models (~5 min total, BLIP-2 takes longest)
   - Cell 9 defines Flask routes
   - Cell 10 starts the server and prints your public URL, e.g.:
     ```
     BACKEND URL: https://abc123.ngrok-free.app
     ```
   - **Keep the kernel running** while using the demo.

---

## Step 2 — Set up the Local Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Step 3 — Connect the Two

1. In the Streamlit sidebar, paste the ngrok URL from Step 1 into **"Backend URL"**
2. Click **"Check Backend Connection"** — you should see "Connected ✓"
3. Start searching!

---

## Usage Flow

| Step | Action | What happens |
|------|--------|-------------|
| 1 | Upload a clothing image | Image displayed |
| 2 | Click "Detect & Crop Product" | YOLO runs on Kaggle, crop returned |
| 3 | Confirm crop (or re-crop) | Proceeds to search |
| 4 | Search runs automatically | CLIP embed → FAISS ANN → BLIP-2 ITM re-rank |
| 5 | Top-K results displayed | Images + item IDs + scores |

---

## API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/health` | GET | — | `{status, device, gallery_size}` |
| `/crop` | POST | `{image: base64}` | `{crop: base64, had_detection: bool, ...}` |
| `/search` | POST | `{image: base64, top_k: int}` | `{results: [...], query_caption: str}` |

### Search result item schema
```json
{
  "rank": 1,
  "item_id": "id_00001001",
  "cosine_score": 0.8821,
  "itm_score": -1.234,
  "combined_score": 0.712,
  "caption": "a blue denim jacket with button front...",
  "image": "<base64 JPEG>"
}
```

---

## ITM Re-ranking Details

The system fetches `ITM_TOP_N=30` candidates from FAISS, then re-ranks them:

1. **Cosine score** — from FAISS inner product on L2-normalised CLIP embeddings
2. **ITM score** — BLIP-2 teacher-forcing log-likelihood of each gallery caption given the query image (higher = better semantic match)
3. **Combined score** = `0.6 × norm(cosine) + 0.4 × norm(ITM)`
4. Return top-K by combined score

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection error` in Streamlit | Check ngrok URL is current (tunnel resets on kernel restart) |
| Search takes > 2 min | Normal for BLIP-2 ITM on 30 candidates; reduce `ITM_TOP_N` in notebook |
| `had_detection: false` | YOLO found no box; full image is used as crop — still valid |
| OOM on Kaggle | BLIP-2 2.7B + CLIP ViT-L/14 together use ~18GB; dual T4 required |
| Caption cache path wrong | Check if you ran Config B or C first; path may be `caption_cache.json` |
