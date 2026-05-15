"""
batch_eval_folder.py  —  Config C  |  Visual Product Search Engine
=================================================================
Given a folder of query images, runs the full retrieval pipeline
(YOLO crop → CLIP visual+text embed → fuse → HNSW search) and
reports Recall@K, NDCG@K, mAP@K for K ∈ {5, 10, 15}.

Ground-truth item_id resolution (auto-detected, priority order):
  1. Subfolder structure  : <query_folder>/<item_id>/<image>.jpg   (DeepFashion default)
  2. Sidecar CSV/TXT      : <query_folder>/labels.csv  or  labels.txt
                            columns: filename,item_id  (with or without header)
  3. Filename prefix      : id_<item_id>_<anything>.jpg
  4. --no-gt flag         : skip metric computation, just show top-K results

Usage examples
--------------
# Standard DeepFashion subfolder layout
python batch_eval_folder.py --query_folder /path/to/query_images

# Custom sidecar label file
python batch_eval_folder.py --query_folder /path/to/query_images \\
       --label_file /path/to/my_labels.csv

# Override alpha or K values
python batch_eval_folder.py --query_folder /path/to/query_images \\
       --alpha 0.7 --k_values 5 10 15

# No ground truth available — retrieval only, no metrics
python batch_eval_folder.py --query_folder /path/to/query_images --no_gt
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import clip
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Local paths (same directory as this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

CLIP_CKPT   = SCRIPT_DIR / "clip_C_finetuned.pt"
YOLO_CKPT   = SCRIPT_DIR / "yolo_fine_tuned.pt"
CAPTION_CACHE = SCRIPT_DIR / "caption_cache.json"
BBOX_CACHE    = SCRIPT_DIR / "bbox_cache_C.pkl"

# Gallery embedding files  (formatted with alpha string, e.g. "0.7")
GAL_EMBS_TMPL = str(SCRIPT_DIR / "gal_embs_C_{alpha}.npy")
GAL_IDS_TMPL  = str(SCRIPT_DIR / "gal_ids_C_{alpha}.json")

OUTPUT_DIR = SCRIPT_DIR
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Helper: resolve item_id from image path / folder layout
# ---------------------------------------------------------------------------

def _build_label_map_subfolder(query_folder: Path):
    """<query_folder>/<item_id>/<image>  →  {abs_path: item_id}"""
    label_map = {}
    for item_dir in query_folder.iterdir():
        if not item_dir.is_dir():
            continue
        item_id = item_dir.name
        for img_path in item_dir.iterdir():
            if img_path.suffix.lower() in IMG_EXTENSIONS:
                label_map[str(img_path)] = item_id
    return label_map


def _build_label_map_sidecar(query_folder: Path, label_file: Path):
    """CSV / TXT with columns  filename,item_id  (header optional)"""
    label_map = {}
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                parts = line.split()          # try whitespace split
            if len(parts) < 2:
                continue
            fname, item_id = parts[0].strip(), parts[1].strip()
            if fname.lower() == "filename":   # skip header row
                continue
            abs_path = str(query_folder / fname)
            label_map[abs_path] = item_id
    return label_map


def _build_label_map_filename(query_folder: Path):
    """Filenames like  id_0000123_01.jpg  →  item_id = '0000123'"""
    label_map = {}
    for img_path in query_folder.iterdir():
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        stem = img_path.stem          # e.g. "id_0000123_01"
        parts = stem.split("_")
        if len(parts) >= 2:
            item_id = parts[1]        # second token
            label_map[str(img_path)] = item_id
    return label_map


def resolve_label_map(query_folder: Path, label_file=None, no_gt=False):
    """
    Auto-detect ground-truth labels. Returns {abs_image_path: item_id}.
    Returns {} if no_gt=True.
    """
    if no_gt:
        # Collect all images without labels
        return {
            str(p): None
            for p in sorted(query_folder.rglob("*"))
            if p.suffix.lower() in IMG_EXTENSIONS
        }

    # Priority 1: explicit sidecar file
    if label_file:
        lf = Path(label_file)
        if not lf.exists():
            sys.exit(f"[ERROR] Label file not found: {lf}")
        print(f"[INFO] Using sidecar label file: {lf}")
        label_map = _build_label_map_sidecar(query_folder, lf)
        if label_map:
            return label_map
        sys.exit("[ERROR] Label file exists but could not be parsed. "
                 "Expected CSV/TXT with columns: filename,item_id")

    # Priority 2: auto-detect sidecar in query folder
    for sidecar_name in ("labels.csv", "labels.txt", "ground_truth.csv", "gt.csv"):
        sidecar = query_folder / sidecar_name
        if sidecar.exists():
            print(f"[INFO] Auto-detected sidecar: {sidecar}")
            label_map = _build_label_map_sidecar(query_folder, sidecar)
            if label_map:
                return label_map

    # Priority 3: subfolder structure
    subdirs = [d for d in query_folder.iterdir() if d.is_dir()]
    if subdirs:
        label_map = _build_label_map_subfolder(query_folder)
        if label_map:
            print("[INFO] Ground truth resolved from subfolder structure.")
            return label_map

    # Priority 4: filename prefix
    label_map = _build_label_map_filename(query_folder)
    if label_map:
        print("[INFO] Ground truth resolved from filename prefix.")
        return label_map

    sys.exit(
        "[ERROR] Could not resolve item_id for query images. Please use one of:\n"
        "  (A) Subfolder layout:  <query_folder>/<item_id>/<image>.jpg\n"
        "  (B) Sidecar file    :  <query_folder>/labels.csv  (filename,item_id)\n"
        "  (C) Filename prefix :  id_<item_id>_<anything>.jpg\n"
        "  (D) Pass --no_gt to skip metrics and just retrieve."
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_clip(ckpt_path):
    print("[INFO] Loading fine-tuned CLIP (ViT-B/32) …")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.data = p.data.float()
    print("[INFO] CLIP loaded.")
    return model, preprocess


def load_yolo(ckpt_path):
    print("[INFO] Loading YOLO …")
    yolo = YOLO(str(ckpt_path))
    print("[INFO] YOLO loaded.")
    return yolo


def load_caption_cache(cache_path):
    print("[INFO] Loading caption cache …")
    with open(cache_path) as f:
        cache = json.load(f)
    print(f"[INFO] Caption cache: {len(cache):,} entries.")
    return cache


def load_bbox_cache(cache_path):
    print("[INFO] Loading bbox cache …")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"[INFO] Bbox cache: {len(cache):,} entries.")
    return cache

# ---------------------------------------------------------------------------
# YOLO crop helper
# ---------------------------------------------------------------------------

def yolo_crop(image: Image.Image, yolo, bbox_cache: dict, abs_path: str):
    """
    Return a cropped PIL image for the primary clothing item.
    Uses bbox_cache when available; falls back to live YOLO inference.
    Falls back to the full image if YOLO finds nothing.
    """
    # Try cache first
    bbox = bbox_cache.get(abs_path)

    if bbox is None:
        # Live inference
        results = yolo(image, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Pick highest-confidence detection
            conf = boxes.conf.cpu().numpy()
            best = int(np.argmax(conf))
            x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
            bbox = (x1, y1, x2, y2)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Clamp to image dimensions
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            return image.crop((x1, y1, x2, y2))

    return image   # fallback: full image

# ---------------------------------------------------------------------------
# Embedding extraction for query images
# ---------------------------------------------------------------------------

def extract_query_embeddings(image_paths, label_map,
                              clip_model, clip_preprocess,
                              yolo, caption_cache, bbox_cache,
                              alpha, batch_size=64):
    """
    Returns:
        qry_embs  : np.ndarray, shape (N, 512), float32, L2-normed
        qry_ids   : list of item_id strings (or None if no_gt)
        skipped   : list of paths that failed to load
    """
    qry_embs, qry_ids, skipped = [], [], []

    paths = list(image_paths)

    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding queries"):
        batch_paths = paths[i : i + batch_size]

        vis_tensors, txt_tokens, valid_ids, valid_paths = [], [], [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except (UnidentifiedImageError, OSError):
                skipped.append(p)
                continue

            # YOLO crop
            cropped = yolo_crop(img, yolo, bbox_cache, p)

            # CLIP visual preprocessing
            vis_tensors.append(clip_preprocess(cropped))

            # Caption  (generate on-the-fly if not cached)
            caption = caption_cache.get(p, "a photo of a clothing item")
            txt_tokens.append(clip.tokenize([caption], truncate=True).squeeze(0))

            valid_paths.append(p)
            valid_ids.append(label_map.get(p))   # None if no_gt

        if not vis_tensors:
            continue

        vis_batch = torch.stack(vis_tensors).to(DEVICE)
        txt_batch = torch.stack(txt_tokens).to(DEVICE)

        with torch.no_grad():
            vis_emb = F.normalize(clip_model.encode_image(vis_batch).float(), dim=-1)
            txt_emb = F.normalize(clip_model.encode_text(txt_batch).float(), dim=-1)
            fused   = F.normalize(alpha * vis_emb + (1 - alpha) * txt_emb, dim=-1)

        qry_embs.append(fused.cpu().numpy())
        qry_ids.extend(valid_ids)

    if not qry_embs:
        sys.exit("[ERROR] No query images could be processed.")

    return np.vstack(qry_embs).astype(np.float32), qry_ids, skipped

# ---------------------------------------------------------------------------
# Gallery index builder
# ---------------------------------------------------------------------------

def build_hnsw_index(gal_embs: np.ndarray):
    print("[INFO] Building HNSW index …")
    index = faiss.IndexHNSWFlat(512, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200
    index.add(gal_embs)
    print(f"[INFO] Index built with {index.ntotal:,} gallery vectors.")
    return index

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def recall_at_k(top_k_ids, true_id, k):
    return int(any(gid == true_id for gid in top_k_ids[:k]))


def ndcg_at_k(top_k_ids, true_id, total_rel, k):
    top = top_k_ids[:k]
    dcg  = sum(1.0 / np.log2(r + 2) for r, gid in enumerate(top) if gid == true_id)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(total_rel, k)))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(top_k_ids, true_id, total_rel, k):
    if total_rel == 0:
        return 0.0
    top  = top_k_ids[:k]
    hits, ap = 0, 0.0
    for r, gid in enumerate(top, start=1):
        if gid == true_id:
            hits += 1
            ap   += hits / r
    return ap / total_rel


def compute_metrics(I, qry_ids, gal_ids, k_values):
    """
    I        : np.ndarray (N_queries, max_k) — FAISS result indices
    qry_ids  : list of true item_ids for each query
    gal_ids  : list of item_ids for each gallery entry
    k_values : list of K values, e.g. [5, 10, 15]
    """
    max_k = max(k_values)
    results = {k: {"recall": [], "ndcg": [], "map": []} for k in k_values}

    # Precompute total relevant per query
    from collections import Counter
    gal_id_counts = Counter(gal_ids)

    for qi, true_id in enumerate(tqdm(qry_ids, desc="Computing metrics")):
        if true_id is None:
            continue
        top_ids = [gal_ids[idx] for idx in I[qi] if 0 <= idx < len(gal_ids)]
        total_rel = gal_id_counts.get(true_id, 0)

        for k in k_values:
            results[k]["recall"].append(recall_at_k(top_ids, true_id, k))
            results[k]["ndcg"].append(ndcg_at_k(top_ids, true_id, total_rel, k))
            results[k]["map"].append(map_at_k(top_ids, true_id, total_rel, k))

    summary = {}
    for k in k_values:
        n = len(results[k]["recall"])
        if n == 0:
            continue
        summary[k] = {
            "n_queries"  : n,
            f"Recall@{k}" : float(np.mean(results[k]["recall"])),
            f"NDCG@{k}"   : float(np.mean(results[k]["ndcg"])),
            f"mAP@{k}"    : float(np.mean(results[k]["map"])),
        }
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Config C retrieval on an arbitrary query folder."
    )
    parser.add_argument("--query_folder", required=True,
                        help="Path to folder containing query images.")
    parser.add_argument("--label_file", default=None,
                        help="(Optional) CSV/TXT mapping filename→item_id. "
                             "Auto-detected if omitted.")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Fusion weight for visual embedding (default: 0.7).")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 15],
                        help="K values for metric evaluation (default: 5 10 15).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for CLIP inference (default: 64).")
    parser.add_argument("--no_gt", action="store_true",
                        help="Skip metric computation (no ground truth available).")
    parser.add_argument("--output_json", default=None,
                        help="Path to save results JSON. "
                             "Default: /kaggle/working/results_folder_eval.json")
    # Path overrides (useful when not on Kaggle)
    parser.add_argument("--clip_ckpt",   default=str(CLIP_CKPT))
    parser.add_argument("--yolo_ckpt",   default=str(YOLO_CKPT))
    parser.add_argument("--caption_cache", default=str(CAPTION_CACHE))
    parser.add_argument("--bbox_cache",    default=str(BBOX_CACHE))
    parser.add_argument("--gal_embs",    default=None,
                        help="Path to gallery embeddings .npy (overrides auto-template).")
    parser.add_argument("--gal_ids",     default=None,
                        help="Path to gallery item IDs .json (overrides auto-template).")
    args = parser.parse_args()

    t0 = time.time()
    query_folder = Path(args.query_folder)
    if not query_folder.exists():
        sys.exit(f"[ERROR] Query folder not found: {query_folder}")

    alpha_str = f"{args.alpha:.1f}".rstrip("0").rstrip(".")  # "0.7" or "0.70" → "0.7"
    # Edge case: alpha=1 → "1", alpha=0.3 → "0.3"
    if "." not in alpha_str:
        alpha_str = alpha_str  # already integer-like

    # ---- Resolve label map ------------------------------------------------
    label_map = resolve_label_map(query_folder, args.label_file, args.no_gt)
    image_paths = sorted(label_map.keys())
    print(f"[INFO] Found {len(image_paths):,} query images.")

    if not image_paths:
        sys.exit("[ERROR] No images found in the query folder.")

    # ---- Load gallery embeddings ------------------------------------------
    gal_embs_path = args.gal_embs or GAL_EMBS_TMPL.format(alpha=alpha_str)
    gal_ids_path  = args.gal_ids  or GAL_IDS_TMPL.format(alpha=alpha_str)

    if not Path(gal_embs_path).exists():
        sys.exit(
            f"[ERROR] Gallery embeddings not found: {gal_embs_path}\n"
            f"        Run extract_gallery_embeddings.py first, or pass --gal_embs."
        )

    print(f"[INFO] Loading gallery embeddings from: {gal_embs_path}")
    gal_embs = np.load(gal_embs_path).astype(np.float32)
    with open(gal_ids_path) as f:
        gal_ids = json.load(f)
    print(f"[INFO] Gallery: {gal_embs.shape[0]:,} vectors, dim={gal_embs.shape[1]}.")

    # ---- Load models -------------------------------------------------------
    clip_model, clip_preprocess = load_clip(args.clip_ckpt)
    yolo        = load_yolo(args.yolo_ckpt)
    caption_cache = load_caption_cache(args.caption_cache)
    bbox_cache    = load_bbox_cache(args.bbox_cache)

    # ---- Extract query embeddings ------------------------------------------
    print(f"\n[INFO] Extracting embeddings for {len(image_paths):,} query images …")
    qry_embs, qry_ids, skipped = extract_query_embeddings(
        image_paths, label_map,
        clip_model, clip_preprocess,
        yolo, caption_cache, bbox_cache,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    print(f"[INFO] Embedded {qry_embs.shape[0]:,} queries. Skipped: {len(skipped)}.")
    if skipped:
        print("[WARN] Skipped images (could not open):")
        for p in skipped:
            print(f"       {p}")

    # ---- Build HNSW index & search ----------------------------------------
    index = build_hnsw_index(gal_embs)
    max_k = max(args.k_values)
    print(f"[INFO] Searching top-{max_k} neighbours …")
    _, I = index.search(qry_embs, max_k)   # I: (N_queries, max_k)

    # ---- Metrics (if ground truth available) --------------------------------
    if args.no_gt:
        print("\n[INFO] --no_gt set: skipping metric computation.")
        summary = {"note": "No ground truth provided; metrics not computed."}
    else:
        has_gt = [qid for qid in qry_ids if qid is not None]
        if not has_gt:
            print("[WARN] All item_ids are None — cannot compute metrics.")
            summary = {"note": "item_ids could not be resolved for any query."}
        else:
            print(f"\n[INFO] Computing metrics for K ∈ {args.k_values} …")
            summary = compute_metrics(I, qry_ids, gal_ids, args.k_values)

    # ---- Print results table -----------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Results — Config C  |  alpha={args.alpha}")
    print("=" * 60)
    for k, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        n = metrics["n_queries"]
        r = metrics.get(f"Recall@{k}", 0)
        d = metrics.get(f"NDCG@{k}", 0)
        m = metrics.get(f"mAP@{k}", 0)
        print(f"  K={k:2d}  |  Recall@{k}={r:.4f}  NDCG@{k}={d:.4f}  mAP@{k}={m:.4f}  (n={n})")
    print("=" * 60)

    elapsed = time.time() - t0
    print(f"\n[INFO] Total time: {elapsed:.1f}s")

    # ---- Save JSON results -------------------------------------------------
    output_path = Path(args.output_json) if args.output_json else \
                  OUTPUT_DIR / "results_folder_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "alpha"         : args.alpha,
        "k_values"      : args.k_values,
        "n_queries_total": len(image_paths),
        "n_queries_evaluated": len([q for q in qry_ids if q is not None]),
        "n_skipped"     : len(skipped),
        "elapsed_seconds": round(elapsed, 1),
        "metrics"       : summary,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
