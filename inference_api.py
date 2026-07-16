"""
RA-Det FastAPI Inference Pipeline — Temporal Video Deepfake Detection.

Loads the trained epoch-4 ensemble checkpoint
(VideoMAE encoder + UNetDecoder3D + VideoEnsembleClassifier)
and exposes a REST API for real/fake video classification.

Usage:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict      — Upload a video file, get real/fake prediction
    GET  /health       — Liveness check
    GET  /info         — Model metadata and config
"""

import os
import sys
import time
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so all local imports resolve correctly
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ra_det_api")


# ===========================================================================
# Configuration — mirrors the training config exactly
# ===========================================================================
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "ensemble_vitl16_raw_lpd_discrepancy",
    "checkpoint_epoch_4.pt",
)

MODEL_NAME     = "MCG-NJU/videomae-large"
NUM_FRAMES     = 16
FRAME_SIZE     = 224
ATTACK_EPS     = 16 / 255
FAKE_THRESHOLD = 0.5  # sigmoid(logit) > 0.5 → FAKE

# Decoder kwargs — must match what the trainer builds in video mode.
# CRITICAL: strategy_channels is forced to 0 in video mode by the trainer
# (embedding_trainer.py lines 1087-1091), even though the config says 15.
DECODER_KWARGS = dict(
    base_channels     = 64,
    num_levels        = 5,
    num_heads         = 8,
    use_attention     = True,
    bottleneck_size   = 14,
    strategy_channels = 0,   # MUST be 0 in video mode
)

# ---------------------------------------------------------------------------
# Embedding-discrepancy calibration (from epoch-4 validation on training set)
# These are the per-class means observed during validation.
# Used as a distribution-robust fallback when classifier is overconfident.
# ---------------------------------------------------------------------------
VAL_SIM_REAL = 0.9935   # mean cosine similarity for real videos
VAL_SIM_FAKE = 0.8816   # mean cosine similarity for fake videos
VAL_L2_REAL  = 0.8070   # mean L2 distance for real videos
VAL_L2_FAKE  = 6.1752   # mean L2 distance for fake videos

# Midpoints — simple linear thresholds derived from validation means
_SIM_THRESHOLD = (VAL_SIM_REAL + VAL_SIM_FAKE) / 2   # 0.9376
_L2_THRESHOLD  = (VAL_L2_REAL  + VAL_L2_FAKE)  / 2   # 3.4911

# VideoMAE pretraining normalisation stats (NOT ImageNet)
# Matches VideoDataset default transform (video_dataset.py line 71)
VIDEOMAE_MEAN = [0.5, 0.5, 0.5]
VIDEOMAE_STD  = [0.5, 0.5, 0.5]


# ===========================================================================
# Global model container (populated at startup)
# ===========================================================================
class ModelStore:
    encoder:         Optional[torch.nn.Module] = None
    decoder:         Optional[torch.nn.Module] = None
    classifier:      Optional[torch.nn.Module] = None
    device:          Optional[torch.device]    = None
    feature_dim:     int                       = 1024
    checkpoint_epoch: int                      = 0
    load_time_s:     float                     = 0.0


_store = ModelStore()


# ===========================================================================
# Model Loading
# ===========================================================================
def _load_models(device: torch.device) -> None:
    """
    Instantiate and load all three model components from the checkpoint.

    Architecture exactly matches embedding_trainer.py video-mode initialisation:
      encoder    → VideoMAEWrapper              (frozen)
      decoder    → UNetDecoder3D               (strategy_channels=0)
      classifier → VideoEnsembleClassifier     (2-branch: foundation + scratch)
    """
    t0 = time.time()
    logger.info("Loading VideoMAE encoder: %s", MODEL_NAME)

    # 1. Encoder — frozen VideoMAE Large
    from models.video_encoder_wrapper import VideoMAEWrapper
    encoder = VideoMAEWrapper(model_name=MODEL_NAME, device=str(device))
    encoder.eval()
    feature_dim = encoder.output_dim  # 1024 for Large
    logger.info("Encoder loaded. Feature dim: %d", feature_dim)

    # 2. Decoder — UNetDecoder3D
    from models.unet_3d import UNetDecoder3D
    decoder = UNetDecoder3D(
        embed_dim = feature_dim,
        eps       = ATTACK_EPS,
        **DECODER_KWARGS,
    ).to(device)
    logger.info("UNetDecoder3D instantiated.")

    # 3. Classifier — VideoEnsembleClassifier
    #    Matches setup_training_mode("ensemble") in video mode:
    #      lpd_channels=0, temporal_diff_channels=3, resnet_variant="r3d_18"
    #      use_four_branch=False  (use_four_branch_ensemble default is False)
    classifier = VideoEnsembleClassifier(
        video_encoder          = encoder,
        feature_dim            = feature_dim,
        lpd_channels           = 0,
        temporal_diff_channels = 3,
        resnet_variant         = "r3d_18",
        dropout                = 0.1,
        temperature            = 1.0,
        fusion_method          = "logit_weighted",
        use_four_branch        = True,   # checkpoint has l2_distance + embedding_diff branches
        use_npr_branch         = True,   # NPR branch
    ).to(device)
    logger.info("VideoEnsembleClassifier instantiated.")

    # 4. Load checkpoint weights
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Make sure the container checkpoint is accessible at this path."
        )

    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    decoder.load_state_dict(ckpt["decoder_state_dict"])
    logger.info("Decoder weights loaded (epoch %d).", ckpt.get("epoch", "?"))

    if "classifier_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'classifier_state_dict'. "
        "Is this the correct checkpoint file?"
        )
    classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
    logger.info("Classifier weights loaded.")

    decoder.eval()
    classifier.eval()

    # 5. Populate global store
    _store.encoder          = encoder
    _store.decoder          = decoder
    _store.classifier       = classifier
    _store.device           = device
    _store.feature_dim      = feature_dim
    _store.checkpoint_epoch = int(ckpt.get("epoch", 0))
    _store.load_time_s      = time.time() - t0

    logger.info(
        "All models ready. Load time: %.1fs | Device: %s",
        _store.load_time_s, device,
    )


# ===========================================================================
# Video Preprocessing
# ===========================================================================
_frame_transform = T.Compose([
    T.Resize((FRAME_SIZE, FRAME_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
])


def preprocess_video(video_path: str) -> torch.Tensor:
    """
    Decode a video file and return a normalised tensor ready for inference.

    Pipeline (matches VideoDataset._load_video_pyav exactly):
      1. Open with PyAV and collect up to 300 decoded frames (prevents OOM on long videos).
      2. Uniformly sample NUM_FRAMES indices.
      3. Convert each to RGB PIL Image, apply _frame_transform.
      4. Stack to [T, C, H, W], unsqueeze to [1, T, C, H, W].

    Returns:
        Tensor [1, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE]
    """
    container = av.open(video_path)
    raw_frames = []
    max_frames = 300  # limit to ~10 seconds at 30fps to avoid RAM exhaustion
    
    try:
        for frame in container.decode(video=0):
            raw_frames.append(frame)
            if len(raw_frames) >= max_frames:
                break
    except Exception:
        pass
    finally:
        container.close()

    total = len(raw_frames)
    if total == 0:
        raise ValueError(f"No frames could be decoded from: {video_path}")

    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = [_frame_transform(raw_frames[i].to_image().convert('RGB')) for i in indices]

    video_tensor = torch.stack(frames, dim=0)   # [T, C, H, W]
    return video_tensor.unsqueeze(0)            # [1, T, C, H, W]


# ===========================================================================
# Core Inference
# ===========================================================================
def run_inference(video_tensor: torch.Tensor) -> dict:
    """
    Execute the full RA-Det forward pass and return ALL raw signals.

    Does NOT make a final real/fake decision — that is the endpoint's job.
    Returns classifier probability, embedding score, and branch probabilities
    so that the /predict endpoint can apply the user-selected decision mode.

    Mirrors CrossGeneratorEvaluator.evaluate_decoder (video branch):
      1. Clean embeddings  : encoder(video)
      2. 3D noise          : decoder(video_c_first, clean_emb)
      3. Noisy embeddings  : encoder(perturbed_video)
      4. Temporal diff     : frame-to-frame delta, normalised, [B,C,T,H,W]
      5. Classifier        : VideoEnsembleClassifier(use_max_for_eval=False)
    """
    device     = _store.device
    encoder    = _store.encoder
    decoder    = _store.decoder
    classifier = _store.classifier

    video = video_tensor.to(device)               # [1, T, C, H, W]

    # Step 1 — Clean embeddings and NPR features
    with torch.no_grad():
        npr_features = extract_npr_features(video)        # [1, 48]
        clean_emb = encoder(video)                    # [1, D]

        # Step 2 — Spatiotemporal noise
        video_c_first = video.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        noise = decoder(
            original_video  = video_c_first,
            clean_embedding = clean_emb,
        )                                              # [1, 3, T, H, W]

        # Step 3 — Noisy embeddings
        perturbed = (video_c_first + noise).permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]
        noisy_emb = encoder(perturbed)                # [1, D]

    # Ensure remaining steps do not track gradients to save memory
    with torch.no_grad():
        # Step 4 — Temporal difference (matches training exactly)
        frame_diffs   = video[:, 1:] - video[:, :-1]                         # [B, T-1, C, H, W]
        frame_diffs   = torch.cat([frame_diffs, frame_diffs[:, -1:]], dim=1) # [B, T,   C, H, W]
        temporal_diff = frame_diffs.permute(0, 2, 1, 3, 4).contiguous()     # [B, C, T, H, W]
        td_std        = temporal_diff.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-4)
        temporal_diff = (temporal_diff / (3.0 * td_std)).clamp(-1.0, 1.0)
    
        # Step 5 — Derived branch inputs
        l2_dist    = torch.norm(clean_emb - noisy_emb, p=2, dim=1, keepdim=True)  # [1, 1]
        emb_diff   = clean_emb - noisy_emb                                          # [1, D]
        cosine_sim = float(F.cosine_similarity(clean_emb, noisy_emb, dim=1).item())
        l2_val     = float(l2_dist.item())
    
        # Step 6 — Classifier with trained logit_weighted fusion.
        # NOTE: use_max_for_eval=False is critical.
        # use_max_for_eval=True switches to MaxFusion which takes max probability
        # across ALL branches — one overconfident branch (foundation=1.0) will
        # poison the whole ensemble and classify everything as fake.
        outputs = classifier(
            video            = video,
            noise            = noise,
            temporal_diff    = temporal_diff,
            lpd_features     = None,      # 2D-LPD disabled in video mode
            l2_distance      = l2_dist,
            embedding_diff   = emb_diff,
            npr_features     = npr_features,
            use_max_for_eval = False,     # logit_weighted fusion (trained weights)
        )

    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
        ensemble_logit, branch_logits = outputs
    else:
        ensemble_logit = outputs
        branch_logits  = {}

    cls_prob = float(torch.sigmoid(ensemble_logit.squeeze()).item())

    # Embedding-discrepancy score calibrated from epoch-4 validation stats.
    # Normalises l2 and sim onto [0, 1] where 0.0 = real, 1.0 = fake.
    emb_score_l2  = (l2_val     - VAL_L2_REAL)  / max(VAL_L2_FAKE  - VAL_L2_REAL,  1e-6)
    emb_score_sim = (VAL_SIM_REAL - cosine_sim)  / max(VAL_SIM_REAL - VAL_SIM_FAKE, 1e-6)
    emb_score     = float(max(0.0, min(1.0, (emb_score_l2 + emb_score_sim) / 2.0)))

    branch_probs = {
        name: round(float(torch.sigmoid(logit.squeeze()).item()), 4)
        for name, logit in branch_logits.items()
    }

    return {
        "cls_prob":    cls_prob,
        "emb_score":   emb_score,
        "l2_val":      l2_val,
        "cosine_sim":  cosine_sim,
        "branch_probs": branch_probs,
    }


def _make_decision(raw: dict, mode: str, threshold: float = FAKE_THRESHOLD) -> dict:
    """
    Apply the selected decision strategy to raw inference signals.

    Modes
    -----
    classifier  (default)
        Use the trained VideoEnsembleClassifier with logit_weighted fusion.
        Most accurate on the training distribution.
        May be biased for out-of-distribution videos.

    embedding
        Use only the embedding-discrepancy score, calibrated against the
        epoch-4 validation statistics (real l2=0.81, fake l2=6.18).
        More distribution-robust but less sensitive to subtle fakes.
        Decision: emb_score > 0.5 -> fake.

    hybrid
        Average of classifier probability and embedding score.
        Balances the strengths of both signals.
    """
    cls_prob  = raw["cls_prob"]
    emb_score = raw["emb_score"]

    if mode == "embedding":
        prob          = emb_score
        signal_source = "embedding_discrepancy"
    elif mode == "hybrid":
        prob          = (cls_prob + emb_score) / 2.0
        signal_source = "hybrid"
    else:  # "classifier" (default)
        prob          = cls_prob
        signal_source = "classifier"

    prediction = "fake" if prob > threshold else "real"

    delta = abs(prob - 0.5)
    if delta > 0.4:
        confidence = "very_high"
    elif delta > 0.25:
        confidence = "high"
    elif delta > 0.1:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "prediction":    prediction,
        "probability":   round(prob, 4),
        "confidence":    confidence,
        "signal_source": signal_source,
        "mode":          mode,
        "details": {
            "cosine_similarity":      round(raw["cosine_sim"], 4),
            "l2_distance":            round(raw["l2_val"], 4),
            "embedding_score":        round(emb_score, 4),     # 0=real, 1=fake
            "classifier_probability": round(cls_prob, 4),
            "threshold":              threshold,
            "calibration": {
                "real_l2":       VAL_L2_REAL,   "fake_l2":  VAL_L2_FAKE,
                "real_sim":      VAL_SIM_REAL,  "fake_sim": VAL_SIM_FAKE,
                "l2_midpoint":   round(_L2_THRESHOLD, 4),
                "sim_midpoint":  round(_SIM_THRESHOLD, 4),
            },
            "branch_probabilities":   raw["branch_probs"],
        },
    }


# ===========================================================================
# FastAPI Application
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== RA-Det Inference API — startup ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    _load_models(device)
    yield
    logger.info("=== RA-Det Inference API — shutdown ===")


app = FastAPI(
    title       = "RA-Det Temporal Video Deepfake Detection API",
    description = (
        "Detects AI-generated / deepfake videos using the RA-Det framework "
        "with VideoMAE spatiotemporal embeddings and a 3D UNet adversarial decoder. "
        "Epoch-4 validation — AUC: 0.9959 | Accuracy: 98.10%"
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)


@app.get("/health", tags=["system"])
async def health():
    """Liveness probe — returns 200 if models are loaded."""
    if _store.encoder is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded.")
    return {
        "status":           "ok",
        "device":           str(_store.device),
        "checkpoint_epoch": _store.checkpoint_epoch,
    }


@app.get("/info", tags=["system"])
async def info():
    """Model configuration and validation performance summary."""
    return {
        "encoder":          MODEL_NAME,
        "decoder":          "UNetDecoder3D",
        "classifier":       "VideoEnsembleClassifier (foundation + scratch, 2-branch)",
        "num_frames":       NUM_FRAMES,
        "frame_size":       FRAME_SIZE,
        "attack_eps":       round(ATTACK_EPS, 5),
        "fake_threshold":   FAKE_THRESHOLD,
        "feature_dim":      _store.feature_dim,
        "checkpoint_epoch": _store.checkpoint_epoch,
        "checkpoint_path":  CHECKPOINT_PATH,
        "load_time_s":      round(_store.load_time_s, 2),
        "validation_metrics": {
            "auc":               0.9959,
            "average_precision": 0.9945,
            "optimal_accuracy":  0.9810,
            "real_l2":           0.8070,
            "fake_l2":           6.1752,
            "l2_gap":            5.3681,
        },
    }


ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
VALID_MODES        = {"classifier", "embedding", "hybrid"}


@app.post("/predict", tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    mode: str = Query("classifier", description="Decision strategy: classifier, embedding, or hybrid"),
    threshold: Optional[float] = Query(None, description="Custom threshold (0.0 to 1.0). If None, defaults to FAKE_THRESHOLD.")
):
    """
    Upload a video file and receive a real/fake prediction.

    **mode** (query parameter) — controls the decision strategy:
    - `classifier` *(default)* — trained VideoEnsembleClassifier with
      logit_weighted fusion. Best on training distribution.
    - `embedding` — embedding-discrepancy score calibrated from
      validation stats (real l2=0.81, fake l2=6.18). More
      distribution-robust; weaker on subtle fakes.
    - `hybrid` — average of classifier + embedding scores.

    Example:
        POST /predict?mode=embedding
        POST /predict?mode=hybrid

    Returns JSON with prediction, probability, confidence, signal_source,
    and a details block containing all raw signals for transparency.
    """
    if _store.encoder is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Choose from: {sorted(VALID_MODES)}"
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = None
    t_start  = time.time()
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("Received: %s (%.1f KB) mode=%s", file.filename, len(content) / 1024, mode)

        t_prep = time.time()
        try:
            video_tensor = preprocess_video(tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        logger.info("Preprocessing: %.2fs", time.time() - t_prep)

        t_inf   = time.time()
        raw     = run_inference(video_tensor)
        
        active_threshold = threshold if threshold is not None else FAKE_THRESHOLD
        result  = _make_decision(raw, mode, threshold=active_threshold)
        inf_time = time.time() - t_inf

        result["timing"]   = {"inference_s": round(inf_time, 3), "total_s": round(time.time() - t_start, 3)}
        result["filename"] = file.filename

        logger.info(
            "Result [%s]: %s (prob=%.4f, %.2fs)",
            mode, result["prediction"], result["probability"], inf_time,
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
