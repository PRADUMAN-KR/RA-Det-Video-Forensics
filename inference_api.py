"""
RA-Det FastAPI Inference Pipeline — Temporal Video Deepfake Detection.

Loads the trained epoch-5 NPR ensemble checkpoint
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

from rfnt_models.ensemble.video_classifier import VideoEnsembleClassifier
from rfnt_models.ensemble.npr import extract_npr_features
from detectors.provenance import check_provenance
from detectors.anomaly_detector import VideoAnomalyDetector

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
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "npr_ensemble",
        "npr_ensemble_epoch_5.pt",
    ),
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
    encoder:          Optional[torch.nn.Module]   = None
    decoder:          Optional[torch.nn.Module]   = None
    classifier:       Optional[torch.nn.Module]   = None
    anomaly_detector: Optional[VideoAnomalyDetector] = None
    device:           Optional[torch.device]      = None
    feature_dim:      int                         = 1024
    checkpoint_epoch: int                         = 0
    load_time_s:      float                       = 0.0


_store = ModelStore()


# ===========================================================================
# Model Loading
# ===========================================================================
def _load_models(device: torch.device) -> None:
    """
    Instantiate and load all three model components from the checkpoint.

    Architecture matches embedding_trainer.py video-mode initialisation:
      encoder    → DINOv2VideoEncoder or VideoMAEWrapper (frozen)
      decoder    → UNetDecoder3D                        (strategy_channels=0)
      classifier → VideoEnsembleClassifier              (multi-branch ensemble)
    """
    t0 = time.time()
    if "dinov2" in MODEL_NAME.lower():
        logger.info("Loading DINOv2 video encoder: %s", MODEL_NAME)
        from models.dinov2_video_encoder import DINOv2VideoEncoder
        encoder = DINOv2VideoEncoder(
            model_name=MODEL_NAME,
            num_frames=NUM_FRAMES,
            temporal_layers=DECODER_KWARGS.get("temporal_transformer_layers", 2),
            unfreeze_last_n=0,
            device=str(device),
        )
    else:
        logger.info("Loading VideoMAE encoder: %s", MODEL_NAME)
        from models.video_encoder_wrapper import VideoMAEWrapper
        encoder = VideoMAEWrapper(model_name=MODEL_NAME, device=str(device))

    encoder.eval()
    feature_dim = encoder.output_dim
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
    classifier = VideoEnsembleClassifier(
        video_encoder                  = encoder,
        feature_dim                    = feature_dim,
        lpd_channels                   = 0,
        temporal_diff_channels         = 3,
        resnet_variant                 = "r3d_18",
        dropout                        = 0.1,
        temperature                    = 1.0,
        fusion_method                  = "logit_weighted",
        use_four_branch                = True,   # checkpoint has l2_distance + embedding_diff branches
        use_npr_branch                 = True,   # NPR branch
        use_direct_feature_branch     = True,
        use_temporal_coherence_branch = "dinov2" in MODEL_NAME.lower() or DECODER_KWARGS.get("use_temporal_coherence_branch", False),
        num_frames                     = NUM_FRAMES,
    ).to(device)
    logger.info("VideoEnsembleClassifier instantiated.")

    # Load weights from checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        logger.info("Loading checkpoint weights from: %s", CHECKPOINT_PATH)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

        if "decoder_state_dict" in ckpt:
            decoder.load_state_dict(ckpt["decoder_state_dict"])
            logger.info("  ✓ Loaded decoder_state_dict")
        if "classifier_state_dict" in ckpt:
            classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
            logger.info("  ✓ Loaded classifier_state_dict")
        if "epoch" in ckpt:
            _store.checkpoint_epoch = ckpt["epoch"]
            logger.info("  ✓ Checkpoint epoch: %d", _store.checkpoint_epoch)
    else:
        logger.warning(
            "Checkpoint NOT found at %s. Running with uninitialised weights!",
            CHECKPOINT_PATH,
        )

    # Freeze all models for inference
    encoder.eval()
    decoder.eval()
    classifier.eval()
    for p in encoder.parameters():    p.requires_grad = False
    for p in decoder.parameters():    p.requires_grad = False
    for p in classifier.parameters(): p.requires_grad = False

    # Store loaded models
    _store.encoder     = encoder
    _store.decoder     = decoder
    _store.classifier  = classifier
    _store.device      = device
    _store.feature_dim = feature_dim
    _store.load_time_s = time.time() - t0

    # Optional Layer 2 Anomaly Detector
    try:
        anomaly_detector = VideoAnomalyDetector(encoder=encoder, feature_dim=feature_dim)
        if os.path.exists(CHECKPOINT_PATH):
            anomaly_detector.load_state_dict(ckpt, strict=False)
        anomaly_detector.eval()
        _store.anomaly_detector = anomaly_detector
        logger.info("  ✓ Loaded VideoAnomalyDetector")
    except Exception as exc:
        logger.warning("Could not initialise VideoAnomalyDetector: %s", exc)

    logger.info("Model loading complete in %.2fs", _store.load_time_s)


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
    """
    device     = _store.device
    encoder    = _store.encoder
    decoder    = _store.decoder
    classifier = _store.classifier

    video = video_tensor.to(device)               # [1, T, C, H, W]

    # Step 1 — Clean embeddings and NPR features
    with torch.no_grad():
        npr_features = extract_npr_features(video)        # [1, 48]
        if hasattr(encoder, "encode_video_detailed"):
            clean_emb, per_frame_cls, _ = encoder.encode_video_detailed(video)
        else:
            clean_emb = encoder(video)                    # [1, D]
            per_frame_cls = None

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
    
        outputs = classifier(
            video              = video,
            noise              = noise,
            temporal_diff      = temporal_diff,
            lpd_features       = None,      # 2D-LPD disabled in video mode
            l2_distance        = l2_dist,
            embedding_diff     = emb_diff,
            npr_features       = npr_features,
            clean_embedding    = clean_emb,  # DirectFeatureBranch anchor
            per_frame_features = per_frame_cls,
            use_max_for_eval   = False,
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
        "cascade_layers": {
            "layer_1": "C2PA Digital Provenance Check (~1ms)",
            "layer_2": "VideoMAE Anomaly Reconstruction Detector (~200ms)",
            "layer_3": "RA-Det 6-Branch Spatiotemporal Ensemble (~1.5s)"
        }
    }


ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
VALID_MODES        = {"classifier", "embedding", "hybrid"}


@app.post("/predict", tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    mode: str = Query("classifier", description="Decision strategy for Layer 3: classifier, embedding, or hybrid"),
    threshold: Optional[float] = Query(None, description="Custom threshold (0.0 to 1.0). If None, defaults to FAKE_THRESHOLD."),
    disable_cascade: bool = Query(False, description="Set True to force full Layer 3 RA-Det execution bypassing Layer 1 and Layer 2 short-circuiting.")
):
    """
    Upload a video file and receive a real/fake prediction through the 3-Layer Cascade.

    **3-Layer Cascade Pipeline**:
    - **Layer 1 (C2PA Provenance)**: Instant (~1ms) digital manifest check for C2PA AI/Camera signatures.
    - **Layer 2 (VideoMAE Anomaly)**: Fast (~200ms) reconstruction MSE check against real camera physics.
    - **Layer 3 (RA-Det Ensemble)**: Deep (~1.5s) 6-branch spatiotemporal ensemble analysis.
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

        # -------------------------------------------------------------------
        # LAYER 1: C2PA Digital Provenance Check (~1ms)
        # -------------------------------------------------------------------
        t_l1 = time.time()
        prov = check_provenance(tmp_path)
        l1_time = time.time() - t_l1

        if not disable_cascade and prov["verdict"] != "unknown":
            is_fake = (prov["verdict"] == "ai_generated")
            total_s = time.time() - t_start
            logger.info("Layer 1 hit: %s (generator=%s, %.3fs)", prov["verdict"], prov.get("generator"), l1_time)
            return JSONResponse(content={
                "prediction": "fake" if is_fake else "real",
                "probability": 1.0 if is_fake else 0.0,
                "confidence": "very_high",
                "decided_by": "layer_1_provenance_c2pa",
                "filename": file.filename,
                "timing": {"l1_s": round(l1_time, 4), "total_s": round(total_s, 4)},
                "cascade": {
                    "layer_1_provenance": prov,
                    "layer_2_anomaly": None,
                    "layer_3_radet": None
                }
            })

        # -------------------------------------------------------------------
        # Preprocess Video
        # -------------------------------------------------------------------
        t_prep = time.time()
        try:
            video_tensor = preprocess_video(tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        prep_time = time.time() - t_prep
        logger.info("Preprocessing: %.2fs", prep_time)

        # -------------------------------------------------------------------
        # LAYER 2: VideoMAE Anomaly Reconstruction Error Check (~200ms)
        # -------------------------------------------------------------------
        t_l2 = time.time()
        anomaly = _store.anomaly_detector.score(video_tensor) if _store.anomaly_detector else {"verdict": "uncertain"}
        l2_time = time.time() - t_l2

        if not disable_cascade and anomaly["verdict"] != "uncertain":
            is_fake = (anomaly["verdict"] == "fake")
            total_s = time.time() - t_start
            logger.info("Layer 2 hit: %s (mse=%.5f, %.3fs)", anomaly["verdict"], anomaly["mse_error"], l2_time)
            return JSONResponse(content={
                "prediction": anomaly["verdict"],
                "probability": anomaly["anomaly_score"],
                "confidence": anomaly["confidence"],
                "decided_by": "layer_2_video_anomaly",
                "filename": file.filename,
                "timing": {"l1_s": round(l1_time, 4), "l2_s": round(l2_time, 4), "total_s": round(total_s, 4)},
                "cascade": {
                    "layer_1_provenance": prov,
                    "layer_2_anomaly": anomaly,
                    "layer_3_radet": None
                }
            })

        # -------------------------------------------------------------------
        # LAYER 3: RA-Det 6-Branch Spatiotemporal Ensemble (~1.5s)
        # -------------------------------------------------------------------
        t_l3 = time.time()
        raw = run_inference(video_tensor)
        active_threshold = threshold if threshold is not None else FAKE_THRESHOLD
        result = _make_decision(raw, mode, threshold=active_threshold)
        l3_time = time.time() - t_l3

        total_s = time.time() - t_start
        result["decided_by"] = "layer_3_radet_ensemble"
        result["filename"]   = file.filename
        result["timing"]     = {
            "l1_s": round(l1_time, 4),
            "l2_s": round(l2_time, 4),
            "l3_s": round(l3_time, 4),
            "total_s": round(total_s, 4)
        }
        result["cascade"] = {
            "layer_1_provenance": prov,
            "layer_2_anomaly": anomaly,
            "layer_3_radet": raw
        }

        logger.info(
            "Layer 3 result [%s]: %s (prob=%.4f, l3=%.2fs, total=%.2fs)",
            mode, result["prediction"], result["probability"], l3_time, total_s,
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
