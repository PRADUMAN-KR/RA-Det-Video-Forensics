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
from configs.config import get_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ra_det_api")


# ===========================================================================
# Configuration — pulled directly from configs/config.py so inference always
# tracks whatever "--config <name>" was actually used to train the checkpoint.
# This mirrors the exact training invocation:
#
#   uv run python train.py --config dinov2_temporal_v1 --video-mode \
#       --num-frames 8 --batch-size 2 --num-workers 4 --niter 20 \
#       --eps-randomization --eps-min 4.0 --eps-max 32.0 --eps-schedule random \
#       --normalize-loss --four-branch-ensemble --fusion-method learned_weight \
#       --checkpoint-dir ./checkpoints/dinov2_temporal_v1 \
#       --resume ./checkpoints/dinov2_temporal_v1/checkpoint_epoch_12.pt
#
# CLI flags override config.py defaults during training (see train.py::main()),
# so any value explicitly passed on the command line (num_frames, fusion_method,
# eps_min/max, normalize_loss, four_branch_ensemble, checkpoint_dir, ...) is
# reproduced here as an override too — never re-derived from config.py alone.
# ---------------------------------------------------------------------------
TRAIN_CONFIG_NAME = os.getenv("TRAIN_CONFIG_NAME", "dinov2_temporal_v1")
_cfg = get_config(TRAIN_CONFIG_NAME)
_decoder_cfg = _cfg.get("decoder_kwargs", {})

_checkpoint_dir = _cfg["checkpoint_dir"]
if not os.path.isabs(_checkpoint_dir):
    _checkpoint_dir = os.path.normpath(os.path.join(PROJECT_ROOT, _checkpoint_dir))
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    os.path.join(_checkpoint_dir, "checkpoint_best.pt"),
)

MODEL_NAME = os.getenv("MODEL_NAME", _cfg["model_name"])

# --num-frames 8 (explicit training flag) — also matches decoder_kwargs.num_frames
# in config.py, but the CLI flag is what train.py actually applied at run time.
NUM_FRAMES = int(os.getenv("NUM_FRAMES", str(_decoder_cfg.get("num_frames", 8))))

FRAME_SIZE = 224

# --eps-randomization/--eps-min 4.0/--eps-max 32.0 were passed explicitly.
# train.py divides these CLI values by 255 (see train.py::main()); at inference
# time (no randomization) the decoder should run at the midpoint of the trained
# eps range, i.e. base attack_eps from config.py (16/255) — NOT eps_min or eps_max
# alone, since randomization means the decoder was trained across that full range
# with attack_eps as its centre/default (config['attack_eps'] is unchanged by
# --eps-randomization; only eps_min/eps_max/eps_schedule are set alongside it).
ATTACK_EPS = float(os.getenv("ATTACK_EPS", str(_cfg["attack_eps"])))

FAKE_THRESHOLD = 0.5  # sigmoid(logit) > 0.5 → FAKE — fallback only; see optimal_threshold below

# Layer 2 (VideoAnomalyDetector) checkpoint — trained SEPARATELY via
# trainers/train_anomaly.py (real-only clips, reconstruction MSE objective).
# This must NOT be confused with CHECKPOINT_PATH above (the Layer 3 ensemble
# checkpoint): that file only contains decoder_state_dict/classifier_state_dict
# and has no recon_head weights, so loading it into VideoAnomalyDetector is a
# no-op that leaves recon_head randomly initialised.
ANOMALY_CHECKPOINT_PATH = os.getenv(
    "ANOMALY_CHECKPOINT_PATH",
    os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "anomaly_detector",
        "anomaly_detector_best.pt",
    ),
)

# Decoder kwargs — sourced from configs/config.py's dinov2_temporal_v1 entry so
# these can never silently drift from what train.py actually built.
# strategy_channels is force-zeroed exactly like embedding_trainer.py does for
# self.video_mode (see EmbeddingTrainer.__init__), and num_levels/is_video_mode
# are dropped since UNetDecoder3D doesn't accept them as constructor kwargs
# (embedding_trainer.py filters to `valid_unet_keys` before calling UNetDecoder3D).
_VALID_UNET_KEYS = {
    "base_channels", "num_levels", "num_heads", "use_attention", "bottleneck_size",
}
DECODER_KWARGS = {k: v for k, v in _decoder_cfg.items() if k in _VALID_UNET_KEYS}
DECODER_KWARGS["strategy_channels"] = 0  # MUST be 0 in video mode

# --fusion-method learned_weight was passed explicitly on the training command
# (train.py::main() overrides config['fusion_method'] unconditionally with
# whatever --fusion-method resolves to, default "logit_weighted" if unset).
FUSION_METHOD = os.getenv("FUSION_METHOD", _cfg.get("fusion_method", "logit_weighted"))

# --four-branch-ensemble was passed explicitly (bool flag → True).
USE_FOUR_BRANCH_ENSEMBLE = os.getenv(
    "USE_FOUR_BRANCH_ENSEMBLE", str(_cfg.get("use_four_branch_ensemble", False))
).lower() in ("1", "true", "yes")

USE_NPR_BRANCH = _cfg.get("use_npr_branch", True)
USE_TEMPORAL_COHERENCE_BRANCH = _decoder_cfg.get("use_temporal_coherence_branch", "dinov2" in MODEL_NAME.lower())
UNFREEZE_LAST_N_BLOCKS = int(_decoder_cfg.get("unfreeze_last_n_blocks", 0))
TEMPORAL_TRANSFORMER_LAYERS = int(_decoder_cfg.get("temporal_transformer_layers", 2))

# ---------------------------------------------------------------------------
# Embedding-discrepancy calibration (from epoch-20 validation)
# These are the per-class means observed during validation.
# Used as a distribution-robust fallback when classifier is overconfident.
# ---------------------------------------------------------------------------
VAL_SIM_REAL = 0.9991   # mean cosine similarity for real videos
VAL_SIM_FAKE = 0.1601   # mean cosine similarity for fake videos
VAL_L2_REAL  = 0.0521   # mean L2 distance for real videos
VAL_L2_FAKE  = 10.7084  # mean L2 distance for fake videos

# Midpoints — simple linear thresholds derived from validation means
_SIM_THRESHOLD = (VAL_SIM_REAL + VAL_SIM_FAKE) / 2   # 0.5796
_L2_THRESHOLD  = (VAL_L2_REAL  + VAL_L2_FAKE)  / 2   # 5.3803

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
    # Optimal decision threshold loaded from the checkpoint.
    # Calibrated from the validation ROC curve during training (replaces hardcoded 0.5).
    optimal_threshold: float                      = FAKE_THRESHOLD


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
        # NOTE: unfreeze_last_n must match training (unfreeze_last_n_blocks=4 in
        # dinov2_temporal_v1), since the last N DINOv2 blocks were fine-tuned
        # during training and the fine-tuned weights differ from the pretrained
        # ones. We set requires_grad=False again right after loading (below) —
        # this only controls *which submodules exist*/get restored, not whether
        # they're trainable at inference time.
        encoder = DINOv2VideoEncoder(
            model_name=MODEL_NAME,
            num_frames=NUM_FRAMES,
            temporal_layers=TEMPORAL_TRANSFORMER_LAYERS,
            unfreeze_last_n=UNFREEZE_LAST_N_BLOCKS,
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
    # All flags below are sourced from configs/config.py's dinov2_temporal_v1
    # entry (with CLI-flag overrides applied above), matching exactly how
    # embedding_trainer.py's setup_training_mode(mode="ensemble") constructs it.
    classifier = VideoEnsembleClassifier(
        video_encoder                  = encoder,
        feature_dim                    = feature_dim,
        lpd_channels                   = 0,
        temporal_diff_channels         = 3,
        resnet_variant                 = "r3d_18",
        dropout                        = 0.1,
        temperature                    = 1.0,
        fusion_method                  = FUSION_METHOD,               # --fusion-method learned_weight
        use_four_branch                = USE_FOUR_BRANCH_ENSEMBLE,    # --four-branch-ensemble
        use_npr_branch                 = USE_NPR_BRANCH,
        use_direct_feature_branch      = True,
        use_temporal_coherence_branch  = USE_TEMPORAL_COHERENCE_BRANCH,
        num_frames                     = NUM_FRAMES,                  # --num-frames 8
    ).to(device)
    logger.info(
        "VideoEnsembleClassifier instantiated (fusion=%s, four_branch=%s, npr=%s, temporal_coherence=%s).",
        FUSION_METHOD, USE_FOUR_BRANCH_ENSEMBLE, USE_NPR_BRANCH, USE_TEMPORAL_COHERENCE_BRANCH,
    )

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
        if "optimal_threshold" in ckpt:
            _store.optimal_threshold = float(ckpt["optimal_threshold"])
            logger.info("  ✓ Loaded calibrated optimal_threshold: %.4f", _store.optimal_threshold)
        else:
            logger.warning(
                "  ⚠ No 'optimal_threshold' in checkpoint — using default FAKE_THRESHOLD=%.2f. "
                "FPR may be elevated. Retrain or patch the checkpoint to fix.",
                FAKE_THRESHOLD,
            )
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
    #
    # IMPORTANT: this is a SEPARATE model trained by trainers/train_anomaly.py
    # (real-only clips, reconstruction MSE objective) and must load from
    # ANOMALY_CHECKPOINT_PATH, NOT the Layer 3 ensemble checkpoint (CHECKPOINT_PATH).
    # The ensemble checkpoint has no "recon_head_state_dict" key, so loading it
    # here previously left recon_head randomly initialised (silent no-op via
    # strict=False) — Layer 2 was never actually running trained weights.
    try:
        anomaly_detector = VideoAnomalyDetector(
            encoder=encoder,
            feature_dim=feature_dim,
            checkpoint_path=ANOMALY_CHECKPOINT_PATH,
        )
        anomaly_detector.eval()
        _store.anomaly_detector = anomaly_detector
        if os.path.exists(ANOMALY_CHECKPOINT_PATH):
            logger.info("  ✓ Loaded VideoAnomalyDetector from %s", ANOMALY_CHECKPOINT_PATH)
        else:
            logger.warning(
                "  ⚠ ANOMALY_CHECKPOINT_PATH not found at %s — Layer 2 will run with "
                "UNTRAINED weights. Train it with trainers/train_anomaly.py or set "
                "ANOMALY_CHECKPOINT_PATH to disable/point at a real checkpoint.",
                ANOMALY_CHECKPOINT_PATH,
            )
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

        video_c_first = video.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

        # Step 2 — Two-pass decoder refinement (MUST match train_epoch() exactly —
        # see trainers/embedding_trainer.py::EmbeddingTrainer.train_epoch, video-mode
        # branch). The decoder's loss/gradients during training are always computed
        # from a SECOND pass that contrasts clean vs. perturbed embeddings in the
        # cross-attention bottleneck; the first pass is only a self-attention warm-up
        # used to generate that reference perturbation. Serving only the first pass
        # (as this code previously did) evaluates a decoder configuration that was
        # never the actual training target, producing a different — and weaker —
        # perturbation than what the classifier was trained against.
        #
        # 2a. First pass — bottleneck self-attends on clean embedding only.
        noise = decoder(
            original_video  = video_c_first,
            clean_embedding = clean_emb,
        )                                              # [1, 3, T, H, W]

        # 2b. Reference noisy embedding from the first-pass noise.
        perturbed_ref = (video_c_first + noise).permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]
        noisy_emb_ref = encoder(perturbed_ref)                          # [1, D]

        # 2c. Second pass — bottleneck now contrasts clean vs. perturbed embeddings.
        noise = decoder(
            original_video  = video_c_first,
            clean_embedding = clean_emb,
            noisy_embedding = noisy_emb_ref,
        )                                              # [1, 3, T, H, W]

        # Step 3 — Final noisy embeddings, from the refined (second-pass) noise.
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
        Use the trained VideoEnsembleClassifier (6-branch ensemble, learned_weight fusion).
        Calibrated threshold from validation ROC curve (Youden's J statistic).
        Expected: FPR ~5.4%, FNR ~0.3% at threshold=0.99 on training generators.

    embedding
        Use only the embedding-discrepancy score (L2 distance in VideoMAE space).
        More distribution-robust but less sensitive to subtle fakes.

    hybrid
        Average of classifier probability and embedding score.
        Balances accuracy and robustness.

    Confidence is measured as distance from the decision threshold, NOT from 0.5.
    This gives meaningful confidence values regardless of what threshold is used.
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

    prediction = "fake" if prob >= threshold else "real"

    # Confidence = distance from the decision threshold (NOT from 0.5).
    # e.g. with threshold=0.99:
    #   prob=0.999 → fake,  margin=0.009 → very_high
    #   prob=0.995 → fake,  margin=0.005 → high
    #   prob=0.991 → fake,  margin=0.001 → low  (borderline)
    #   prob=0.985 → real,  margin=0.005 → high  (comfortably real)
    #   prob=0.50  → real,  margin=0.490 → very_high (clearly real)
    margin = abs(prob - threshold)
    if margin > 0.05:
        confidence = "very_high"
    elif margin > 0.02:
        confidence = "high"
    elif margin > 0.005:
        confidence = "medium"
    else:
        confidence = "low"   # within 0.5 percentage points of threshold — borderline

    return {
        "prediction":         prediction,
        "probability":        round(prob, 6),
        "confidence":         confidence,
        "margin_to_threshold": round(margin, 6),   # how far from the decision boundary
        "signal_source":      signal_source,
        "mode":               mode,
        "details": {
            "cosine_similarity":      round(raw["cosine_sim"], 4),
            "l2_distance":            round(raw["l2_val"], 4),
            "embedding_score":        round(emb_score, 4),
            "classifier_probability": round(cls_prob, 4),
            "threshold":              threshold,
            "calibration": {
                "real_l2":            VAL_L2_REAL,
                "fake_l2":            VAL_L2_FAKE,
                "real_sim":           VAL_SIM_REAL,
                "fake_sim":           VAL_SIM_FAKE,
                "l2_midpoint":        round(_L2_THRESHOLD, 4),
                "sim_midpoint":       round(_SIM_THRESHOLD, 4),
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
    thr = _store.optimal_threshold
    return {
        "train_config_name": TRAIN_CONFIG_NAME,
        "encoder":          MODEL_NAME,
        "decoder":          "UNetDecoder3D (3D spatiotemporal, 5-level U-Net)",
        "classifier":       "VideoEnsembleClassifier (6-branch: foundation, scratch, l2_distance, embedding_diff, npr, direct_feature)",
        "fusion_method":    FUSION_METHOD,
        "use_four_branch_ensemble": USE_FOUR_BRANCH_ENSEMBLE,
        "use_npr_branch":   USE_NPR_BRANCH,
        "use_temporal_coherence_branch": USE_TEMPORAL_COHERENCE_BRANCH,
        "unfreeze_last_n_blocks": UNFREEZE_LAST_N_BLOCKS,
        "num_frames":       NUM_FRAMES,
        "frame_size":       FRAME_SIZE,
        "attack_eps":       round(ATTACK_EPS, 5),
        "fake_threshold":   thr,
        "threshold_source": "checkpoint_roc_youden_j" if thr != FAKE_THRESHOLD else "default_fallback",
        "feature_dim":      _store.feature_dim,
        "checkpoint_epoch": _store.checkpoint_epoch,
        "checkpoint_path":  CHECKPOINT_PATH,
        "load_time_s":      round(_store.load_time_s, 2),
        "validation_metrics": {
            "epoch":              20,
            "auc":               0.9993,
            "average_precision": 0.9994,
            "f1_at_threshold":   0.9722,
            "accuracy_at_threshold": 0.9715,
            "fpr_at_threshold":  0.054,
            "fnr_at_threshold":  0.003,
            "optimal_threshold": thr,
            "real_l2":           0.0521,
            "fake_l2":           10.7084,
            "l2_gap":            10.6563,
            "real_sim":          0.9991,
            "fake_sim":          0.1601,
        },
        "training_generators": ["cogvideox", "easyanimate", "hunyuanvideo", "ltxvideo"],
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
        # Use the calibrated threshold from the checkpoint (fixes high FPR from hard-coded 0.5).
        # User can still override per-request via the ?threshold= query param.
        active_threshold = threshold if threshold is not None else _store.optimal_threshold
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
