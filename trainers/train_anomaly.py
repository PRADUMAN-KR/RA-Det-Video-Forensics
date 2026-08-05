"""
Layer 2 Anomaly Detector Training Script.

Trains ReconstructionHead3D ONLY on real video clips (e.g. Kinetics-400).
Goal: Minimize MSE reconstruction error for natural camera physics.

IMPORTANT — encoder/frame-count parity with inference_api.py:
    The frozen encoder used here MUST match the encoder inference_api.py loads
    (MODEL_NAME / TRAIN_CONFIG_NAME there, default "facebook/dinov2-large" via
    the dinov2_temporal_v1 config). ReconstructionHead3D is trained against ONE
    specific embedding space; feeding it embeddings from a different encoder at
    inference time (e.g. training with VideoMAE but serving with DINOv2) makes
    the learned reconstruction meaningless. By default this script now mirrors
    inference_api.py's defaults exactly (facebook/dinov2-large, 8 frames).

Usage:
    python trainers/train_anomaly.py \
        --real-data-path /path/to/kinetics400_real \
        --model-name facebook/dinov2-large \
        --num-frames 8 \
        --epochs 5 \
        --batch-size 4 \
        --checkpoint-dir ./checkpoints/anomaly_detector
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detectors.anomaly_detector import ReconstructionHead3D
from datasets.video_dataset import VideoDataset


def build_encoder(model_name: str, num_frames: int, device: str):
    """
    Encoder selection mirrors inference_api.py::_load_models exactly, so the
    embedding space seen during Layer-2 training always matches the embedding
    space it will see in production.
    """
    if "dinov2" in model_name.lower():
        from models.dinov2_video_encoder import DINOv2VideoEncoder
        encoder = DINOv2VideoEncoder(
            model_name=model_name,
            num_frames=num_frames,
            temporal_layers=2,
            unfreeze_last_n=0,
            device=device,
        )
    else:
        from models.video_encoder_wrapper import VideoMAEWrapper
        encoder = VideoMAEWrapper(model_name=model_name, device=device)
    return encoder


def train_anomaly_head(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Layer 2 Anomaly Detector on {device} ===")
    print(f"    Encoder:    {args.model_name}")
    print(f"    Num frames: {args.num_frames}")

    # 1. Load real-only video dataset
    dataset = VideoDataset(
        root_dir=args.real_data_path,
        num_frames=args.num_frames,
        split="train"
    )
    # Filter to ensure only real samples (label == 0) are used
    real_samples = [s for s in dataset.samples if s['label'] == 0]
    if len(real_samples) > 0:
        dataset.samples = real_samples

    print(f"Total real training clips: {len(dataset)}")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. Instantiate frozen encoder (same selection logic as inference_api.py)
    #    and the trainable ReconstructionHead3D.
    encoder = build_encoder(args.model_name, args.num_frames, str(device))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    recon_head = ReconstructionHead3D(
        in_dim=encoder.output_dim, num_frames=args.num_frames
    ).to(device)
    optimizer = torch.optim.AdamW(recon_head.parameters(), lr=1e-3, weight_decay=1e-4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    # 3. Training loop
    for epoch in range(1, args.epochs + 1):
        recon_head.train()
        total_mse = 0.0

        for batch_idx, batch in enumerate(train_loader):
            video = batch["video"].to(device)  # [B, T, C, H, W]

            with torch.no_grad():
                clean_emb = encoder(video)

            recon_video = recon_head(clean_emb, num_frames=video.shape[1])
            loss = F.mse_loss(recon_video, video)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] MSE Loss: {loss.item():.5f}")

        avg_loss = total_mse / max(len(train_loader), 1)
        print(f"--> Epoch {epoch} Complete. Average MSE Loss: {avg_loss:.5f}")

        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"anomaly_detector_epoch_{epoch}.pt")
        torch.save({"recon_head_state_dict": recon_head.state_dict(), "epoch": epoch, "loss": avg_loss}, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "anomaly_detector_best.pt")
            torch.save({"recon_head_state_dict": recon_head.state_dict(), "epoch": epoch, "loss": avg_loss}, best_path)
            print(f"  Best model saved to {best_path} (MSE: {best_loss:.5f})")

    print("=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 2 Anomaly Detector")
    parser.add_argument("--real-data-path", type=str, required=True, help="Path to directory containing real videos")
    parser.add_argument(
        "--model-name", type=str, default="facebook/dinov2-large",
        help="Frozen encoder to extract embeddings from. MUST match MODEL_NAME/"
             "TRAIN_CONFIG_NAME used by inference_api.py (default: facebook/dinov2-large)."
    )
    parser.add_argument(
        "--num-frames", type=int, default=8,
        help="Frames per clip. MUST match NUM_FRAMES used by inference_api.py (default: 8)."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/anomaly_detector", help="Directory to save checkpoints")

    args = parser.parse_args()
    train_anomaly_head(args)
