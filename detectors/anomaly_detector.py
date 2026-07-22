"""
Layer 2: VideoMAE Anomaly Reconstruction Detector.

Measures reconstruction error against a distribution of real camera videos.
Trained ONLY on real video clips (Kinetics-400).
Real videos reconstruct with low MSE; AI-generated videos exhibit high MSE.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class ReconstructionHead3D(nn.Module):
    """
    Lightweight 3D Deconvolutional head that reconstructs spatiotemporal video frames
    from VideoMAE 1024-dim CLS/pooled embeddings.
    """

    def __init__(self, in_dim: int = 1024, num_frames: int = 16, frame_size: int = 224):
        super().__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size

        # Map 1024-dim embedding to 3D spatial feature map [B, 512, 2, 7, 7]
        self.fc = nn.Linear(in_dim, 512 * 2 * 7 * 7)

        # 3D Deconv blocks: [B, 512, 2, 7, 7] -> [B, 3, 16, 224, 224]
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), # -> [4, 14, 14]
            nn.BatchNorm3d(256),
            nn.SiLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), # -> [8, 28, 28]
            nn.BatchNorm3d(128),
            nn.SiLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # -> [16, 56, 56]
            nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),   # -> [16, 112, 112]
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 3, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),    # -> [16, 224, 224]
            nn.Tanh()
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        B = embedding.shape[0]
        x = self.fc(embedding).view(B, 512, 2, 7, 7)
        reconstructed = self.deconv(x)  # [B, 3, T, H, W]
        # Permute to [B, T, 3, H, W] to match video tensor
        return reconstructed.permute(0, 2, 1, 3, 4)


class VideoAnomalyDetector(nn.Module):
    """
    Video Anomaly Detector using VideoMAE + Reconstruction Head.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int = 1024, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.recon_head = ReconstructionHead3D(in_dim=feature_dim)
        
        # Thresholds derived from real video validation MSE distribution
        # Real videos: MSE ~ 0.02 - 0.05
        # AI videos: MSE ~ 0.12 - 0.35
        self.low_threshold = 0.06   # Below this -> High confidence REAL
        self.high_threshold = 0.14  # Above this -> High confidence FAKE

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "recon_head_state_dict" in ckpt:
                self.recon_head.load_state_dict(ckpt["recon_head_state_dict"])
            elif "state_dict" in ckpt:
                self.recon_head.load_state_dict(ckpt["state_dict"])
            print(f"Loaded Anomaly Detector weights from {checkpoint_path}")

    @torch.no_grad()
    def score(self, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compute reconstruction error (MSE) for input video tensor [1, T, C, H, W].
        """
        device = next(self.recon_head.parameters()).device
        video = video_tensor.to(device)

        # Extract frozen VideoMAE embedding
        clean_emb = self.encoder(video)  # [1, D]

        # Reconstruct video frames
        reconstructed_video = self.recon_head(clean_emb)  # [1, T, C, H, W]

        # Compute MSE loss per sample
        mse = F.mse_loss(video, reconstructed_video, reduction="mean").item()

        # Map MSE to probability score [0.0, 1.0]
        # Normalized score where 0.0 = completely real, 1.0 = completely fake
        norm_score = float(min(1.0, max(0.0, (mse - 0.04) / (0.16 - 0.04))))

        if mse > self.high_threshold:
            verdict = "fake"
            confidence = "high"
        elif mse < self.low_threshold:
            verdict = "real"
            confidence = "high"
        else:
            verdict = "uncertain"
            confidence = "low"

        return {
            "mse_error": round(mse, 5),
            "anomaly_score": round(norm_score, 4),
            "verdict": verdict,
            "confidence": confidence,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold
        }
