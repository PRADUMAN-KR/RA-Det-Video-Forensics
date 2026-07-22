"""
Video Ensemble Classifier Module.

Extends the generic FlexibleEnsembleClassifier for 3D spatiotemporal video/deepfake detection.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union, Optional
from .classifier import FlexibleEnsembleClassifier
from .base import BaseBranch
from .branches import L2DistanceBranch, EmbeddingDiffBranch


class VideoFoundationBranch(BaseBranch):
    """
    Foundation model branch that takes video tensors [B, T, C, H, W],
    extracts spatiotemporal VideoMAE embeddings [B, D], and runs a trainable classification MLP.
    """

    def __init__(self,
                 name: str = "foundation",
                 video_encoder: nn.Module = None,
                 feature_dim: int = 1024,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1):
        super().__init__(name)
        if hidden_dims is None:
            hidden_dims = [1024, 512]
            
        self.video_encoder = video_encoder
        self.feature_dim = feature_dim
        
        # Build classifier head on top of spatiotemporal embeddings.
        # Use LayerNorm instead of BatchNorm1d: video training often runs at B=1
        # or B=2 (VRAM-limited), where BatchNorm1d produces unstable/wrong statistics.
        layers = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),      # Works for any batch size including B=1
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, **kwargs) -> torch.Tensor:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VideoFoundationBranch requires 'video' input tensor")

        # CRITICAL: Do NOT use torch.no_grad() here.
        # The VideoMAE encoder parameters are frozen (requires_grad=False) so they
        # will never accumulate or apply gradients. However, gradients must still
        # FLOW THROUGH the frozen encoder's computation graph so that BCE loss can
        # reach and update the trainable MLP classifier head below.
        # Wrapping in no_grad() severs this path entirely, making the MLP untrainable.
        features = self.video_encoder(video)

        logits = self.classifier(features)
        return logits.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        return ["video"]


class VideoScratchBranch(BaseBranch):
    """
    Scratch CNN branch that takes noise [B, C, T, H, W] and LPD features [B, C, T, H, W],
    concatenates them, and runs a ResNet3DScratch model trained from scratch.
    """

    def __init__(self,
                 name: str = "scratch",
                 input_channels: int = 6,  # 3 (noise) + 3 (LPD)
                 architecture: str = "r3d_18",
                 num_classes: int = 1):
        super().__init__(name)
        from rfnt_models.scratch_models.resnet3d_scratch import ResNet3DScratch
        self.model = ResNet3DScratch(
            input_channels=input_channels,
            architecture=architecture,
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, **kwargs) -> torch.Tensor:
        noise = kwargs.get("noise")
        temporal_diff = kwargs.get("temporal_diff")  # [B, C, T, H, W] frame-to-frame delta
        lpd_features = kwargs.get("lpd_features")

        if noise is None:
            raise ValueError("VideoScratchBranch requires 'noise' input tensor")

        parts = [noise]
        if temporal_diff is not None:
            parts.append(temporal_diff)
        if lpd_features is not None:
            parts.append(lpd_features)

        # Concatenate all available signals along channel dim: [B, C_total, T, H, W]
        x = torch.cat(parts, dim=1)
        logits = self.model(x)
        return logits.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        return ["noise", "temporal_diff", "lpd_features"]


class VideoNPRBranch(BaseBranch):
    """
    NPR Branch that takes precomputed NPR features [B, 48] and runs a small MLP.
    """
    def __init__(self, name: str = "npr", input_dim: int = 48, hidden_dims: List[int] = None, dropout: float = 0.1):
        super().__init__(name)
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, **kwargs) -> torch.Tensor:
        npr_features = kwargs.get("npr_features")
        if npr_features is None:
            raise ValueError("VideoNPRBranch requires 'npr_features' input tensor")
        return self.mlp(npr_features)

    def get_input_keys(self) -> List[str]:
        return ["npr_features"]


class DirectFeatureBranch(BaseBranch):
    """
    Anchor branch: classifies directly from the frozen VideoMAE CLS embedding.

    This branch does NOT use the adversarial decoder output at all — it takes the
    clean VideoMAE embedding and runs it through a small MLP. Because it is completely
    independent of the 3D UNet decoder, it prevents ensemble collapse when the decoder
    learns a bad or overfit perturbation strategy on the training distribution.

    This is the most important branch for out-of-distribution generalisation:
    VideoMAE was pretrained on real camera footage (Kinetics-400) and already
    encodes temporal physics and motion patterns. A linear probe on top of its
    CLS token can find the real/fake boundary without needing adversarial noise.

    Input:  clean_embedding [B, D] — frozen VideoMAE CLS token (no gradient required)
    Output: logits [B, 1]
    """

    def __init__(
        self,
        name: str = "direct_feature",
        feature_dim: int = 1024,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__(name)
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, **kwargs) -> torch.Tensor:
        clean_embedding = kwargs.get("clean_embedding")
        if clean_embedding is None:
            raise ValueError("DirectFeatureBranch requires 'clean_embedding' input tensor")
        # Detach from the encoder graph — we only want to train the MLP head,
        # not accidentally allow gradients to flow back into VideoMAE.
        return self.mlp(clean_embedding.detach())

    def get_input_keys(self) -> List[str]:
        return ["clean_embedding"]


class VideoEnsembleClassifier(FlexibleEnsembleClassifier):
    """
    Video Ensemble Classifier combining foundation, scratch, and difference branches in 3D.

    Branches:
      1. Foundation:      frozen VideoMAE + trainable MLP classifier head.
      2. Scratch:         R3D-18 3D CNN trained from scratch on noise + temporal diffs.
      3. L2Distance:      MLP on L2 norm of (clean_emb - noisy_emb).
      4. EmbeddingDiff:   MLP on element-wise (clean_emb - noisy_emb) vector.
      5. NPR:             MLP on NPR pixel-relation features.
      6. DirectFeature:   MLP on raw clean VideoMAE embedding (decoder-independent anchor).
    """

    def __init__(self,
                 video_encoder: nn.Module,
                 feature_dim: int = 1024,
                 lpd_channels: int = 0,
                 temporal_diff_channels: int = 3,
                 resnet_variant: str = "r3d_18",
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 fusion_method: str = "logit_weighted",
                 use_four_branch: bool = True,
                 use_npr_branch: bool = False,
                 use_direct_feature_branch: bool = True):
        super().__init__(fusion_method=fusion_method, temperature=temperature)
        
        self.video_encoder = video_encoder
        self.feature_dim = feature_dim
        self.use_four_branch = use_four_branch
        self.temporal_diff_channels = temporal_diff_channels

        # 1. Foundation Branch (VideoMAE + MLP)
        foundation_branch = VideoFoundationBranch(
            name="foundation",
            video_encoder=video_encoder,
            feature_dim=feature_dim,
            hidden_dims=[1024, 512],
            dropout=dropout
        )
        self.register_branch(foundation_branch)

        # 2. Scratch Branch (ResNet3D on noise + temporal_diff + optional LPD)
        # Total channels = 3 (noise) + temporal_diff_channels (3) + lpd_channels (0 in video mode)
        scratch_input_channels = 3 + temporal_diff_channels + (lpd_channels if lpd_channels > 0 else 0)
        scratch_branch = VideoScratchBranch(
            name="scratch",
            input_channels=scratch_input_channels,
            architecture=resnet_variant,
            num_classes=1
        )
        self.register_branch(scratch_branch)

        if use_four_branch:
            # 3. L2 Distance Branch (MLP on embedding L2 distance)
            l2_branch = L2DistanceBranch(
                name="l2_distance",
                hidden_dims=[128, 64, 1],
                dropout=dropout
            )
            self.register_branch(l2_branch)

            # 4. Embedding Difference Branch (MLP on clean - noisy embedding vector)
            diff_branch = EmbeddingDiffBranch(
                name="embedding_diff",
                hidden_dims=[512, 256, 1],
                feature_dim=feature_dim,
                dropout=dropout
            )
            self.register_branch(diff_branch)

        if use_npr_branch:
            # 5. NPR Branch (MLP on NPR pixel relation features)
            npr_branch = VideoNPRBranch(
                name="npr",
                input_dim=48,
                hidden_dims=[128, 64],
                dropout=dropout
            )
            self.register_branch(npr_branch)

        if use_direct_feature_branch:
            # 6. DirectFeature Branch — decoder-independent anchor branch.
            # Classifies from the raw frozen VideoMAE CLS embedding.
            # Completely independent of the adversarial 3D UNet decoder:
            # if the decoder overfits to training-distribution artifacts,
            # this branch still provides a correct gradient signal.
            direct_branch = DirectFeatureBranch(
                name="direct_feature",
                feature_dim=feature_dim,
                hidden_dims=[512, 256],
                dropout=dropout
            )
            self.register_branch(direct_branch)

    def forward(self,
                video: torch.Tensor,
                noise: torch.Tensor,
                temporal_diff: torch.Tensor,
                lpd_features: Optional[torch.Tensor] = None,
                l2_distance: Optional[torch.Tensor] = None,
                embedding_diff: Optional[torch.Tensor] = None,
                npr_features: Optional[torch.Tensor] = None,
                clean_embedding: Optional[torch.Tensor] = None,
                use_max_for_eval: bool = False,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Overridden forward to run the generic FlexibleEnsembleClassifier routing.

        Args:
            video:           [B, T, C, H, W]  - original video frames (T-first for VideoMAE).
            noise:           [B, C, T, H, W]  - 3D UNet adversarial noise.
            temporal_diff:   [B, C, T, H, W]  - frame-to-frame difference (motion signal).
            lpd_features:    [B, C, T, H, W]  - optional, 2D-LPD features (None in video mode).
            l2_distance:     [B, 1]           - L2 norm of embedding delta.
            embedding_diff:  [B, D]           - element-wise embedding delta.
            npr_features:    [B, 48]          - NPR pixel-relation features.
            clean_embedding: [B, D]           - raw frozen VideoMAE CLS embedding (anchor branch).
        """
        inputs = {
            "video": video,
            "noise": noise,
            "temporal_diff": temporal_diff,
            "lpd_features": lpd_features,
            "l2_distance": l2_distance,
            "embedding_diff": embedding_diff,
            "npr_features": npr_features,
            "clean_embedding": clean_embedding,
        }

        # Route to base implementation
        return super().forward(
            return_all_logits=True,
            use_max_for_eval=use_max_for_eval,
            **inputs
        )
