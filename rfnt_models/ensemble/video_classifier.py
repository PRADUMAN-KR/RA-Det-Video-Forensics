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
        
        # Build classifier head on top of spatiotemporal embeddings
        layers = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, **kwargs) -> torch.Tensor:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VideoFoundationBranch requires 'video' input tensor")
            
        # Get video embeddings from preloaded wrapper
        # Encoder is evaluated under no_grad since it's frozen
        with torch.no_grad():
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
        lpd_features = kwargs.get("lpd_features")
        
        if noise is None or lpd_features is None:
            raise ValueError("VideoScratchBranch requires both 'noise' and 'lpd_features' input tensors")
            
        # Concatenate along channel dimension: [B, C_noise + C_lpd, T, H, W]
        x = torch.cat([noise, lpd_features], dim=1)
        logits = self.model(x)
        return logits.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        return ["noise", "lpd_features"]


class VideoEnsembleClassifier(FlexibleEnsembleClassifier):
    """
    Video Ensemble Classifier combining foundation, scratch, and difference branches in 3D.
    """

    def __init__(self,
                 video_encoder: nn.Module,
                 feature_dim: int = 1024,
                 lpd_channels: int = 3,
                 resnet_variant: str = "r3d_18",
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 fusion_method: str = "logit_weighted",
                 use_four_branch: bool = True):
        super().__init__(fusion_method=fusion_method, temperature=temperature)
        
        self.video_encoder = video_encoder
        self.feature_dim = feature_dim
        self.use_four_branch = use_four_branch

        # 1. Foundation Branch (VideoMAE + MLP)
        foundation_branch = VideoFoundationBranch(
            name="foundation",
            video_encoder=video_encoder,
            feature_dim=feature_dim,
            hidden_dims=[1024, 512],
            dropout=dropout
        )
        self.register_branch(foundation_branch)

        # 2. Scratch Branch (ResNet3D on noise + LPD)
        # Input channels = 3 (perturbation noise) + lpd_channels (typically 3) = 6
        scratch_branch = VideoScratchBranch(
            name="scratch",
            input_channels=3 + lpd_channels,
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

    def forward(self,
                video: torch.Tensor,
                noise: torch.Tensor,
                lpd_features: torch.Tensor,
                l2_distance: Optional[torch.Tensor] = None,
                embedding_diff: Optional[torch.Tensor] = None,
                use_max_for_eval: bool = False,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Overridden forward to run the generic FlexibleEnsembleClassifier routing.
        """
        inputs = {
            "video": video,
            "noise": noise,
            "lpd_features": lpd_features,
            "l2_distance": l2_distance,
            "embedding_diff": embedding_diff
        }
        
        # Route to base implementation
        return super().forward(
            return_all_logits=True,
            use_max_for_eval=use_max_for_eval,
            **inputs
        )
