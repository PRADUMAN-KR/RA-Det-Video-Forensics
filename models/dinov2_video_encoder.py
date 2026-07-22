"""
DINOv2 Per-Frame Video Encoder with Temporal Transformer Head.

Processes each video frame through frozen DINOv2-Large to extract
forensic-grade spatial features, then models temporal coherence
via a lightweight transformer.

Key advantages over VideoMAE for deepfake detection:
  - DINOv2 preserves fine-grained spatial detail (no 75% masking)
  - Per-frame CLS tokens enable explicit temporal coherence analysis
  - Temporal transformer learns motion/flicker/drift patterns
  - Compatible with RA-Det's adversarial perturbation pipeline
    (DINOv2 = same family as DINOv3 used for images)

Interface is drop-in compatible with VideoMAEWrapper:
    encoder = DINOv2VideoEncoder(...)
    embedding = encoder(video_tensor)  # [B, 1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class DINOv2VideoEncoder(nn.Module):
    """
    Per-frame DINOv2 encoder with temporal transformer head.

    Accepts video tensors of shape [B, T, C, H, W] and returns:
      - Global video embeddings [B, D]  (via forward / encode_video)
      - Per-frame CLS tokens [B, T, D]  (via encode_video_detailed)
      - Temporal transformer outputs [B, T, D] (via encode_video_detailed)

    Args:
        model_name: HuggingFace DINOv2 model name (e.g. "facebook/dinov2-large").
        num_frames: Number of frames per clip (default: 16).
        temporal_layers: Number of temporal transformer layers (default: 2).
        temporal_heads: Number of attention heads in temporal transformer (default: 8).
        unfreeze_last_n: Number of DINOv2 encoder blocks to unfreeze for fine-tuning.
                         0 = fully frozen (default). 2 = unfreeze last 2 blocks.
        device: Device string (default: "cuda").
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        num_frames: int = 16,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        unfreeze_last_n: int = 0,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.num_frames = num_frames
        self.unfreeze_last_n = unfreeze_last_n

        # Load DINOv2 backbone
        print(f"Loading DINOv2 model: {model_name}...")
        self.dino = Dinov2Model.from_pretrained(model_name)
        self.dino = self.dino.to(device)

        self.output_dim = self.dino.config.hidden_size  # 1024 for Large, 768 for Base
        print(f"DINOv2 model loaded. Embedding dimension: {self.output_dim}")

        # ---- Freeze backbone ----
        for param in self.dino.parameters():
            param.requires_grad = False

        # Optionally unfreeze last N encoder blocks for fine-tuning
        if unfreeze_last_n > 0:
            total_layers = len(self.dino.encoder.layer)
            for layer in self.dino.encoder.layer[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"  Unfroze last {unfreeze_last_n}/{total_layers} DINOv2 encoder blocks")

        # ---- Temporal transformer ----
        # Processes the sequence of per-frame CLS tokens [B, T, D]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,
            nhead=temporal_heads,
            dim_feedforward=self.output_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
        )

        # Learnable temporal position embeddings
        self.temporal_pos = nn.Parameter(
            torch.randn(1, num_frames, self.output_dim) * 0.02
        )

        # Temporal statistics projection
        # Computes [mean, std, drift, jerk] from per-frame CLS → combined feature
        self.temporal_stats_proj = nn.Sequential(
            nn.Linear(self.output_dim * 4, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

        # Final projection: fuse temporal transformer output + temporal stats
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        # Store last per-frame features for external access
        self._last_per_frame_cls = None
        self._last_temporal_output = None

        self._initialize_temporal_weights()
        print(f"  Temporal transformer: {temporal_layers} layers, {temporal_heads} heads")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def _initialize_temporal_weights(self):
        """Initialize temporal head weights for stable training."""
        for module in [self.temporal_transformer, self.temporal_stats_proj, self.fusion_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _compute_temporal_stats(self, per_frame_cls: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal statistics from per-frame CLS tokens.

        Args:
            per_frame_cls: [B, T, D] CLS tokens per frame

        Returns:
            temporal_stats: [B, D] fused temporal statistics
        """
        mean_feat = per_frame_cls.mean(dim=1)       # [B, D] — global average
        std_feat = per_frame_cls.std(dim=1)          # [B, D] — temporal variance

        # Frame-to-frame drift (first derivative)
        drift = (per_frame_cls[:, 1:] - per_frame_cls[:, :-1]).mean(dim=1)  # [B, D]

        # Jerk (second derivative) — captures motion discontinuities
        jerk = (
            per_frame_cls[:, 2:] - 2 * per_frame_cls[:, 1:-1] + per_frame_cls[:, :-2]
        ).mean(dim=1)  # [B, D]

        combined = torch.cat([mean_feat, std_feat, drift, jerk], dim=-1)  # [B, 4*D]
        return self.temporal_stats_proj(combined)  # [B, D]

    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def encode_video_detailed(
        self, video_tensor: torch.Tensor
    ) -> tuple:
        """
        Full encoding with all intermediate outputs.

        Args:
            video_tensor: [B, T, C, H, W] video clip

        Returns:
            global_embedding: [B, D] fused temporal embedding
            per_frame_cls:    [B, T, D] per-frame CLS tokens from DINOv2
            temporal_output:  [B, T, D] temporal transformer output
        """
        assert video_tensor.dim() == 5, f"Expected 5D [B, T, C, H, W], got {video_tensor.dim()}D"
        B, T, C, H, W = video_tensor.shape

        # Process all frames through DINOv2 in a single batch
        frames = video_tensor.reshape(B * T, C, H, W)

        # DINOv2 forward — extract CLS token (position 0)
        # Frozen params → no gradient accumulation on backbone weights,
        # but gradients still flow through for trainable downstream modules.
        outputs = self.dino(frames)
        cls_tokens = outputs.last_hidden_state[:, 0]  # [B*T, D]

        # Reshape to [B, T, D]
        per_frame_cls = cls_tokens.reshape(B, T, -1).float()

        # Temporal transformer
        temporal_input = per_frame_cls + self.temporal_pos[:, :T]
        temporal_output = self.temporal_transformer(temporal_input)  # [B, T, D]

        # Global embedding: fuse transformer pooling + temporal stats
        transformer_pooled = temporal_output.mean(dim=1)  # [B, D]
        temporal_stats = self._compute_temporal_stats(per_frame_cls)  # [B, D]

        global_embedding = self.fusion_proj(
            torch.cat([transformer_pooled, temporal_stats], dim=-1)
        )  # [B, D]

        # Cache for external access
        self._last_per_frame_cls = per_frame_cls
        self._last_temporal_output = temporal_output

        return global_embedding, per_frame_cls, temporal_output

    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode video to global embedding (VideoMAEWrapper-compatible interface).

        Args:
            video_tensor: [B, T, C, H, W]

        Returns:
            global_embedding: [B, D]
        """
        global_embedding, _, _ = self.encode_video_detailed(video_tensor)
        return global_embedding

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns global embedding [B, D]."""
        return self.encode_video(video_tensor)

    def get_per_frame_features(self) -> torch.Tensor:
        """
        Get per-frame CLS tokens from the last forward pass.

        Returns:
            per_frame_cls: [B, T, D] or None if no forward pass has been done.
        """
        return self._last_per_frame_cls


def load_dinov2_video_encoder(
    model_name: str = "facebook/dinov2-large",
    num_frames: int = 16,
    temporal_layers: int = 2,
    unfreeze_last_n: int = 0,
    device: str = "cuda",
) -> DINOv2VideoEncoder:
    """Helper function to instantiate the DINOv2 video encoder."""
    return DINOv2VideoEncoder(
        model_name=model_name,
        num_frames=num_frames,
        temporal_layers=temporal_layers,
        unfreeze_last_n=unfreeze_last_n,
        device=device,
    )


if __name__ == "__main__":
    # Quick sanity check with dummy tensor
    print("Testing DINOv2VideoEncoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use base model for faster testing
    wrapper = DINOv2VideoEncoder(
        model_name="facebook/dinov2-base",
        num_frames=16,
        temporal_layers=2,
        unfreeze_last_n=0,
        device=device,
    )

    # Dummy video tensor: [B=2, T=16, C=3, H=224, W=224]
    dummy_video = torch.randn(2, 16, 3, 224, 224, device=device)

    # Test forward (VideoMAEWrapper-compatible)
    with torch.no_grad():
        embeddings = wrapper(dummy_video)
    print(f"Input shape: {dummy_video.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 768), f"Expected (2, 768), got {embeddings.shape}"

    # Test detailed encoding
    with torch.no_grad():
        global_emb, per_frame, temporal_out = wrapper.encode_video_detailed(dummy_video)
    print(f"Global embedding: {global_emb.shape}")
    print(f"Per-frame CLS: {per_frame.shape}")
    print(f"Temporal output: {temporal_out.shape}")
    assert per_frame.shape == (2, 16, 768)
    assert temporal_out.shape == (2, 16, 768)

    # Test per_frame accessor
    cached = wrapper.get_per_frame_features()
    assert cached is not None
    assert cached.shape == (2, 16, 768)

    print("✓ All DINOv2VideoEncoder tests passed!")
