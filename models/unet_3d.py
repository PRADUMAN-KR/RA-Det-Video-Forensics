"""
3D UNet-based Perturbation Decoder (DRP) for Video.

This module implements a 3D spatiotemporal UNet decoder that:
1. Accepts original video, clean embedding, and optional multi-scale features.
2. Performs 3D downsampling, spatiotemporal cross-attention bottleneck, and 3D upsampling.
3. Generates temporally coherent adversarial noise bounded by epsilon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention3D(nn.Module):
    """Efficient spatiotemporal multi-head attention module."""

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv3d(in_channels, key_channels, 1)
        self.queries = nn.Conv3d(in_channels, key_channels, 1)
        self.values = nn.Conv3d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, c, t, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, t * h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, t * h * w)
        values = self.values(input_).reshape((n, self.value_channels, t * h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, t, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class ConvBlock3D(nn.Module):
    """
    Basic 3D convolutional block with optional attention.
    """

    def __init__(self, in_channels, out_channels, use_attention=False, key_channels=None, head_count=8):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.use_attention = use_attention
        if use_attention:
            if key_channels is None:
                key_channels = out_channels // 8
            self.attention = EfficientAttention3D(
                out_channels,
                key_channels=key_channels,
                head_count=head_count,
                value_channels=out_channels
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        if self.use_attention:
            x = self.attention(x)

        return x


class EncoderStage3D(nn.Module):
    """
    Encoder 3D downsampling stage.
    """

    def __init__(self, in_channels, out_channels, downsample=True, use_attention=False):
        super().__init__()

        self.downsample = downsample
        if downsample:
            # Downsample spatially by factor of 2, keep temporal resolution constant
            self.downsample_conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1)
            )
            self.downsample_bn = nn.BatchNorm3d(in_channels)

        self.conv_block = ConvBlock3D(
            in_channels,
            out_channels,
            use_attention=use_attention
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_conv(x)
            x = self.downsample_bn(x)
            x = F.leaky_relu_(x)

        x = self.conv_block(x)
        return x


class DecoderStage3D(nn.Module):
    """
    Decoder 3D upsampling stage with skip connection.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False, upsample_first=True):
        super().__init__()

        self.upsample_first = upsample_first
        # Spatially upsample by factor of 2, keeping time resolution T constant
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')

        total_channels = in_channels + skip_channels

        self.conv_block = ConvBlock3D(
            total_channels,
            out_channels,
            use_attention=use_attention
        )

    def forward(self, x, skip):
        if self.upsample_first:
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
        else:
            x = torch.cat([x, skip], dim=1)
            x = self.upsample(x)

        x = self.conv_block(x)
        return x


class CrossAttentionBottleneck3D(nn.Module):
    """
    Cross-attention bottleneck for fusing spatiotemporal image and embedding features.
    """

    def __init__(self, image_channels=512, embed_channels=512, num_heads=8):
        super().__init__()

        self.image_channels = image_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads

        # Cross-attention: video query attends to embedding key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Self-attention for spatiotemporal refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(image_channels)
        self.norm2 = nn.LayerNorm(image_channels)

        self.ffn = nn.Sequential(
            nn.Linear(image_channels, image_channels * 4),
            nn.GELU(),
            nn.Linear(image_channels * 4, image_channels)
        )
        self.norm3 = nn.LayerNorm(image_channels)

    def forward(self, image_feat, clean_embed_feat, noisy_embed_feat):
        B, C, T, H, W = image_feat.shape

        # Flatten spatiotemporal dimensions: [B, T*H*W, C]
        image_flat = image_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        clean_flat = clean_embed_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        noisy_flat = noisy_embed_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)

        # Concat embeddings along sequence length
        embed_feat = torch.cat([clean_flat, noisy_flat], dim=1)  # [B, 2*T*H*W, C]

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=image_flat,
            key=embed_feat,
            value=embed_feat
        )
        image_flat = self.norm1(image_flat + attn_out)

        # Self-attention
        self_attn_out, _ = self.self_attn(image_flat, image_flat, image_flat)
        image_flat = self.norm2(image_flat + self_attn_out)

        # FFN
        ffn_out = self.ffn(image_flat)
        image_flat = self.norm3(image_flat + ffn_out)

        # Reshape back to [B, C, T, H, W]
        output = image_flat.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        return output


class InputProcessor3D(nn.Module):
    """
    Concatenates original video and optional multi-scale features, projecting to base channels.
    """

    def __init__(self, base_channels=64, strategy_channels=0):
        super().__init__()

        total_input_channels = 3 + strategy_channels

        self.input_conv = nn.Sequential(
            nn.Conv3d(total_input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, original_video, multi_scale_videos=None):
        inputs = [original_video]

        if multi_scale_videos is not None:
            inputs.append(multi_scale_videos)

        x = torch.cat(inputs, dim=1)
        x = self.input_conv(x)

        return x


class EmbeddingEncoder3D(nn.Module):
    """
    Converts 1D video embeddings to spatiotemporal 3D features.
    
    Uses parameter-efficient spatial projection and temporal tiling.
    """

    def __init__(self, embed_dim=1024, spatial_size=7, channels=512):
        super().__init__()

        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.channels = channels

        # Projects global embedding to spatial 2D grid
        self.proj = nn.Linear(embed_dim, channels * spatial_size ** 2)

        # Refines in 3D after temporal tiling
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, embeddings, num_frames):
        B = embeddings.size(0)

        # Project to 2D
        x = self.proj(embeddings.float())  # [B, C * H * W]
        x = x.view(B, self.channels, 1, self.spatial_size, self.spatial_size) # [B, C, 1, H, W]

        # Tile along temporal dimension
        x = x.repeat(1, 1, num_frames, 1, 1) # [B, C, T, H, W]

        # Refine spatiotemporally
        x = self.refine(x)

        return x


class OutputHead3D(nn.Module):
    """Generates final bounded spatiotemporal noise."""

    def __init__(self, in_channels=64, eps=16/255):
        super().__init__()
        self.eps = eps

        self.output = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        noise = torch.tanh(self.output(x))
        noise = noise * self.eps
        return noise


class UNetDecoder3D(nn.Module):
    """
    3D spatiotemporal UNet-based perturbation decoder.
    """

    def __init__(
        self,
        embed_dim=1024,
        strategy_channels=0,
        base_channels=64,
        num_levels=5,
        num_heads=8,
        use_attention=True,
        eps=16/255,
        bottleneck_size=7
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.strategy_channels = strategy_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.eps = eps
        self.bottleneck_size = bottleneck_size

        # Input processor
        self.input_processor = InputProcessor3D(
            base_channels=base_channels,
            strategy_channels=strategy_channels
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        channels = base_channels

        for i in range(num_levels):
            out_channels = min(base_channels * (2 ** i), 512)
            self.encoder_stages.append(
                EncoderStage3D(
                    channels,
                    out_channels,
                    downsample=(i < num_levels - 1),
                    use_attention=use_attention and (i > 0)
                )
            )
            channels = out_channels

        # Embedding encoders
        self.clean_embedding_encoder = EmbeddingEncoder3D(
            embed_dim=embed_dim,
            spatial_size=bottleneck_size,
            channels=channels
        )
        self.noisy_embedding_encoder = EmbeddingEncoder3D(
            embed_dim=embed_dim,
            spatial_size=bottleneck_size,
            channels=channels
        )

        # Spatiotemporal bottleneck
        self.bottleneck = CrossAttentionBottleneck3D(
            image_channels=channels,
            embed_channels=channels,
            num_heads=num_heads
        )

        # Decoder stages with skip connections
        self.decoder_stages = nn.ModuleList()
        self.skip_connections = []

        for decoder_stage_idx in range(num_levels - 1):
            encoder_stage_idx = (num_levels - 2) - decoder_stage_idx
            skip_channels = min(base_channels * (2 ** encoder_stage_idx), 512)

            if decoder_stage_idx < num_levels - 2:
                out_channels = min(base_channels * (2 ** ((num_levels - 2) - decoder_stage_idx - 1)), 512)
            else:
                out_channels = base_channels

            self.decoder_stages.append(
                DecoderStage3D(
                    in_channels=channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    use_attention=use_attention,
                    upsample_first=False
                )
            )
            self.skip_connections.append(encoder_stage_idx)
            channels = out_channels

        # Output head
        self.output_head = OutputHead3D(
            in_channels=channels,
            eps=eps
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, original_video, clean_embedding, multi_scale_videos=None):
        """
        Args:
            original_video: [B, 3, T, H, W]
            clean_embedding: [B, embed_dim]
            multi_scale_videos: [B, strategy_channels, T, H, W] or None
            
        Returns:
            noise: [B, 3, T, H, W] Bounded spatiotemporal perturbation
        """
        num_frames = original_video.shape[2]

        # 1. Input processing
        x = self.input_processor(original_video, multi_scale_videos)

        # 2. Downsampling encoder
        encoder_skips = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i < self.num_levels - 1:
                encoder_skips.append(x)

        # 3. Embedding Projection & Tiling
        clean_embed_feat = self.clean_embedding_encoder(clean_embedding, num_frames)
        noisy_embed_feat = self.noisy_embedding_encoder(clean_embedding, num_frames)

        # 4. Spatiotemporal Bottleneck fusion
        x = self.bottleneck(x, clean_embed_feat, noisy_embed_feat)

        # 5. Decoder with skip connections
        for i, stage in enumerate(self.decoder_stages):
            encoder_stage_idx = self.skip_connections[i]
            skip = encoder_skips[encoder_stage_idx]
            x = stage(x, skip)

        # 6. Generate final noise
        noise = self.output_head(x)

        return noise


if __name__ == "__main__":
    print("Testing 3D UNet...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate 3D UNet
    model = UNetDecoder3D(
        embed_dim=768, 
        strategy_channels=15, 
        bottleneck_size=7,
        eps=16/255
    ).to(device)
    
    # Dummy tensors
    original_video = torch.randn(2, 3, 16, 224, 224, device=device) # [B=2, C=3, T=16, H=224, W=224]
    multi_scale = torch.randn(2, 15, 16, 224, 224, device=device)
    embeddings = torch.randn(2, 768, device=device)
    
    with torch.no_grad():
        noise = model(original_video, embeddings, multi_scale)
        
    print(f"Original video shape: {original_video.shape}")
    print(f"Output noise shape: {noise.shape}")
    assert noise.shape == original_video.shape
    print(f"Noise range: [{noise.min().item():.4f}, {noise.max().item():.4f}]")
    print("✓ 3D UNet test passed!")
