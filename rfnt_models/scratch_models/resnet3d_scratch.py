"""
3D CNN Scratch Model for Spatiotemporal AI Detection.

This module implements a 3D ResNet model trained from scratch to process
spatiotemporal differences (e.g. concatenated noise and LPD features).
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


class ResNet3DScratch(nn.Module):
    """
    3D ResNet model (e.g., r3d_18) trained from scratch.
    
    Accepts spatiotemporal difference tensors of shape [B, C_in, T, H, W]
    and outputs binary classification logits [B, 1].
    """

    def __init__(self,
                 input_channels: int = 6,  # 3 (noise) + 3 (LPD) = 6 channels
                 architecture: str = "r3d_18",
                 pretrained: bool = False,
                 num_classes: int = 1):
        """
        Args:
            input_channels: Channels in input video tensor.
            architecture: Video ResNet variant ("r3d_18", "mc3_18", "r2plus1d_18").
            pretrained: Whether to use pretrained Kinetics-400 weights.
            num_classes: Number of output classes (typically 1 for binary).
        """
        super().__init__()
        self.input_channels = input_channels
        self.architecture = architecture
        self.pretrained = pretrained
        self.num_classes = num_classes

        self._build_model()

    def _build_model(self):
        """Loads and modifies the torchvision video ResNet backbone."""
        if self.architecture == "r3d_18":
            self.backbone = video_models.r3d_18(pretrained=self.pretrained)
        elif self.architecture == "mc3_18":
            self.backbone = video_models.mc3_18(pretrained=self.pretrained)
        elif self.architecture == "r2plus1d_18":
            self.backbone = video_models.r2plus1d_18(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported 3D backbone architecture: {self.architecture}")

        # Modify first convolution layer in self.backbone.stem[0] to accept self.input_channels
        # torchvision's video stem is usually a Sequential where index 0 is the Conv3D layer
        original_conv = self.backbone.stem[0]
        
        self.backbone.stem[0] = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Initialize new weights
        with torch.no_grad():
            if self.input_channels > 3:
                # Copy original 3-channel weights to first 3 channels
                self.backbone.stem[0].weight[:, :3, :, :, :] = original_conv.weight
                # Kaiming normal initialization for remaining channels
                for i in range(3, self.input_channels):
                    nn.init.kaiming_normal_(self.backbone.stem[0].weight[:, i, :, :, :])
            else:
                nn.init.kaiming_normal_(self.backbone.stem[0].weight)

        # Replace final classification head (self.backbone.fc)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatiotemporal tensor of shape [B, C_in, T, H, W]
        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        return self.backbone(x)

    def get_name(self) -> str:
        return f"resnet3d_scratch_{self.architecture}_{self.input_channels}ch"


if __name__ == "__main__":
    print("Testing ResNet3DScratch...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate 3D CNN scratch branch
    model = ResNet3DScratch(input_channels=6, architecture="r3d_18").to(device)
    
    # Dummy spatiotemporal difference tensor: [B=2, C=6, T=16, H=224, W=224]
    dummy_input = torch.randn(2, 6, 16, 224, 224, device=device)
    
    with torch.no_grad():
        logits = model(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (2, 1)
    print("✓ ResNet3DScratch test passed!")
