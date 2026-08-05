"""
VideoMAEv2 Wrapper for Spatiotemporal Video Detection.

This module wraps VideoMAEv2-Large from Hugging Face to provide
a spatiotemporal feature extractor that processes video tensors.
"""

import os
import torch
import torch.nn as nn
from transformers import VideoMAEModel


class VideoMAEWrapper(nn.Module):
    """
    Wrapper for VideoMAE models providing a unified spatiotemporal embedding interface.
    
    Accepts video tensors of shape [B, T, C, H, W] and returns global video embeddings [B, D].
    """

    def __init__(self, model_name: str = "MCG-NJU/videomae-large", device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        
        # Load VideoMAE model from Hugging Face
        print(f"Loading VideoMAE model: {model_name}...")
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.output_dim = self.model.config.hidden_size  # Typically 1024 for Large, 768 for Base
        print(f"VideoMAE model loaded successfully. Embedding dimension: {self.output_dim}")

    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to spatiotemporal embeddings.

        Args:
            video_tensor: Video tensor of shape [B, T, C, H, W]
                         where T is sequence length (number of frames).
                         Values should be normalized.

        Returns:
            embeddings: Video level embeddings of shape [B, D].
        """
        assert video_tensor.dim() == 5, f"Expected 5D [B, T, C, H, W], got {video_tensor.dim()}D"
        target_device = next(self.model.parameters()).device
        if video_tensor.device != target_device:
            video_tensor = video_tensor.to(target_device)

        # IMPORTANT: Do NOT gate on requires_grad here.
        #
        # The old check — `requires_grad = tensor.requires_grad or any(p.requires_grad ...)`
        # — always evaluates False in video mode because:
        #   (a) DataLoader tensors have requires_grad=False by default, and
        #   (b) the encoder is fully frozen (all params have requires_grad=False).
        # This caused the no_grad branch to be taken unconditionally, severing the
        # gradient path from the decoder output through this encoder to any
        # downstream trainable module (MLP classification heads, etc.).
        #
        # Frozen params will never accumulate or apply gradients regardless.
        # We simply let PyTorch autograd decide what to compute — if an outer
        # torch.no_grad() context is active (e.g. during validation), it will
        # suppress gradient computation automatically.
        outputs = self.model(pixel_values=video_tensor)
        features = outputs.last_hidden_state.mean(dim=1)
        return features.float()

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        return self.encode_video(video_tensor)


def load_video_encoder(model_name: str = "MCG-NJU/videomae-large", device: str = "cuda") -> VideoMAEWrapper:
    """Helper function to instantiate and return the video wrapper."""
    return VideoMAEWrapper(model_name=model_name, device=device)


if __name__ == "__main__":
    # Quick sanity check with dummy tensor
    print("Testing VideoMAEWrapper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use base model for dummy testing (faster loading)
    wrapper = VideoMAEWrapper(model_name="MCG-NJU/videomae-base", device=device)
    
    # Dummy video tensor: [B=2, T=16, C=3, H=224, W=224]
    dummy_video = torch.randn(2, 16, 3, 224, 224, device=device)
    
    with torch.no_grad():
        embeddings = wrapper(dummy_video)
        
    print(f"Input shape: {dummy_video.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 768)
    print("✓ Sanity test passed!")
