import torch

def extract_npr_features(video: torch.Tensor) -> torch.Tensor:
    """
    Extract Neighboring Pixel Relationships (NPR) features from video frames.
    
    This computes statistical moments (mean, std, skewness, kurtosis) for 
    the differences between adjacent pixels in 4 directions (vertical, 
    horizontal, 2x diagonal).
    
    Args:
        video: Tensor of shape [B, T, C, H, W] or [B, C, H, W].
               Values can be in [0, 1] or normalized.
               
    Returns:
        features: Tensor of shape [B, 48] if C=3 (4 stats * 4 dirs * 3 channels)
                  If input is 5D, features are averaged across the temporal dimension.
    """
    if video.dim() == 5:
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)
    elif video.dim() == 4:
        B, C, H, W = video.shape
        x = video
        T = 1
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {video.dim()}D")
        
    # Calculate pixel differences in 4 directions
    diff_v  = x[:, :, 1:, :] - x[:, :, :-1, :]              # Vertical
    diff_h  = x[:, :, :, 1:] - x[:, :, :, :-1]              # Horizontal
    diff_d1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]           # Diagonal (top-left to bottom-right)
    diff_d2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]           # Anti-diagonal (top-right to bottom-left)
    
    features = []
    
    for diff in [diff_v, diff_h, diff_d1, diff_d2]:
        # Flatten spatial dimensions: [BT, C, N]
        diff_flat = diff.reshape(diff.shape[0], diff.shape[1], -1)
        
        # 1. Mean
        mean = diff_flat.mean(dim=2)  # [BT, C]
        
        # 2. Standard Deviation
        var = diff_flat.var(dim=2, unbiased=False)
        std = torch.sqrt(var + 1e-8)  # [BT, C]
        
        # Center the differences for higher moments
        diff_centered = diff_flat - mean.unsqueeze(2)
        
        # 3. Skewness = E[(X - mu)^3] / sigma^3
        skew = (diff_centered ** 3).mean(dim=2) / (std ** 3 + 1e-8)  # [BT, C]
        
        # 4. Kurtosis = E[(X - mu)^4] / sigma^4
        kurt = (diff_centered ** 4).mean(dim=2) / (var ** 2 + 1e-8)  # [BT, C]
        
        features.extend([mean, std, skew, kurt])
        
    # Concatenate all features: 16 stats (4 moments * 4 dirs) per channel
    # For C=3, this gives 48 features per frame
    feature_tensor = torch.cat(features, dim=1)  # [BT, 16*C]
    
    if video.dim() == 5:
        # Aggregate across time (average features over all frames in the clip)
        feature_tensor = feature_tensor.reshape(B, T, -1).mean(dim=1)  # [B, 16*C]
        
    return feature_tensor
