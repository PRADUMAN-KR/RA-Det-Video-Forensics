import torch
import torch.nn.functional as F
from models.video_encoder_wrapper import load_video_encoder

def test_gradients():
    print("Initializing test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_video_encoder(model_name="MCG-NJU/videomae-base", device=device)
    
    clean_video = torch.randn(1, 16, 3, 224, 224, device=device)
    
    # Test Cosine Similarity with MAXIMAL noise (simulating tanh saturation)
    eps = 32/255
    # Initialize noise to +eps and -eps randomly
    noise = (torch.randint_like(clean_video, 0, 2) * 2 - 1) * eps
    noise.requires_grad = True
    
    perturbed_video = clean_video + noise
    
    with torch.no_grad():
        clean_emb = encoder(clean_video)
        
    noisy_emb = encoder(perturbed_video)
    
    sim = F.cosine_similarity(clean_emb, noisy_emb, dim=1)
    loss = sim.mean()
    
    loss.backward()
    
    grad_norm = noise.grad.norm().item()
    grad_mean = noise.grad.abs().mean().item()
    
    print(f"\n--- Cosine Loss with Maximal Initial Noise (eps={eps:.3f}) ---")
    print(f"Similarity: {loss.item():.6f}")
    print(f"Gradient norm: {grad_norm:.6e}")
    print(f"Gradient mean: {grad_mean:.6e}")

if __name__ == "__main__":
    test_gradients()
