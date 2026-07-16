import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add release root directory to path
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RELEASE_ROOT)

from datasets.video_dataset import VideoDataset
from models.video_encoder_wrapper import load_video_encoder
from models.unet_3d import UNetDecoder3D
from rfnt_models.ensemble.video_classifier import VideoEnsembleClassifier
from rfnt_models.ensemble.npr import extract_npr_features
from configs.config import AIGCTEST_DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Train ONLY the NPR branch and fusion weights in RA-Det Ensemble.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training videos")
    parser.add_argument("--val_data", type=str, default=AIGCTEST_DATA_PATH, help="Path to validation videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to existing epoch-4 checkpoint to load")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/npr_ensemble", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Datasets
    print("Loading datasets...")
    train_dataset = VideoDataset(root_dir=args.train_data, split="train", num_frames=16, balance_classes=True)
    val_dataset = VideoDataset(root_dir=args.val_data, split="val", num_frames=16, balance_classes=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Models
    print("Initializing models...")
    feature_dim = 768  # For videomae_base, adjust if using large
    encoder = load_video_encoder("videomae_base", device=device)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    decoder = UNetDecoder3D(embed_dim=feature_dim, eps=16/255).to(device)
    
    classifier = VideoEnsembleClassifier(
        video_encoder=encoder,
        feature_dim=feature_dim,
        lpd_channels=0,
        temporal_diff_channels=3,
        resnet_variant="r3d_18",
        dropout=0.1,
        temperature=1.0,
        fusion_method="logit_weighted",
        use_four_branch=True,
        use_npr_branch=True  # ENABLE NPR BRANCH
    ).to(device)
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load decoder
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    for param in decoder.parameters():
        param.requires_grad = False
    decoder.eval()
    
    # Load classifier (strict=False because npr branch is new)
    missing_keys, unexpected_keys = classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
    print(f"Classifier load - Missing keys: {len(missing_keys)} (expected for NPR branch)")
    if unexpected_keys:
        print(f"Classifier load - Unexpected keys: {len(unexpected_keys)}")
        
    # 4. Freeze everything EXCEPT NPR branch and fusion strategy
    print("Freezing parameters...")
    for name, param in classifier.named_parameters():
        if "branches.npr" in name or "fusion_strategy" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Verify what is trainable
    trainable_params = [p for p in classifier.parameters() if p.requires_grad]
    print(f"Number of trainable tensors: {len(trainable_params)}")
    
    # 5. Optimizer and Loss
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # 6. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # We can keep encoder/decoder in eval mode to save memory
        encoder.eval()
        decoder.eval()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for videos, labels in pbar:
            videos, labels = videos.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # [B, 1]
            
            # Step 1: Compute NPR Features (no grads needed for extraction)
            with torch.no_grad():
                npr_features = extract_npr_features(videos)
                
                # Step 2: RA-Det forward pass to get all branch inputs
                clean_emb = encoder(videos)
                video_c_first = videos.permute(0, 2, 1, 3, 4)
                noise = decoder(original_video=video_c_first, clean_embedding=clean_emb)
                perturbed = (video_c_first + noise).permute(0, 2, 1, 3, 4)
                noisy_emb = encoder(perturbed)
                
                # Temporal diff
                frame_diffs = videos[:, 1:] - videos[:, :-1]
                frame_diffs = torch.cat([frame_diffs, frame_diffs[:, -1:]], dim=1)
                temporal_diff = frame_diffs.permute(0, 2, 1, 3, 4).contiguous()
                td_std = temporal_diff.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-4)
                temporal_diff = (temporal_diff / (3.0 * td_std)).clamp(-1.0, 1.0)
                
                l2_dist = torch.norm(clean_emb - noisy_emb, p=2, dim=1, keepdim=True)
                emb_diff = clean_emb - noisy_emb
            
            # Step 3: Classifier forward pass (computes gradients for NPR and fusion)
            optimizer.zero_grad()
            outputs = classifier(
                video=videos,
                noise=noise,
                temporal_diff=temporal_diff,
                l2_distance=l2_dist,
                embedding_diff=emb_diff,
                npr_features=npr_features,
                use_max_for_eval=False
            )
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * videos.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += videos.size(0)
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": correct / total})
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Train - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        # Save checkpoint
        save_path = os.path.join(args.out_dir, f"npr_ensemble_epoch_{epoch+1}.pt")
        # Merge with existing checkpoint dict
        checkpoint['classifier_state_dict'] = classifier.state_dict()
        checkpoint['epoch_npr'] = epoch + 1
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
