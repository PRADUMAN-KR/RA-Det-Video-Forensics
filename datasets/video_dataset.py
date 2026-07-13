"""
Video Dataset Loader for Spatiotemporal AI Detection.

This module provides dataset classes for loading video sequences (real and fake)
either directly from video files (mp4, mkv, etc.) or folders of pre-extracted frame images.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

# Try to import decord for fast video loading, fallback to PyAV or PIL sequence
try:
    import decord
    decord.bridge.set_bridge('torch')
    # Force disable decord due to known deadlock/hang issues inside PyTorch DataLoader threads
    DECORD_AVAILABLE = False
except ImportError:
    DECORD_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False


class VideoDataset(Dataset):
    """
    Spatiotemporal video dataset.
    Loads video clips and returns tensors of shape [T, C, H, W] representing T frames.
    """
    
    VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def __init__(self,
                 root_dir: str,
                 num_frames: int = 16,
                 transform=None,
                 split: str = "train",  # "train" or "val"
                 max_samples: int = None,
                 balance_classes: bool = False):
        """
        Args:
            root_dir: Root directory containing real and fake videos.
            num_frames: Sequence length (number of frames to extract).
            transform: Transforms to apply to each frame individually.
            split: Dataset split.
            max_samples: Limit total samples.
            balance_classes: Balance real and fake sample counts.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.split = split
        self.balance_classes = balance_classes

        # Default transforms matching VideoMAE expected resolution and normalization.
        # VideoMAE was pretrained with mean=[0.5, 0.5, 0.5] std=[0.5, 0.5, 0.5],
        # NOT ImageNet statistics. Using ImageNet stats shifts the input distribution
        # and degrades embedding quality.
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # VideoMAE pretraining stats
            ])
        else:
            self.transform = transform

        self.samples = []  # List of dicts with 'path', 'label', 'type' ('video' or 'folder')
        self._load_samples()
        
        if balance_classes:
            self._balance_dataset()

        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        print(f"Loaded {len(self.samples)} video samples: "
              f"{sum(1 for s in self.samples if s['label'] == 1)} fake, "
              f"{sum(1 for s in self.samples if s['label'] == 0)} real.")

    def _load_samples(self):
        """Recursively scans root_dir for video files or image folders."""
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Check if this directory represents a real or fake folder
            label = None
            if "0_real" in Path(dirpath).parts or "0_real" in dirnames:
                label = 0
            elif "1_fake" in Path(dirpath).parts or "1_fake" in dirnames:
                label = 1

            # Extract generator type from path parts if available
            generator = "unknown"
            # Comprehensive list of known generators and dataset names:
            # - Classic deepfake datasets: FaceForensics++, DFDC, CelebDF, WildDeepfake, DeeperForensics
            # - FF++ manipulation methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter
            # - GAN-based video generators: ProGAN, StyleGAN, BigGAN
            # - Diffusion/flow-based video generators: Stable Diffusion Video, ModelScope, ZeroScope
            # - Next-gen commercial platforms: Sora, Runway Gen-2/Gen-3, Pika, Kling, Luma Dream Machine,
            #                                  Veo (Google), Wan (Alibaba), Hailuo, Morphic, Lightricks LTX
            KNOWN_GENERATORS = {
                # --- Classic Deepfake Datasets ---
                'faceforensics', 'faceforensics++', 'ff++',
                'dfdc', 'deepfakedetectionchallenge',
                'celebdf', 'celeb_df', 'celeb-df',
                'wilddeepfake', 'wild_deepfake',
                'deeperforensics', 'deeper_forensics',
                'fakedav', 'fake_av',

                # --- FF++ Manipulation Methods ---
                'deepfakes', 'deepfake',
                'face2face', 'faceswap', 'face_swap',
                'neuraltextures', 'neural_textures',
                'faceshifter', 'face_shifter',

                # --- GAN-based Generators ---
                'progan', 'stylegan', 'stylegan2', 'biggan',
                'stargan', 'cyclegan',

                # --- Diffusion / Flow-based Video Generators ---
                'stable_diffusion', 'stable_diffusion_video', 'sdv',
                'modelscope', 'zeroscope', 'animatediff',
                'opensora', 'open_sora', 'cogvideo', 'cogvideox',
                'lavie', 'show_one', 'videocrafter',

                # --- Next-Gen Commercial Platforms ---
                'sora',                          # OpenAI Sora
                'runway', 'gen2', 'gen3',        # Runway Gen-2 / Gen-3
                'pika', 'pika_labs',             # Pika Labs
                'luma', 'lumaai',                # Luma Dream Machine
                'kling', 'klingai',              # Kuaishou Kling
                'veo', 'veo2',                   # Google Veo
                'wan', 'wan2',                   # Alibaba Wan
                'hailuo', 'minimax',             # MiniMax Hailuo
                'ltx', 'lightricks',             # Lightricks LTX-Video
                'morphic', 'pixverse',           # Other platforms
                'genmo', 'mochi',                # Genmo Mochi
                'hunyuan', 'hunyuanvideo',       # Tencent HunyuanVideo
                'step', 'stepvideo',             # Step Video
            }
            for part in Path(dirpath).parts:
                if part.lower() in KNOWN_GENERATORS:
                    generator = part.lower()
                    break

            # 1. Collect Video Files
            for name in filenames:
                if name.lower().endswith(self.VIDEO_EXTS):
                    video_path = os.path.join(dirpath, name)
                    # Correct label detection based on filename parent directory
                    item_label = label
                    if item_label is None:
                        # Try to detect from file parent dir name
                        parent_name = os.path.basename(dirpath)
                        if "real" in parent_name.lower() or "0" in parent_name:
                            item_label = 0
                        elif "fake" in parent_name.lower() or "1" in parent_name:
                            item_label = 1
                        else:
                            continue  # Skip if can't resolve label
                            
                    self.samples.append({
                        'path': video_path,
                        'label': item_label,
                        'type': 'video',
                        'generator': generator
                    })

            # 2. Collect Pre-extracted Frame Directories (Folders containing images)
            # If a folder contains images, and its name matches a layout pattern
            has_images = any(fn.lower().endswith(self.IMG_EXTS) for fn in filenames)
            if has_images and label is not None:
                # Add this directory as a folder sample
                # Verify it has enough frames to sample from
                img_count = sum(1 for fn in filenames if fn.lower().endswith(self.IMG_EXTS))
                if img_count >= 2:  # Minimum frames to interpolate or repeat
                    self.samples.append({
                        'path': dirpath,
                        'label': label,
                        'type': 'folder',
                        'generator': generator
                    })

    def _balance_dataset(self):
        """Balances real and fake video counts."""
        real_samples = [s for s in self.samples if s['label'] == 0]
        fake_samples = [s for s in self.samples if s['label'] == 1]
        
        min_count = min(len(real_samples), len(fake_samples))
        if min_count == 0:
            print("Warning: Cannot balance classes since one class has 0 samples.")
            return

        sampled_real = random.sample(real_samples, min_count)
        sampled_fake = random.sample(fake_samples, min_count)
        
        self.samples = sampled_real + sampled_fake
        random.shuffle(self.samples)

    def _load_video_decord(self, video_path: str) -> torch.Tensor:
        """Loads video file using decord."""
        vr = decord.VideoReader(video_path, width=224, height=224)
        total_frames = len(vr)
        
        # Sample T frame indices uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices)  # Shape [T, H, W, C] (PyTorch Tensor due to bridge)
        
        # Permute to [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        
        # Apply standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        return frames

    def _load_video_pyav(self, video_path: str) -> torch.Tensor:
        """Loads video file using PyAV fallback."""
        container = av.open(video_path)
        
        # Decode and store raw frames first to avoid processing unused frames
        raw_frames = []
        try:
            for frame in container.decode(video=0):
                raw_frames.append(frame)
        except Exception:
            # Ignore decoder issues at the end of the video stream
            pass
        finally:
            container.close()
            
        total_frames = len(raw_frames)
        if total_frames == 0:
            raise ValueError(f"No frames could be decoded from video: {video_path}")
            
        # Sample T frame indices uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # Only process (convert to PIL image and apply transforms) the selected frames
        selected_frames = []
        for i in indices:
            img = raw_frames[i].to_image()
            selected_frames.append(self.transform(img))
            
        return torch.stack(selected_frames, dim=0) # [T, C, H, W]

    def _load_folder_frames(self, folder_path: str) -> torch.Tensor:
        """Loads sequence of images from frame directory."""
        files = sorted(os.listdir(folder_path))
        img_paths = [os.path.join(folder_path, f) for f in files if f.lower().endswith(self.IMG_EXTS)]
        
        total_frames = len(img_paths)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames_list = []
        for idx in indices:
            img = Image.open(img_paths[idx]).convert('RGB')
            frames_list.append(self.transform(img))
            
        return torch.stack(frames_list, dim=0)  # [T, C, H, W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        label = sample['label']
        sample_type = sample['type']
        generator = sample['generator']

        try:
            if sample_type == 'folder':
                video_tensor = self._load_folder_frames(path)
            else:
                if DECORD_AVAILABLE:
                    video_tensor = self._load_video_decord(path)
                elif AV_AVAILABLE:
                    video_tensor = self._load_video_pyav(path)
                else:
                    raise ImportError("Neither Decord nor PyAV is available for direct video loading. "
                                      "Please pre-extract your frames or install decord.")
        except Exception as e:
            # If load fails (e.g. corrupt video file), fall back to next random index to prevent crash
            print(f"Error loading video {path}: {e}. Retrying with another sample.")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        return {
            'video': video_tensor,  # [T, C, H, W]
            'label': torch.tensor(label, dtype=torch.long),
            'generator': generator,
            'path': path
        }

    def get_generator_list(self) -> list:
        """Get list of all unique generators in the dataset"""
        return list(set(s['generator'] for s in self.samples))


if __name__ == "__main__":
    # Test loader with a temporary mock directory structure
    print("Testing VideoDataset loader...")
    # Create temp directory layout for test (simulated)
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "0_real")
        fake_dir = os.path.join(tmpdir, "1_fake")
        os.makedirs(real_dir)
        os.makedirs(fake_dir)
        
        # Save mock image sequences
        for d in [real_dir, fake_dir]:
            for i in range(16):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(os.path.join(d, f"frame_{i:04d}.png"))
                
        dataset = VideoDataset(root_dir=tmpdir, num_frames=8)
        dataloader = DataLoader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        print(f"Loaded batch video tensor shape: {batch['video'].shape}")
        print(f"Labels: {batch['label']}")
        
        assert batch['video'].shape == (2, 8, 3, 224, 224)
        print("✓ VideoDataset test passed!")
