"""
Balanced batch sampler for video deepfake detection training.

Guarantees every batch contains exactly half real (label=0) and half fake (label=1)
samples. This is critical for:
  1. Discrepancy loss — requires both real and fake in each batch to compute the
     real_similarity / fake_similarity gap. A single-class batch makes embedding_loss=0.
  2. Classifier accuracy — prevents trivial acc=0.0 or acc=1.0 from same-class batches.
  3. Small batch sizes (e.g. batch_size=2) — with random sampling at B=2, ~50% of
     batches are single-class when dataset is imbalanced (Kinetics-400 vs GenBuster).
"""

import random
from torch.utils.data import Sampler
from typing import Iterator, List


class BalancedVideoSampler(Sampler):
    """
    Yields batch indices guaranteeing each batch has exactly half real and half fake samples.

    Compatible with:
      - Single-GPU training (use as batch_sampler in DataLoader)
      - DDP training (set_epoch() mirrors DistributedSampler API for reproducibility)

    Args:
        dataset:    VideoDataset instance (must have .samples list with 'label' key).
        batch_size: Total number of samples per batch (must be even).
        drop_last:  Drop the last incomplete batch (recommended True for training).
        seed:       Base random seed. Updated each epoch via set_epoch().
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True, seed: int = 42):
        super().__init__()

        if batch_size % 2 != 0:
            raise ValueError(
                f"BalancedVideoSampler requires an even batch_size, got {batch_size}."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Partition indices by label
        self.real_indices: List[int] = [
            i for i, s in enumerate(dataset.samples) if s['label'] == 0
        ]
        self.fake_indices: List[int] = [
            i for i, s in enumerate(dataset.samples) if s['label'] == 1
        ]

        if len(self.real_indices) == 0 or len(self.fake_indices) == 0:
            raise ValueError(
                f"BalancedVideoSampler: dataset must contain BOTH real and fake samples.\n"
                f"  Found: real={len(self.real_indices)}, fake={len(self.fake_indices)}\n"
                f"  Check that your dataset directory has both 0_real/ and 1_fake/ subdirs\n"
                f"  (or that real_dir / fake_dir args are set correctly)."
            )

        self.half = batch_size // 2
        print(
            f"[BalancedVideoSampler] real={len(self.real_indices)}, "
            f"fake={len(self.fake_indices)}, half_per_batch={self.half}"
        )

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch (mirrors DistributedSampler API)."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        real = self.real_indices.copy()
        fake = self.fake_indices.copy()
        rng.shuffle(real)
        rng.shuffle(fake)

        # Extend shorter list by repeating so both are the same length
        max_len = max(len(real), len(fake))
        if len(real) < max_len:
            repeats = (max_len // len(real)) + 1
            real = (real * repeats)[:max_len]
        if len(fake) < max_len:
            repeats = (max_len // len(fake)) + 1
            fake = (fake * repeats)[:max_len]

        # Build balanced batches: half real + half fake each
        for start in range(0, max_len - self.half + 1, self.half):
            batch_real = real[start: start + self.half]
            batch_fake = fake[start: start + self.half]

            if len(batch_real) < self.half or len(batch_fake) < self.half:
                if self.drop_last:
                    break
                while len(batch_real) < self.half:
                    batch_real.append(rng.choice(self.real_indices))
                while len(batch_fake) < self.half:
                    batch_fake.append(rng.choice(self.fake_indices))

            combined = batch_real + batch_fake
            rng.shuffle(combined)   # Interleave real and fake within the batch
            yield combined

    def __len__(self) -> int:
        max_len = max(len(self.real_indices), len(self.fake_indices))
        n_complete = max_len // self.half
        return n_complete if self.drop_last else n_complete + (1 if max_len % self.half else 0)
