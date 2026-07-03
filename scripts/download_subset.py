#!/usr/bin/env python3
"""
Dataset download script for RA-Det (Robustness-Aware Deepfake Detection).

RA-Det detects deepfakes via BEHAVIORAL signatures under adversarial perturbation —
not pixel-level appearance. Strategy:
  - Real  : Kinetics-400 (all 400 classes → maximum behavioral diversity)
  - Fake  : AI/diffusion-model generated videos (embedding-space artifacts
            exposed by RA-Det's adversarial perturbation)

Verified HuggingFace fake sources (2026-07):
  ┌─────────────────────────────────┬──────────┬──────────────────────────────────────┬────────┐
  │ Repo                            │ Videos   │ Models                               │ Access │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────┼────────┤
  │ l8cv/GenBuster-200K             │ 200,000  │ Sora, WanX, Kling, CogVideo, SVD,    │ Gated* │
  │                                 │          │ Pika, ModelScope, VideoCrafter        │        │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────┼────────┤
  │ ductai199x/synth-vid-detect     │ ~10,000  │ CogVideo, LumaAI, Pika, Sora, SVD,  │ Public │
  │                                 │          │ VideoCrafter, VideoCrafter-v2         │        │
  ├─────────────────────────────────┼──────────┼──────────────────────────────────────┼────────┤
  │ 34data/v14-fake-opensora        │ ~2-10K   │ OpenSora v1                          │ Public │
  │ 34data/v14-fake-opensora12      │ ~2-10K   │ OpenSora v1.2                        │ Public │
  └─────────────────────────────────┴──────────┴──────────────────────────────────────┴────────┘
  * Gated = auto-approve. Accept terms at: https://huggingface.co/datasets/l8cv/GenBuster-200K
    GenBuster-200K stored as 27x 4GB .7z parts — requires p7zip to extract.
    Install: brew install p7zip  OR  apt install p7zip-full

Real source:
  liuhuanjim013/kinetics400
    videos/<action_class>/<video_id>.mp4  (400 class subdirs, gated)
    Accept terms at: https://huggingface.co/datasets/liuhuanjim013/kinetics400

Usage:
  # Best for VideoMAE Large (0.3B) — GenBuster-200K:
  python3 scripts/download_subset.py --count 50000 --hf-token <TOKEN> --fake-source genbuster

  # Public-only, no gating (synth-vid-detect + OpenSora):
  python3 scripts/download_subset.py --count 10000 --hf-token <TOKEN> --fake-source public

  # All sources combined:
  python3 scripts/download_subset.py --count 50000 --hf-token <TOKEN> --fake-source all

  # Quick test:
  python3 scripts/download_subset.py --count 100 --hf-token <TOKEN>
"""

import os
import sys
import shutil
import zipfile
import tarfile
import argparse
import subprocess
import random

# Force unbuffered output so you can see progress in log files immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------

def install_dependencies():
    print("Checking and installing dependencies...")
    try:
        import huggingface_hub  # noqa: F401
        import tqdm             # noqa: F401
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub", "tqdm"]
            )
        except subprocess.CalledProcessError:
            print("\nStandard pip install failed. Trying with --break-system-packages (for modern Ubuntu containers)...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "huggingface_hub", "tqdm", "--break-system-packages"]
                )
            except subprocess.CalledProcessError:
                print("\nFallback failed. Trying with --user...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "huggingface_hub", "tqdm", "--user"]
                )


# ---------------------------------------------------------------------------
# Real videos: Kinetics-400 (ALL 400 classes — behavioral diversity)
# ---------------------------------------------------------------------------

def download_kinetics400_real(dest_dir, num_videos, hf_token):
    """Download up to num_videos MP4s from liuhuanjim013/kinetics400 (all classes)."""
    from huggingface_hub import HfFileSystem, hf_hub_download
    from tqdm import tqdm

    _section(f"Downloading {num_videos} REAL Videos — Kinetics-400 (all 400 classes)")
    print("  Rationale: behavioral diversity across all human actions")
    os.makedirs(dest_dir, exist_ok=True)

    REPO_ID = "liuhuanjim013/kinetics400"

    print("\n  Listing class directories...")
    try:
        fs = HfFileSystem(token=hf_token)
        class_dirs = fs.ls(f"datasets/{REPO_ID}/videos", detail=False)
    except Exception as e:
        print(f"  Error: {e}")
        print("  → Accept terms: https://huggingface.co/datasets/liuhuanjim013/kinetics400")
        return 0

    print(f"  Found {len(class_dirs)} action classes")
    per_class = max(1, (num_videos + len(class_dirs) - 1) // len(class_dirs))
    all_files = []

    for class_path in tqdm(class_dirs, desc="  Scanning classes"):
        try:
            files = fs.ls(class_path, detail=False)
            mp4_files = [f for f in files if f.endswith(".mp4")]
            sample = random.sample(mp4_files, min(per_class, len(mp4_files)))
            all_files.extend(sample)
        except Exception:
            continue
        if len(all_files) >= num_videos * 2:
            break

    random.shuffle(all_files)
    all_files = all_files[:num_videos]
    print(f"  Downloading {len(all_files)} videos...")

    downloaded, failed = 0, 0
    prefix = f"datasets/{REPO_ID}/"
    for hf_path in tqdm(all_files, desc="  Kinetics-400"):
        filename = hf_path[len(prefix):] if hf_path.startswith(prefix) else hf_path
        dest_path = os.path.join(dest_dir, os.path.basename(filename))
        if os.path.exists(dest_path):
            downloaded += 1
            continue
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type="dataset",
                token=hf_token, cache_dir="/tmp/hf_cache",
            )
            shutil.copy2(cached, dest_path)
            downloaded += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n  Warning: {os.path.basename(filename)}: {e}")

    print(f"\n  ✓ Real: {downloaded} videos → {dest_dir}" +
          (f" ({failed} failed)" if failed else ""))
    return downloaded


# ---------------------------------------------------------------------------
# Fake source 1: GenBuster-200K  (200K diffusion-generated videos, auto-gated)
# ---------------------------------------------------------------------------

def download_genbuster(dest_dir, num_videos, hf_token):
    """
    Download from l8cv/GenBuster-200K.
    200K videos: Sora, WanX, Kling, CogVideo, SVD, Pika, ModelScope, VideoCrafter.
    Stored as 27x 4GB .7z parts. Requires p7zip.
    Accept terms: https://huggingface.co/datasets/l8cv/GenBuster-200K
    """
    from huggingface_hub import hf_hub_download

    _section(f"Downloading {num_videos} FAKE Videos — GenBuster-200K")
    print("  Models: Sora, WanX, Kling, CogVideo, SVD, Pika, ModelScope, VideoCrafter")
    os.makedirs(dest_dir, exist_ok=True)

    REPO_ID = "l8cv/GenBuster-200K"
    NUM_PARTS = 27

    cmd_7z = shutil.which("7z") or shutil.which("7za")
    if cmd_7z is None:
        print("  ✗ p7zip not found.")
        print("    Install: brew install p7zip   OR   apt install p7zip-full")
        print("  Skipping GenBuster-200K.")
        return 0

    staging = os.path.join(dest_dir, "_genbuster_staging")
    os.makedirs(staging, exist_ok=True)

    print(f"\n  Downloading {NUM_PARTS} archive parts (~108 GB total)...")
    for i in range(1, NUM_PARTS + 1):
        part_name = f"GenBuster-200K.7z.{i:03d}"
        part_path = os.path.join(staging, part_name)
        if os.path.exists(part_path):
            print(f"  ✓ {part_name} cached")
            continue
        print(f"  ↓ {part_name}...")
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=part_name, repo_type="dataset",
                token=hf_token, cache_dir="/tmp/hf_cache",
            )
            shutil.copy2(cached, part_path)
        except Exception as e:
            print(f"  ✗ {part_name}: {e}")
            if i == 1:
                print("  Cannot extract without part 1. Skipping.")
                return 0

    print(f"\n  Extracting with {cmd_7z}...")
    first_part = os.path.join(staging, "GenBuster-200K.7z.001")
    result = subprocess.run(
        [cmd_7z, "x", first_part, f"-o{staging}", "-y"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"  ✗ Extraction failed: {result.stderr.decode()[:300]}")
        return 0

    count = 0
    for root, _, files in os.walk(staging):
        for fname in files:
            if fname.lower().endswith(".mp4"):
                src = os.path.join(root, fname)
                dst = os.path.join(dest_dir, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                count += 1
                if count >= num_videos:
                    break
        if count >= num_videos:
            break

    shutil.rmtree(staging, ignore_errors=True)
    print(f"  ✓ GenBuster-200K: {count} videos → {dest_dir}")
    return count


# ---------------------------------------------------------------------------
# Fake source 2: synth-vid-detect  (public, 7 diffusion models, ~10K videos)
# ---------------------------------------------------------------------------

def download_synth_vid_detect(dest_dir, num_videos, hf_token=None):
    """
    Download from ductai199x/synth-vid-detect (public, no token needed).
    ~10K videos: CogVideo, LumaAI, Pika, Sora, SVD, VideoCrafter, VideoCrafter-v2.
    Structure: train/fake/<model>/<video>.mp4
    """
    from huggingface_hub import HfFileSystem, hf_hub_download
    from tqdm import tqdm

    _section(f"Downloading {num_videos} FAKE Videos — synth-vid-detect")
    print("  Models: CogVideo, LumaAI, Pika, Sora, SVD, VideoCrafter, VideoCrafter-v2")
    print("  Access: Public — no token required")
    os.makedirs(dest_dir, exist_ok=True)

    REPO_ID = "ductai199x/synth-vid-detect"
    MODELS = ["cogvid", "luma", "pika", "sora", "svd", "videocrafter", "videocrafter_v2"]

    try:
        fs = HfFileSystem(token=hf_token)
        all_files = []
        for model in MODELS:
            model_path = f"datasets/{REPO_ID}/train/fake/{model}"
            try:
                files = fs.ls(model_path, detail=False)
                mp4s = [f for f in files if f.lower().endswith(".mp4")]
                all_files.extend(mp4s)
                print(f"    {model:20s}: {len(mp4s)} videos")
            except Exception as e:
                print(f"    {model:20s}: skipped ({e})")
    except Exception as e:
        print(f"  Error accessing synth-vid-detect: {e}")
        return 0

    random.shuffle(all_files)
    all_files = all_files[:num_videos]
    print(f"\n  Downloading {len(all_files)} videos...")

    downloaded, failed = 0, 0
    prefix = f"datasets/{REPO_ID}/"
    for hf_path in tqdm(all_files, desc="  synth-vid-detect"):
        filename = hf_path[len(prefix):] if hf_path.startswith(prefix) else hf_path
        model_name = filename.split("/")[-2] if "/" in filename else "fake"
        flat_name = f"{model_name}_{os.path.basename(filename)}"
        dest_path = os.path.join(dest_dir, flat_name)
        if os.path.exists(dest_path):
            downloaded += 1
            continue
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type="dataset",
                token=hf_token, cache_dir="/tmp/hf_cache",
            )
            shutil.copy2(cached, dest_path)
            downloaded += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n  Warning: {os.path.basename(filename)}: {e}")

    print(f"  ✓ synth-vid-detect: {downloaded} videos → {dest_dir}" +
          (f" ({failed} failed)" if failed else ""))
    return downloaded


# ---------------------------------------------------------------------------
# Fake source 3: OpenSora zips  (public fallback, ~2-10K total)
# ---------------------------------------------------------------------------

def download_opensora_zips(dest_dir, num_videos, hf_token=None):
    """Download OpenSora-generated videos from 34data repos (public)."""
    from huggingface_hub import hf_hub_download

    _section(f"Downloading {num_videos} FAKE Videos — OpenSora zips")
    print("  Models: OpenSora v1, OpenSora v1.2")
    print("  Access: Public — no token required")
    os.makedirs(dest_dir, exist_ok=True)

    SOURCES = [
        {"repo_id": "34data/v14-fake-opensora",  "filename": "data_001.zip", "label": "opensora_v1"},
        {"repo_id": "34data/v14-fake-opensora12", "filename": "data_001.zip", "label": "opensora_v12"},
    ]
    per = num_videos // len(SOURCES) + 1
    total = 0
    for cfg in SOURCES:
        need = min(per, num_videos - total)
        if need <= 0:
            break
        print(f"\n  Fetching {cfg['label']}...")
        try:
            zip_path = hf_hub_download(
                repo_id=cfg["repo_id"], filename=cfg["filename"],
                repo_type="dataset", token=hf_token, cache_dir="/tmp/hf_cache",
            )
        except Exception as e:
            print(f"  ✗ {cfg['label']}: {e}")
            continue
        extracted = _extract_zip_mp4s(zip_path, dest_dir, need, prefix=cfg["label"])
        print(f"  ✓ {cfg['label']}: {extracted} videos")
        total += extracted
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def _extract_zip_mp4s(zip_path, dest_dir, max_count, prefix=""):
    from tqdm import tqdm
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            mp4s = [n for n in zf.namelist() if n.lower().endswith(".mp4")]
            for name in tqdm(mp4s[:max_count], desc=f"  Extracting {prefix}"):
                flat = f"{prefix}_{os.path.basename(name)}" if prefix else os.path.basename(name)
                dst = os.path.join(dest_dir, flat)
                if not os.path.exists(dst):
                    with zf.open(name) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                extracted += 1
    except zipfile.BadZipFile:
        try:
            with tarfile.open(zip_path) as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".mp4")]
                for m in members[:max_count]:
                    m.name = os.path.basename(m.name)
                    tf.extract(m, path=dest_dir)
                    extracted += 1
        except Exception as e:
            print(f"  Error: {e}")
    return extracted


def _generate_dummy_videos(dest_dir, count, start_index=0):
    if count <= 0:
        return
    os.makedirs(dest_dir, exist_ok=True)
    ok = 0
    for i in range(count):
        dst = os.path.join(dest_dir, f"dummy_{start_index + i:05d}.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=c=black:s=256x256:r=16:d=2", "-c:v", "libx264", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            ok += 1
    print(f"  Generated {ok}/{count} dummy placeholders")


def download_fake_videos(dest_dir, num_videos, hf_token, source):
    total = 0
    if source in ("genbuster", "all"):
        total += download_genbuster(dest_dir, num_videos - total, hf_token)
    if source in ("public", "synth", "all") and total < num_videos:
        total += download_synth_vid_detect(dest_dir, num_videos - total, hf_token)
    if source in ("public", "opensora", "all") and total < num_videos:
        total += download_opensora_zips(dest_dir, num_videos - total, hf_token)
    if total < num_videos:
        _generate_dummy_videos(dest_dir, num_videos - total, start_index=total)
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RA-Det dataset downloader. Real: Kinetics-400. Fake: diffusion-model videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
fake-source options:
  genbuster  200K videos (Sora/WanX/Kling/CogVideo/SVD/Pika) — needs p7zip + token + ~108GB
             Accept: https://huggingface.co/datasets/l8cv/GenBuster-200K
  public     ~10K total, no gating (synth-vid-detect + OpenSora zips)
  synth      Only synth-vid-detect (~10K, 7 models, public)
  opensora   Only OpenSora zips (~2-10K, public)
  all        All sources combined

Recommended for VideoMAE Large (0.3B):
  python3 scripts/download_subset.py --count 50000 --hf-token TOKEN --fake-source genbuster
        """
    )
    parser.add_argument("--count", type=int, default=1000,
                        help="Videos per class (real and fake). Default: 1000")
    parser.add_argument("--hf-token", dest="hf_token", type=str, required=True,
                        help="HuggingFace token. Required for Kinetics-400 + GenBuster-200K.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--fake-source", dest="fake_source",
                        choices=["genbuster", "public", "synth", "opensora", "all"],
                        default="public",
                        help="Fake video source. Default: public")
    parser.add_argument("--skip-real", action="store_true", help="Skip real video download.")
    parser.add_argument("--skip-fake", action="store_true", help="Skip fake video download.")

    args = parser.parse_args()
    install_dependencies()
    random.seed(args.seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_dest = os.path.join(base_dir, "data", "video_dataset", "train", "0_real")
    fake_dest = os.path.join(base_dir, "data", "video_dataset", "train", "1_fake")

    print(f"\nRA-Det Dataset Downloader")
    print(f"  Real  → {real_dest}")
    print(f"  Fake  → {fake_dest}")
    print(f"  Count : {args.count:,} each")
    print(f"  Fake source: {args.fake_source}")

    real_count = fake_count = 0
    if not args.skip_real:
        real_count = download_kinetics400_real(real_dest, args.count, args.hf_token)
    if not args.skip_fake:
        fake_count = download_fake_videos(fake_dest, args.count, args.hf_token, args.fake_source)

    _section("Download Complete")
    print(f"  Real videos  : {real_count:,}")
    print(f"  Fake videos  : {fake_count:,}")


if __name__ == "__main__":
    main()
