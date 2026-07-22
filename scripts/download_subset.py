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
import time
from pathlib import Path

# Force unbuffered output so you can see progress in log files immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", ".hf_cache")

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
            mp4_files = fs.glob(f"{class_path}/*/*.mp4")
            sample = random.sample(mp4_files, min(per_class, len(mp4_files)))
            all_files.extend(sample)
        except Exception:
            continue
        if len(all_files) >= num_videos * 2:
            break

    random.shuffle(all_files)
    random.shuffle(all_files)
    all_files = all_files[:num_videos]
    print(f"  Downloading {len(all_files)} videos (Parallel)...")

    prefix = f"datasets/{REPO_ID}/"
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def download_one(hf_path):
        filename = hf_path[len(prefix):] if hf_path.startswith(prefix) else hf_path
        dest_path = os.path.join(dest_dir, os.path.basename(filename))
        if os.path.exists(dest_path):
            return True, None
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type="dataset",
                token=hf_token, cache_dir=HF_CACHE_DIR,
            )
            shutil.copy2(cached, dest_path)
            return True, None
        except Exception as e:
            return False, str(e)

    downloaded = 0
    failed = 0
    errors = []

    # Using 24 parallel threads to mask connection latency for 50,000 small files
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(download_one, path): path for path in all_files}
        with tqdm(total=len(all_files), desc="  Kinetics-400") as pbar:
            for future in as_completed(futures):
                success, err_msg = future.result()
                if success:
                    downloaded += 1
                else:
                    failed += 1
                    if len(errors) < 3:
                        errors.append(f"{os.path.basename(futures[future])}: {err_msg}")
                pbar.update(1)

    if errors:
        print("\n  Sample errors during download:")
        for err in errors:
            print(f"    - {err}")

    print(f"\n  ✓ Real: {downloaded} videos → {dest_dir}" +
          (f" ({failed} failed)" if failed else ""))
    return downloaded


# ---------------------------------------------------------------------------
# Fake source 1: GenBuster-200K  (200K diffusion-generated videos, auto-gated)
# ---------------------------------------------------------------------------

def _download_hf_with_progress(repo_id, filename, repo_type, token, dest_path, part_label):
    """
    Download a single HuggingFace file with a real-time tqdm progress bar
    showing speed, size transferred, and ETA.
    Falls back to silent hf_hub_download on older huggingface_hub versions.
    """
    try:
        import requests
        from tqdm import tqdm
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        t0 = time.time()
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            total_mb = f"{total/1024**2:.0f} MB" if total else "? MB"
            print(f"  ↓ {part_label}  ({total_mb})", flush=True)
            with tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"    {part_label[:28]:28s}",
                dynamic_ncols=True, leave=True,
            ) as bar:
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        elapsed = time.time() - t0
        size_mb = os.path.getsize(dest_path) / 1024**2
        print(f"    ✓ {part_label} done — {size_mb:.0f} MB in {elapsed:.0f}s ({size_mb/elapsed:.1f} MB/s)",
              flush=True)
        return True
    except Exception as e:
        # Fallback: silent hf_hub_download (works even without requests)
        print(f"  ↓ {part_label} (progress unavailable: {e}) — downloading silently...", flush=True)
        from huggingface_hub import hf_hub_download
        t0 = time.time()
        cached = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type,
            token=token, cache_dir=HF_CACHE_DIR,
        )
        shutil.copy2(cached, dest_path)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(dest_path) / 1024**2
        print(f"    ✓ {part_label} done — {size_mb:.0f} MB in {elapsed:.0f}s", flush=True)
        return True


def download_genbuster(dest_dir, num_videos, hf_token):
    """
    Download from l8cv/GenBuster-200K.
    200K videos: Sora, WanX, Kling, CogVideo, SVD, Pika, ModelScope, VideoCrafter.
    Stored as 27x 4GB .7z parts. Requires p7zip.
    Accept terms: https://huggingface.co/datasets/l8cv/GenBuster-200K
    """
    max_limit = num_videos if (num_videos is not None and num_videos > 0) else None
    display_count = f"{max_limit:,}" if max_limit else "ALL (~200,000)"
    _section(f"Downloading {display_count} FAKE Videos — GenBuster-200K")
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

    overall_t0 = time.time()
    print(f"\n  Downloading {NUM_PARTS} archive parts (~108 GB total)...")
    print(f"  Each part ≈ 4 GB — expect 3–15 min/part depending on network speed.\n")

    for i in range(1, NUM_PARTS + 1):
        part_name = f"GenBuster-200K.7z.{i:03d}"
        part_path = os.path.join(staging, part_name)

        if os.path.exists(part_path) and os.path.getsize(part_path) > 100_000:
            size_mb = os.path.getsize(part_path) / 1024**2
            print(f"  ✓ [{i:02d}/{NUM_PARTS}] {part_name} already cached ({size_mb:.0f} MB)",
                  flush=True)
            continue

        elapsed_total = time.time() - overall_t0
        parts_done = i - 1
        eta_str = ""
        if parts_done > 0:
            avg_per_part = elapsed_total / parts_done
            remaining = (NUM_PARTS - parts_done) * avg_per_part
            eta_str = f"  ~{remaining/60:.0f} min remaining"
        print(f"  [{i:02d}/{NUM_PARTS}]{eta_str}", flush=True)

        try:
            _download_hf_with_progress(
                repo_id=REPO_ID, filename=part_name, repo_type="dataset",
                token=hf_token, dest_path=part_path, part_label=part_name,
            )
        except Exception as e:
            print(f"  ✗ {part_name}: {e}", flush=True)
            if i == 1:
                print("  Cannot extract without part 1. Skipping.")
                return 0

    total_dl_time = time.time() - overall_t0
    print(f"\n  ✓ All {NUM_PARTS} parts downloaded in {total_dl_time/60:.1f} min")
    print(f"\n  Extracting with {cmd_7z} (this may take 10–30 min for 200K videos)...",
          flush=True)
    first_part = os.path.join(staging, "GenBuster-200K.7z.001")
    result = subprocess.run(
        [cmd_7z, "x", first_part, f"-o{staging}", "-y"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"  ✗ Extraction failed: {result.stderr.decode()[:300]}")
        return 0

    print("  Organising videos...", flush=True)
    count = 0
    for root, _, files in os.walk(staging):
        for fname in files:
            if fname.lower().endswith(".mp4"):
                src = os.path.join(root, fname)
                rel_dir = os.path.relpath(root, staging)
                # Preserve generator subfolders (e.g. Sora, Kling, Pika) under dest_dir.
                # Strip top-level archive wrapper folder if present (e.g. GenBuster-200K/).
                path_parts = Path(rel_dir).parts
                if len(path_parts) > 0 and path_parts[0].lower().startswith("genbuster"):
                    rel_dir = os.path.join(*path_parts[1:]) if len(path_parts) > 1 else "."

                out_dir = os.path.normpath(os.path.join(dest_dir, rel_dir))
                os.makedirs(out_dir, exist_ok=True)
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                count += 1
                if count % 5000 == 0:
                    print(f"    Organised {count:,} videos so far...", flush=True)
                if max_limit and count >= max_limit:
                    break
        if max_limit and count >= max_limit:
            break

    shutil.rmtree(staging, ignore_errors=True)
    print(f"  ✓ GenBuster-200K: {count:,} videos → {dest_dir}")
    return count


# ---------------------------------------------------------------------------
# Fake source 1.5: GenBuster-200K-mini
# ---------------------------------------------------------------------------

def download_genbuster_mini(dest_dir, num_videos, hf_token):
    from huggingface_hub import hf_hub_download
    
    _section(f"Downloading {num_videos} FAKE Videos — GenBuster-200K-mini")
    print("  Models: Sora, WanX, Kling, CogVideo, SVD, Pika, ModelScope, VideoCrafter")
    os.makedirs(dest_dir, exist_ok=True)
    
    REPO_ID = "l8cv/GenBuster-200K-mini"
    print("\n  Fetching GenBuster-200K-mini.zip (~5.4 GB)...")
    try:
        zip_path = hf_hub_download(
            repo_id=REPO_ID, filename="GenBuster-200K-mini.zip",
            repo_type="dataset", token=hf_token, cache_dir=HF_CACHE_DIR,
        )
    except Exception as e:
        print(f"  ✗ GenBuster-200K-mini download failed: {e}")
        return 0
        
    extracted = _extract_zip_mp4s(zip_path, dest_dir, num_videos, prefix="genbuster")
    print(f"  ✓ GenBuster-200K-mini: {extracted} videos → {dest_dir}")
    return extracted


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
        model_name = filename.split("/")[-2] if "/" in filename else "synth_unknown"
        
        out_dir = os.path.join(dest_dir, "synth", model_name)
        os.makedirs(out_dir, exist_ok=True)
        dest_path = os.path.join(out_dir, os.path.basename(filename))
        
        if os.path.exists(dest_path):
            downloaded += 1
            continue
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=filename, repo_type="dataset",
                token=hf_token, cache_dir=HF_CACHE_DIR,
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
                repo_type="dataset", token=hf_token, cache_dir=HF_CACHE_DIR,
            )
        except Exception as e:
            print(f"  ✗ {cfg['label']}: {e}")
            continue
        
        prefix = f"opensora/{cfg['label'].split('_')[1]}" if "_" in cfg["label"] else "opensora"
        extracted = _extract_zip_mp4s(zip_path, dest_dir, need, prefix=prefix)
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
                rel_dir = os.path.dirname(name)
                out_dir = os.path.join(dest_dir, prefix, rel_dir) if prefix else os.path.join(dest_dir, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                
                dst = os.path.join(out_dir, os.path.basename(name))
                if not os.path.exists(dst):
                    with zf.open(name) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                extracted += 1
    except zipfile.BadZipFile:
        try:
            with tarfile.open(zip_path) as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".mp4")]
                for m in members[:max_count]:
                    rel_dir = os.path.dirname(m.name)
                    out_dir = os.path.join(dest_dir, prefix, rel_dir) if prefix else os.path.join(dest_dir, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    m.name = os.path.basename(m.name)
                    tf.extract(m, path=out_dir)
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
    max_limit = None if (num_videos is not None and num_videos <= 0) else num_videos

    if source in ("genbuster", "all"):
        needed = (max_limit - total) if max_limit is not None else None
        total += download_genbuster(dest_dir, needed, hf_token)
    if source in ("genbuster-mini", "all") and (max_limit is None or total < max_limit):
        needed = (max_limit - total) if max_limit is not None else None
        total += download_genbuster_mini(dest_dir, needed, hf_token)
    if source in ("public", "synth", "all") and (max_limit is None or total < max_limit):
        needed = (max_limit - total) if max_limit is not None else None
        total += download_synth_vid_detect(dest_dir, needed, hf_token)
    if source in ("public", "opensora", "all") and (max_limit is None or total < max_limit):
        needed = (max_limit - total) if max_limit is not None else None
        total += download_opensora_zips(dest_dir, needed, hf_token)
    if max_limit is not None and total < max_limit:
        _generate_dummy_videos(dest_dir, max_limit - total, start_index=total)
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
  genbuster       ALL 200K videos (Sora/WanX/Kling/CogVideo/SVD/Pika) — needs p7zip + token + ~108GB
             Accept: https://huggingface.co/datasets/l8cv/GenBuster-200K
  genbuster-mini  ~10K subset of GenBuster (5.4GB zip). High quality, fast.
  public     ~10K total, no gating (synth-vid-detect + OpenSora zips)
  synth      Only synth-vid-detect (~10K, 7 models, public)
  opensora   Only OpenSora zips (~2-10K, public)
  all        All sources combined

Download ALL 200K GenBuster videos:
  python3 scripts/download_subset.py --fake-source genbuster --count 0 --hf-token YOUR_TOKEN
        """
    )
    parser.add_argument("--count", type=int, default=1000,
                        help="Videos per class. Pass 0 or -1 to download ALL available videos. Default: 1000")
    parser.add_argument("--hf-token", dest="hf_token", type=str, required=True,
                        help="HuggingFace token. Required for Kinetics-400 + GenBuster-200K.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--fake-source", dest="fake_source",
                        choices=["genbuster", "genbuster-mini", "public", "synth", "opensora", "all"],
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

    count_str = f"{args.count:,}" if args.count > 0 else "ALL"
    print(f"\nRA-Det Dataset Downloader")
    print(f"  Real  → {real_dest}")
    print(f"  Fake  → {fake_dest}")
    print(f"  Count : {count_str} per source")
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
