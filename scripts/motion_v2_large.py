import csv
from pathlib import Path
from datasets import load_dataset, Video
from tqdm import tqdm

PARQUET_DIR = Path("motion_v2_large/data")
OUTPUT_DIR = Path("videos")
METADATA_CSV = Path("metadata.csv")

OUTPUT_DIR.mkdir(exist_ok=True)

parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files.")

CSV_FIELDS = [
    "filename",
    "generator",
    "prompt_id",
    "prompt",
    "subcategory",
    "video_col",
    "parquet_file",
    "row_index",
    "weighted_score_Aesthetic",
    "weighted_score_Coherence",
    "weighted_score_Prompt_Adherence",
    "num_annotations_Aesthetic",
    "num_annotations_Coherence",
    "num_annotations_Prompt_Adherence",
]

# Open CSV in append mode so re-runs don't wipe existing rows;
# write header only when the file is new.
csv_is_new = not METADATA_CSV.exists()
csv_file = open(METADATA_CSV, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
if csv_is_new:
    writer.writeheader()

total = 0

for parquet_file in parquet_files:
    print(f"\nProcessing {parquet_file.name}")

    ds = load_dataset(
        "parquet",
        data_files=str(parquet_file),
        split="train",
    )
    ds = ds.cast_column("video1", Video(decode=False))
    ds = ds.cast_column("video2", Video(decode=False))

    for idx, sample in enumerate(tqdm(ds)):

        # Each row pairs (video1, model1) and (video2, model2)
        pairs = [
            ("video1", sample.get("model1") or "unknown"),
            ("video2", sample.get("model2") or "unknown"),
        ]

        for col, generator in pairs:
            video = sample.get(col)

            if video is None or video.get("bytes") is None:
                continue

            # Normalise generator name to a safe directory component
            # e.g. "veo3_fast" → "veo3_fast", "luma_ray2" → "luma_ray2"
            safe_gen = generator.replace(" ", "_").replace("/", "-")

            # Create per-generator sub-directory
            gen_dir = OUTPUT_DIR / safe_gen
            gen_dir.mkdir(exist_ok=True)

            # Build filename
            original_path = video.get("path") or ""
            if original_path:
                basename = Path(original_path).name
                filename = f"{parquet_file.stem}_{idx:06d}_{basename}"
            else:
                filename = f"{parquet_file.stem}_{idx:06d}_{col}.mp4"

            output_path = gen_dir / filename

            # Skip if already extracted
            if not output_path.exists():
                with open(output_path, "wb") as f:
                    f.write(video["bytes"])
                total += 1

            # --- Scores for this specific video ---
            # col is "video1" → suffix "1"; "video2" → suffix "2"
            suffix = "1" if col == "video1" else "2"

            row = {
                "filename": str(output_path),
                "generator": generator,
                "prompt_id": sample.get("prompt_id", ""),
                "prompt": sample.get("prompt", ""),
                "subcategory": sample.get("subcategory", ""),
                "video_col": col,
                "parquet_file": parquet_file.name,
                "row_index": idx,
                "weighted_score_Aesthetic": sample.get(
                    f"weighted_results{suffix}_Aesthetic", ""
                ),
                "weighted_score_Coherence": sample.get(
                    f"weighted_results{suffix}_Coherence", ""
                ),
                "weighted_score_Prompt_Adherence": sample.get(
                    f"weighted_results{suffix}_Prompt_Adherence", ""
                ),
                "num_annotations_Aesthetic": sample.get("num_annotations_Aesthetic", ""),
                "num_annotations_Coherence": sample.get("num_annotations_Coherence", ""),
                "num_annotations_Prompt_Adherence": sample.get(
                    "num_annotations_Prompt_Adherence", ""
                ),
            }
            writer.writerow(row)

csv_file.close()

print(f"\n✅ Extracted {total} new videos.")
print(f"Videos saved in:  {OUTPUT_DIR.resolve()}")
print(f"Metadata saved to: {METADATA_CSV.resolve()}")
