#!/usr/bin/env python3
"""
Model Reliability Test — evaluates a trained RA-Det checkpoint on a held-out,
UNSEEN video set and reports standard binary-classification reliability
metrics (confusion matrix, ROC/PR curves, AUC, precision/recall/F1, FPR/FNR,
per-generator breakdown).

Why this exists (vs. scripts/revalidate_checkpoints.sh):
  - revalidate_checkpoints.sh re-runs train.py's internal validation loop
    (same dataloader/eval code path as training) to calibrate/patch
    `optimal_threshold` into a checkpoint.
  - This script instead drives the checkpoint through the EXACT same code
    path production inference uses (imports preprocess_video/run_inference
    directly from inference_api.py) against a directory of real/fake videos.
    It never touches training code, so it also validates that inference_api.py
    itself (preprocessing, decision logic, multi-clip mode, thresholds) behaves
    correctly end-to-end — not just the raw model.

Expected data layout (matches data/test_validation/):
    <data-dir>/
        real/            # ground-truth REAL videos (label 0)
            *.mp4
            optional_subfolder_per_source/*.mp4
        fake/            # ground-truth FAKE videos (label 1)
            GROK/*.mp4
            Seedance/*.mp4
            klingai/*.mp4
            ...

Any immediate subfolder under real/ or fake/ is treated as a "generator" tag
for the per-generator breakdown (e.g. GROK, Seedance, klingai). Videos placed
directly in real/ or fake/ with no subfolder are tagged "unknown".

Usage:
    # Quick run against the bundled held-out set, using checkpoint_best.pt
    uv run python scripts/test_model_reliability.py

    # Point at a specific checkpoint / data dir, save a full report
    uv run python scripts/test_model_reliability.py \\
        --checkpoint ./checkpoints/dinov2_temporal_v1/checkpoint_epoch_18.pt \\
        --data-dir ./data/test_validation \\
        --output-dir ./eval_reports/epoch_18

    # Exercise multi-clip inference (whole-video coverage) instead of the
    # single 8-frame sample — mirrors /predict?multi_clip=true
    uv run python scripts/test_model_reliability.py --multi-clip --num-clips 3

    # Force every video through Layer 3 even if C2PA/anomaly would have
    # short-circuited it — mirrors /predict?disable_cascade=true
    uv run python scripts/test_model_reliability.py --disable-cascade

    # Skip Layer 1/2 computation entirely, pure Layer-3 ensemble read
    uv run python scripts/test_model_reliability.py --layer3-only

    # Explicitly turn a flag off (BooleanOptionalAction gives --no-<flag> too)
    uv run python scripts/test_model_reliability.py --no-multi-clip --no-disable-cascade

Outputs (written to --output-dir, default ./eval_reports/<timestamp>/):
    predictions.csv       — one row per video: path, generator, label, prob, prediction, correct
    metrics.json          — all computed metrics (machine-readable)
    report.txt            — human-readable summary (sklearn classification_report + extras)
    confusion_matrix.png
    roc_curve.png
    pr_curve.png
    misclassified.txt     — paths of every wrong prediction, for manual inspection
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI — parsed BEFORE importing inference_api, since inference_api reads
# CHECKPOINT_PATH / ANOMALY_CHECKPOINT_PATH / MODEL_NAME / etc. from the
# environment at module-import time (see inference_api.py top-level config
# block). Setting os.environ here lets --checkpoint etc. actually take effect.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RA-Det model reliability on unseen video data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=str, default=None,
                    help="Root dir containing real/ and fake/ subfolders of unseen videos. "
                         "Mutually exclusive with --real-dir/--fake-dir.")
    p.add_argument("--real-dir", type=str, default=None,
                    help="Directory containing ground-truth REAL videos (label 0). "
                         "Use this when real and fake data live in separate paths.")
    p.add_argument("--fake-dir", type=str, default=None,
                    help="Directory containing ground-truth FAKE videos (label 1). "
                         "Use this when real and fake data live in separate paths.")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to Layer-3 ensemble checkpoint (.pt). Defaults to inference_api's CHECKPOINT_PATH resolution.")
    p.add_argument("--anomaly-checkpoint", type=str, default=None,
                    help="Path to Layer-2 anomaly detector checkpoint. Used unless --layer3-only is set.")
    p.add_argument("--train-config", type=str, default=None,
                    help="Override TRAIN_CONFIG_NAME (defaults to dinov2_temporal_v1, or $TRAIN_CONFIG_NAME).")
    p.add_argument("--mode", type=str, default="classifier", choices=["classifier", "embedding", "hybrid"],
                    help="Decision strategy for Layer 3 (matches /predict's ?mode=).")
    p.add_argument("--threshold", type=float, default=None,
                    help="Override decision threshold. Defaults to the checkpoint's calibrated optimal_threshold.")
    # --multi-clip / --disable-cascade mirror inference_api.py's /predict boolean
    # query params (multi_clip, disable_cascade) exactly — same names, same
    # semantics, same defaults — so this script's flags map 1:1 onto what a
    # real API request would set. BooleanOptionalAction gives you an explicit
    # --no-multi-clip / --no-disable-cascade form too, instead of relying on
    # a flag's mere absence to mean "off" (useful when scripting many runs).
    p.add_argument("--multi-clip", action=argparse.BooleanOptionalAction, default=False,
                    help="[mirrors /predict?multi_clip=] Sample multiple temporal segments spanning "
                         "the whole video and aggregate Layer-3 scores, instead of one 8-frame clip.")
    p.add_argument("--num-clips", type=int, default=3,
                    help="Number of segments to use when --multi-clip is set.")
    p.add_argument("--clip-aggregation", type=str, default="mean", choices=["mean", "max", "median"],
                    help="How to combine per-clip scores when --multi-clip is set.")
    p.add_argument("--disable-cascade", action=argparse.BooleanOptionalAction, default=False,
                    help="[mirrors /predict?disable_cascade=] Compute Layer 1 (C2PA) and Layer 2 "
                         "(anomaly) as usual but ignore their short-circuit verdicts, forcing every "
                         "video through the full Layer-3 ensemble. Use this to measure Layer 3's own "
                         "reliability without an L1/L2 shortcut masking its result.")
    p.add_argument("--layer3-only", action=argparse.BooleanOptionalAction, default=False,
                    help="Skip Layer 1/2 entirely (no C2PA/anomaly computation at all) and run only "
                         "the Layer-3 ensemble. Different from --disable-cascade, which still computes "
                         "L1/L2 for reporting but never lets them short-circuit the verdict.")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps"],
                    help="Force a device. Defaults to CUDA if available, else CPU.")
    p.add_argument("--max-per-class", type=int, default=None,
                    help="Cap the number of videos evaluated per class (for a quick smoke test).")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where to write the report. Defaults to ./eval_reports/<timestamp>.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible subsampling (--max-per-class).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = str(Path(args.checkpoint).resolve())
    if args.anomaly_checkpoint:
        os.environ["ANOMALY_CHECKPOINT_PATH"] = str(Path(args.anomaly_checkpoint).resolve())
    if args.train_config:
        os.environ["TRAIN_CONFIG_NAME"] = args.train_config

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    print("Loading inference_api module (model construction code, not the HTTP server)...")
    import inference_api as api  # noqa: E402  (must come after env vars are set)
    import torch  # noqa: E402

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, roc_curve, precision_recall_curve,
    )
    import matplotlib
    matplotlib.use("Agg")  # headless — no display server needed
    import matplotlib.pyplot as plt

    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Resolve device + load models (reuses inference_api's exact loader, so
    # this test is guaranteed to construct the model identically to production)
    # -----------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {api.CHECKPOINT_PATH}")
    if not os.path.exists(api.CHECKPOINT_PATH):
        print(f"ERROR: checkpoint not found at {api.CHECKPOINT_PATH}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    api._load_models(device)
    print(f"Models loaded in {time.time() - t0:.1f}s "
          f"(checkpoint epoch={api._store.checkpoint_epoch}, "
          f"optimal_threshold={api._store.optimal_threshold:.4f})")

    active_threshold = args.threshold if args.threshold is not None else api._store.optimal_threshold

    # -----------------------------------------------------------------------
    # Discover unseen videos
    # -----------------------------------------------------------------------
    # Two modes for specifying data:
    #   1. --data-dir <root>          expects <root>/real/ and <root>/fake/
    #   2. --real-dir X --fake-dir Y  separate paths for each class
    if args.real_dir or args.fake_dir:
        if not args.real_dir or not args.fake_dir:
            print("ERROR: --real-dir and --fake-dir must both be provided together.", file=sys.stderr)
            sys.exit(1)
        real_dir = Path(args.real_dir).expanduser().resolve()
        fake_dir = Path(args.fake_dir).expanduser().resolve()
        data_dir_label = f"real={real_dir}, fake={fake_dir}"
    elif args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"
        data_dir_label = str(data_dir)
    else:
        # Default fallback
        data_dir = Path("./data/test_validation").resolve()
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"
        data_dir_label = str(data_dir)

    if not real_dir.is_dir():
        print(f"ERROR: real video directory not found: {real_dir}", file=sys.stderr)
        sys.exit(1)
    if not fake_dir.is_dir():
        print(f"ERROR: fake video directory not found: {fake_dir}", file=sys.stderr)
        sys.exit(1)

    samples = _discover_videos(real_dir, label=0) + _discover_videos(fake_dir, label=1)
    if not samples:
        print(f"ERROR: no video files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_per_class:
        by_label = {0: [], 1: []}
        for s in samples:
            by_label[s["label"]].append(s)
        samples = []
        for label, group in by_label.items():
            idx = np.random.permutation(len(group))[: args.max_per_class]
            samples.extend([group[i] for i in idx])

    n_real = sum(1 for s in samples if s["label"] == 0)
    n_fake = sum(1 for s in samples if s["label"] == 1)
    print(f"Evaluating {len(samples)} unseen videos ({n_real} real, {n_fake} fake) "
          f"mode={args.mode} threshold={active_threshold:.4f} "
          f"multi_clip={args.multi_clip} disable_cascade={args.disable_cascade} "
          f"layer3_only={args.layer3_only}")

    # -----------------------------------------------------------------------
    # Run inference over every video
    # -----------------------------------------------------------------------
    from tqdm import tqdm

    results = []
    errors = []
    t_eval_start = time.time()

    for sample in tqdm(samples, desc="Running inference", unit="video"):
        path = sample["path"]
        try:
            record = _predict_one(api, torch, path, args, active_threshold)
            record.update(label=sample["label"], generator=sample["generator"], path=path)
            results.append(record)
        except Exception as e:
            errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
            tqdm.write(f"  [SKIP] {path}: {e}")

    eval_time = time.time() - t_eval_start
    print(f"\nCompleted {len(results)}/{len(samples)} videos in {eval_time:.1f}s "
          f"({eval_time / max(len(results), 1):.2f}s/video). {len(errors)} errors.")

    if not results:
        print("ERROR: every video failed inference — nothing to report.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    y_true = np.array([r["label"] for r in results])
    y_prob = np.array([r["probability"] for r in results])
    y_pred = np.array([r["prediction_label"] for r in results])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "num_videos_evaluated": int(len(results)),
        "num_videos_failed": int(len(errors)),
        "num_real": int(n_real),
        "num_fake": int(n_fake),
        "threshold_used": float(active_threshold),
        "mode": args.mode,
        "multi_clip": bool(args.multi_clip),
        "num_clips": args.num_clips if args.multi_clip else None,
        "clip_aggregation": args.clip_aggregation if args.multi_clip else None,
        "disable_cascade": bool(args.disable_cascade),
        "layer3_only": bool(args.layer3_only),
        "eval_time_s": round(eval_time, 2),
        "avg_time_per_video_s": round(eval_time / max(len(results), 1), 3),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity_tpr": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_tnr": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        "confusion_matrix": {
            "true_negative_real_as_real": int(tn),
            "false_positive_real_as_fake": int(fp),
            "false_negative_fake_as_real": int(fn),
            "true_positive_fake_as_fake": int(tp),
        },
    }

    # AUC/AP need both classes present
    if len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision_pr_auc"] = float(average_precision_score(y_true, y_prob))
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        metrics["youdens_j_optimal_threshold"] = float(np.clip(roc_thresholds[best_idx], 0.0, 1.0))
        metrics["youdens_j_value"] = float(j_scores[best_idx])
    else:
        print("WARNING: only one class present in this run — AUC/PR-AUC/ROC skipped.")
        fpr = tpr = roc_thresholds = None

    # -----------------------------------------------------------------------
    # Per-generator breakdown
    # -----------------------------------------------------------------------
    generator_metrics = {}
    generators = sorted(set(r["generator"] for r in results))
    for gen in generators:
        gen_rows = [r for r in results if r["generator"] == gen]
        gt = np.array([r["label"] for r in gen_rows])
        pd_ = np.array([r["prediction_label"] for r in gen_rows])
        generator_metrics[gen] = {
            "count": len(gen_rows),
            "label": "fake" if gen_rows[0]["label"] == 1 else "real",
            "accuracy": float(accuracy_score(gt, pd_)),
            "num_correct": int((gt == pd_).sum()),
            "num_wrong": int((gt != pd_).sum()),
        }
    metrics["per_generator"] = generator_metrics

    # Breakdown of which layer actually produced each verdict — most useful
    # with --disable-cascade OFF (default), since that's when L1/L2 can
    # short-circuit and this tells you how often that happened.
    decided_by_counts = {}
    for r in results:
        decided_by_counts[r["decided_by"]] = decided_by_counts.get(r["decided_by"], 0) + 1
    metrics["decided_by_counts"] = decided_by_counts

    # -----------------------------------------------------------------------
    # Output dir
    # -----------------------------------------------------------------------
    out_dir = Path(args.output_dir) if args.output_dir else Path("eval_reports") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting report to {out_dir}/")

    # predictions.csv
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "generator", "true_label", "predicted_label",
                          "probability", "confidence", "correct"])
        for r in results:
            true_str = "fake" if r["label"] == 1 else "real"
            pred_str = "fake" if r["prediction_label"] == 1 else "real"
            writer.writerow([r["path"], r["generator"], true_str, pred_str,
                              round(r["probability"], 6), r["confidence"],
                              true_str == pred_str])

    # misclassified.txt
    misclassified = [r for r in results if r["label"] != r["prediction_label"]]
    with open(out_dir / "misclassified.txt", "w") as f:
        f.write(f"{len(misclassified)} / {len(results)} videos misclassified\n\n")
        for r in misclassified:
            true_str = "fake" if r["label"] == 1 else "real"
            pred_str = "fake" if r["prediction_label"] == 1 else "real"
            f.write(f"[{true_str} -> {pred_str}] prob={r['probability']:.4f} "
                    f"generator={r['generator']} path={r['path']}\n")

    if errors:
        with open(out_dir / "errors.txt", "w") as f:
            f.write(f"{len(errors)} videos failed to process\n\n")
            for e in errors:
                f.write(f"{e['path']}: {e['error']}\n")

    # metrics.json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # report.txt — human readable
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("RA-Det Model Reliability Report — Unseen Data Evaluation")
    report_lines.append("=" * 72)
    report_lines.append(f"Timestamp:            {datetime.now().isoformat()}")
    report_lines.append(f"Checkpoint:            {api.CHECKPOINT_PATH}")
    report_lines.append(f"Checkpoint epoch:      {api._store.checkpoint_epoch}")
    report_lines.append(f"Data dir:              {data_dir_label}")
    report_lines.append(f"Mode:                  {args.mode}")
    report_lines.append(f"Threshold used:        {active_threshold:.4f}")
    report_lines.append(f"Multi-clip:            {args.multi_clip} "
                         f"(num_clips={args.num_clips}, agg={args.clip_aggregation})" if args.multi_clip else "Multi-clip:            False")
    report_lines.append(f"Disable cascade:       {args.disable_cascade}")
    report_lines.append(f"Layer 3 only:          {args.layer3_only}")
    report_lines.append(f"Videos evaluated:      {len(results)} ({n_real} real, {n_fake} fake), {len(errors)} failed")
    report_lines.append(f"Decided-by breakdown:  {metrics['decided_by_counts']}")
    report_lines.append(f"Avg inference time:    {metrics['avg_time_per_video_s']:.3f}s/video")
    report_lines.append("-" * 72)
    report_lines.append("CONFUSION MATRIX")
    report_lines.append(f"                  Predicted Real   Predicted Fake")
    report_lines.append(f"  Actual Real     {tn:>14d}   {fp:>14d}")
    report_lines.append(f"  Actual Fake     {fn:>14d}   {tp:>14d}")
    report_lines.append("-" * 72)
    report_lines.append("CORE METRICS")
    report_lines.append(f"  Accuracy:              {metrics['accuracy']:.4f}")
    report_lines.append(f"  Precision (fake):      {metrics['precision']:.4f}")
    report_lines.append(f"  Recall/Sensitivity:    {metrics['recall_sensitivity_tpr']:.4f}")
    report_lines.append(f"  Specificity:           {metrics['specificity_tnr']:.4f}" if metrics['specificity_tnr'] is not None else "  Specificity:           N/A")
    report_lines.append(f"  F1 Score:              {metrics['f1_score']:.4f}")
    report_lines.append(f"  False Positive Rate:   {metrics['false_positive_rate']:.4f}" if metrics['false_positive_rate'] is not None else "  False Positive Rate:   N/A")
    report_lines.append(f"  False Negative Rate:   {metrics['false_negative_rate']:.4f}" if metrics['false_negative_rate'] is not None else "  False Negative Rate:   N/A")
    if "roc_auc" in metrics:
        report_lines.append(f"  ROC AUC:               {metrics['roc_auc']:.4f}")
        report_lines.append(f"  PR AUC (avg prec.):    {metrics['average_precision_pr_auc']:.4f}")
        report_lines.append(f"  Youden's J threshold:  {metrics['youdens_j_optimal_threshold']:.4f} (J={metrics['youdens_j_value']:.4f})")
    report_lines.append("-" * 72)
    report_lines.append("SKLEARN CLASSIFICATION REPORT")
    report_lines.append(classification_report(y_true, y_pred, target_names=["real", "fake"], zero_division=0))
    report_lines.append("-" * 72)
    report_lines.append("PER-GENERATOR BREAKDOWN")
    for gen, gm in sorted(generator_metrics.items(), key=lambda kv: (-kv[1]["num_wrong"], kv[0])):
        report_lines.append(f"  {gen:<20s} label={gm['label']:<5s} n={gm['count']:<4d} "
                             f"accuracy={gm['accuracy']:.4f} wrong={gm['num_wrong']}")
    report_lines.append("=" * 72)
    report_text = "\n".join(report_lines)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report_text)

    print()
    print(report_text)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    _plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    if fpr is not None:
        _plot_roc_curve(fpr, tpr, metrics["roc_auc"], out_dir / "roc_curve.png")
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        _plot_pr_curve(precision_curve, recall_curve, metrics["average_precision_pr_auc"], out_dir / "pr_curve.png")

    print(f"\nFull report written to: {out_dir}/")


def _discover_videos(root: Path, label: int) -> list:
    """
    Recursively find video files under `root`. The immediate child directory
    name (if any) between `root` and the file is used as the "generator" tag
    for per-generator metrics; files directly under `root` are tagged "unknown".
    """
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    samples = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in video_exts:
            continue
        rel_parts = path.relative_to(root).parts
        generator = rel_parts[0] if len(rel_parts) > 1 else "unknown"
        samples.append({"path": str(path), "label": label, "generator": generator})
    return samples


def _predict_one(api, torch, video_path: str, args, active_threshold: float) -> dict:
    """
    Run one video through inference_api's own functions, replicating the
    exact branch structure of the /predict route (see inference_api.py):

      1. Layer 1 (C2PA provenance) — always computed. Short-circuits the
         verdict unless args.disable_cascade or args.layer3_only.
      2. Layer 2 (anomaly detector) — always computed (unless layer3_only).
         Short-circuits the verdict unless args.disable_cascade or
         args.layer3_only.
      3. Layer 3 (RA-Det ensemble) — runs whenever L1/L2 didn't short-circuit,
         or whenever disable_cascade/layer3_only forces it to.

    --disable-cascade mirrors /predict?disable_cascade=true: L1/L2 still run
    (so their verdicts show up in reporting) but can never override Layer 3.
    --layer3-only skips L1/L2's computation entirely, for a pure ensemble
    speed/accuracy read with no provenance/anomaly overhead.
    """
    prov = None
    if not args.layer3_only:
        prov = api.check_provenance(video_path)
        if not args.disable_cascade and prov["verdict"] != "unknown":
            is_fake = prov["verdict"] == "ai_generated"
            return {"probability": 1.0 if is_fake else 0.0, "prediction_label": int(is_fake),
                    "confidence": "very_high", "decided_by": "layer_1_provenance"}

    if args.multi_clip:
        clip_tensors, _meta = api.preprocess_video_multi_clip(video_path, num_clips=args.num_clips)
        video_tensor = clip_tensors[0]
    else:
        video_tensor = api.preprocess_video(video_path)
        clip_tensors = None

    if not args.layer3_only:
        anomaly = api._store.anomaly_detector.score(video_tensor) if api._store.anomaly_detector else {"verdict": "uncertain"}
        if not args.disable_cascade and anomaly["verdict"] != "uncertain":
            is_fake = anomaly["verdict"] == "fake"
            return {"probability": anomaly["anomaly_score"], "prediction_label": int(is_fake),
                    "confidence": anomaly["confidence"], "decided_by": "layer_2_anomaly"}

    if args.multi_clip:
        per_clip_raw = [api.run_inference(ct) for ct in clip_tensors]
        raw = api._aggregate_clip_signals(per_clip_raw, aggregation=args.clip_aggregation)
    else:
        raw = api.run_inference(video_tensor)

    decision = api._make_decision(raw, args.mode, threshold=active_threshold)
    return {
        "probability": decision["probability"],
        "prediction_label": int(decision["prediction"] == "fake"),
        "confidence": decision["confidence"],
        "decided_by": "layer_3_radet_ensemble",
    }


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["real", "fake"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100.0 * count / total if total else 0.0
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                    color="white" if count > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(fpr, tpr, auc, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(precision, recall, ap, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
