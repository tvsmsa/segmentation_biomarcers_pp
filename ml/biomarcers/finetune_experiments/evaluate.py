"""Evaluate available best checkpoints, with separate and pooled test metrics."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from ml.biomarcers.config import Config
from .common import EXPERIMENTS, check_prepared, file_digest, read_split, write_csv, write_json
from .models import checkpoint_data, load_model
from .scoring import evaluate_loader, metric_frame, source_frame
from .training import make_loader


def attach_metadata(frame, metadata):
    frame = frame.copy()
    for key, value in reversed(list(metadata.items())):
        frame.insert(0, key, value)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Folder containing run.json and experiment_1/2/3")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true", help="Optional; default FP32 matches the original tester")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    manifest = check_prepared(args.prepared_dir)
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    if run["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Evaluation dataset/splits differ from the training run")
    device, summaries, progress = torch.device(args.device), [], []
    results = args.run_dir / "results"
    for experiment, spec in EXPERIMENTS.items():
        exp_dir = args.run_dir / f"experiment_{experiment}"
        best = exp_dir / "best.pth"
        if not best.exists():
            progress.append({"experiment": experiment, "status": "no_validated_checkpoint", "epoch": None})
            print(f"Experiment {experiment}: no best checkpoint yet; skipped", flush=True)
            continue
        config = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
        ck = checkpoint_data(best)
        if ck["signature"] != config["signature"]:
            raise ValueError(f"Checkpoint/config mismatch: {best}")
        best_epoch = ck["epoch"]
        del ck
        status_file = exp_dir / "status.json"
        status = json.loads(status_file.read_text(encoding="utf-8"))["status"] if status_file.exists() else "incomplete"
        progress.append({"experiment": experiment, "status": status, "epoch": best_epoch})
        metadata = {"model": f"{run['architecture']}_{run['loss']}", "group": run["group"],
                    "initial_checkpoint": run["mapping"][str(experiment)], "experiment": experiment,
                    "best_epoch": best_epoch, "training_status": status,
                    "smoke_run": bool(run["smoke_patches"])}
        model = load_model(best, run["architecture"]).to(device)
        combined = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES), dtype=np.int64)
        image_frames, patch_count, source_count = [], 0, 0
        output = results / f"experiment_{experiment}"
        for name in spec["test"]:
            frame = read_split(args.prepared_dir, name)
            if run["smoke_patches"]:
                frame = frame.head(run["smoke_patches"])
            loader = make_loader(frame, args.batch_size, args.num_workers)
            confusion, sources = evaluate_loader(model, loader, device, per_source=True, amp=args.amp)
            combined += confusion
            patch_count += len(frame)
            source_count += len(sources)
            metrics = attach_metadata(metric_frame(confusion), {**metadata, "test_set": name,
                                      "patches": len(frame), "source_images": len(sources)})
            write_csv(output / f"{name}_metrics.csv", metrics)
            summaries.append(metrics)
            image_frames.append(attach_metadata(source_frame(sources), {**metadata, "test_set": name}))
        metrics = attach_metadata(metric_frame(combined), {**metadata, "test_set": "+".join(spec["test"]),
                                  "patches": patch_count, "source_images": source_count})
        write_csv(output / "combined_metrics.csv", metrics)
        summaries.append(metrics)
        images = pd.concat(image_frames, ignore_index=True).rename(columns={
            "model": "Модель", "source_id": "ID снимка", "biomarkers_in_markup": "Биомаркеры в маске",
            "biomarkers_predicted": "Биомаркеры, выделенные моделью",
            "biomarkers_without_markup": "Биомаркеры, которых нет в маске"})
        write_csv(output / "biomarkers_by_image.csv", images, sep=";", encoding="utf-8-sig")
        write_json(output / "evaluation.json", {"checkpoint_sha256": file_digest(best),
                   "data_fingerprint": manifest["fingerprint"], "amp": args.amp,
                   "class_ids": list(range(1, 15)), "ignore_index": 255,
                   "test_sets": list(spec["test"]), "training_status": status})
        # Keep a useful summary even if a later experiment is interrupted.
        write_csv(results / "summary.csv", pd.concat(summaries, ignore_index=True))
        print(metrics[["experiment", "test_set", "class_name", "dice", "iou"]].to_string(index=False), flush=True)
        del model, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(results / "progress.csv", pd.DataFrame(progress))
    if not summaries:
        print("No validated models available yet. Resume training first.", flush=True)
    else:
        print(f"Metrics: {(results / 'summary.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
