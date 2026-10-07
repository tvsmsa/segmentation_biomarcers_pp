"""All-class extension of test_maples_models.py metrics and source reports."""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ml.biomarcers.config import Config
from .common import CLASS_IDS, CLASS_NAMES
from .models import forward_logits


def sample_confusions(predictions, targets):
    valid = targets != Config.IGNORE_INDEX
    batch = targets.shape[0]
    offsets = torch.arange(batch, device=targets.device)[:, None, None] * Config.NUM_CLASSES ** 2
    codes = offsets + targets * Config.NUM_CLASSES + predictions
    return torch.bincount(codes[valid], minlength=batch * Config.NUM_CLASSES ** 2).reshape(
        batch, Config.NUM_CLASSES, Config.NUM_CLASSES)


def counts_for_class(confusion, class_id):
    tp = int(confusion[class_id, class_id])
    return tp, int(confusion[:, class_id].sum()) - tp, int(confusion[class_id, :].sum()) - tp


def scores(tp, fp, fn, missing=0.0):
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else missing,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else missing,
            "precision": tp / (tp + fp) if tp + fp else missing,
            "recall": tp / (tp + fn) if tp + fn else missing}


def metric_frame(confusion):
    rows = []
    for class_id in CLASS_IDS:
        tp, fp, fn = counts_for_class(confusion, class_id)
        rows.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id],
                     "tp": tp, "fp": fp, "fn": fn, **scores(tp, fp, fn)})
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {"class_id": None, "class_name": "MEAN", "tp": None, "fp": None, "fn": None,
                             **{key: frame[key].mean() for key in ("iou", "dice", "precision", "recall")}}
    return frame


def source_frame(counts):
    rows = []
    for (dataset, source_id), confusion in sorted(counts.items()):
        marked = {c for c in CLASS_IDS if confusion[c, :].sum() > 0}
        predicted = {c for c in CLASS_IDS if confusion[:, c].sum() > 0}
        names = lambda ids: ", ".join(CLASS_NAMES[c] for c in sorted(ids))
        row = {"dataset": dataset, "source_id": source_id,
               "biomarkers_in_markup": names(marked), "biomarkers_predicted": names(predicted),
               "biomarkers_without_markup": names(predicted - marked)}
        for c in CLASS_IDS:
            for key, value in scores(*counts_for_class(confusion, c), missing=None).items():
                label = {"iou": "IoU", "dice": "Dice", "precision": "Precision", "recall": "Recall"}[key]
                row[f"{CLASS_NAMES[c]} {label}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


@torch.inference_mode()
def evaluate_loader(model, loader, device, *, per_source=False, check_time=None, amp=False):
    model.eval()
    total = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES), dtype=np.int64)
    by_source, offset = {}, 0
    records = loader.dataset.df[["dataset", "source_id"]].to_dict("records") if per_source else None
    for images, masks in tqdm(loader, desc="Evaluate", leave=False, unit="batch"):
        if check_time:
            check_time()
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = forward_logits(model, images, masks.shape[-2:])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits; disable --amp if enabled and verify the checkpoint")
        counts = sample_confusions(logits.argmax(1), masks).cpu().numpy()
        total += counts.sum(axis=0)
        if per_source:
            for index, confusion in enumerate(counts):
                row = records[offset + index]
                key = (row["dataset"], row["source_id"])
                by_source.setdefault(key, np.zeros_like(total))[:] += confusion
        offset += len(images)
    if offset != len(loader.dataset):
        raise ValueError("Evaluation loader must cover the complete dataset with drop_last=False")
    return total, by_source
