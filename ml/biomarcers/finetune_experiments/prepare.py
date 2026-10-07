"""Build source-grouped 70/30 splits and resolve paths on the current OS."""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import CLASS_NAMES, digest, file_digest, validate_partition, write_csv, write_json

SPECS = {
    "maples": ("test_patches.csv", "images", "masks"),
    "idrid": ("idrid_dataset.csv", "image_patches", "mask_patches"),
}


def filename(value: str) -> str:
    return PureWindowsPath(str(value)).name


def infer_source(dataset: str, name: str) -> str:
    pattern = r"(IDRiD_\d+)_\d+_\d+\.npy" if dataset == "idrid" else r"(.+)_\d+_\d+\.npy"
    match = re.fullmatch(pattern, name)
    if not match:
        raise ValueError(f"Cannot identify source image from {name}")
    return match.group(1)


def inspect_dataset(root: Path, dataset: str):
    csv_name, image_dir, mask_dir = SPECS[dataset]
    frame = pd.read_csv(root / csv_name, dtype=str)
    if not {"image", "mask"}.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Missing image/mask columns or empty CSV: {root / csv_name}")
    rows, records, source_pixels = [], [], {}
    for row in tqdm(frame.to_dict("records"), desc=f"Check {dataset}", unit="patch"):
        image_name, mask_name = filename(row["image"]), filename(row["mask"])
        if image_name != mask_name:
            raise ValueError(f"Image/mask names differ: {image_name}, {mask_name}")
        sid = infer_source(dataset, image_name)
        if "source_id" in row and row["source_id"] != sid:
            raise ValueError(f"source_id disagrees with filename: {row}")
        image_path, mask_path = root / image_dir / image_name, root / mask_dir / mask_name
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)
        if image.shape != (512, 512, 3) or image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 512x512x3 image: {image_path}, {image.shape}, {image.dtype}")
        if mask.shape != (512, 512) or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Invalid mask shape/dtype: {mask_path}")
        labels, counts = np.unique(mask, return_counts=True)
        if set(labels.tolist()) - (set(CLASS_NAMES) | {255}):
            raise ValueError(f"Unknown class IDs in {mask_path}: {labels}")
        pixels = source_pixels.setdefault(sid, np.zeros(256, dtype=np.int64))
        pixels[labels.astype(int)] += counts
        rows.append({"dataset": dataset, "source_id": sid,
                     "image": image_path.resolve().as_posix(), "mask": mask_path.resolve().as_posix()})
        records.append((image_name, file_digest(image_path), hashlib.sha256(mask.tobytes()).hexdigest()))
    result = pd.DataFrame(rows).sort_values(["source_id", "image"]).reset_index(drop=True)
    if result.image.duplicated().any() or result["mask"].duplicated().any():
        raise ValueError(f"Duplicate patch entries in {dataset}")
    return result, source_pixels, sorted(records)


def grouped_split(frame: pd.DataFrame, valid_fraction: float, seed: int):
    ids = np.array(sorted(frame.source_id.unique()))
    valid_count = math.ceil(len(ids) * valid_fraction)
    if valid_count < 1 or valid_count >= len(ids):
        raise ValueError("Split must contain at least one source image in both train and valid")
    valid_ids = set(np.random.default_rng(seed).permutation(ids)[:valid_count])
    valid = frame[frame.source_id.isin(valid_ids)].reset_index(drop=True)
    train = frame[~frame.source_id.isin(valid_ids)].reset_index(drop=True)
    validate_partition(train, valid)
    return train, valid


def prepare(maples_root: Path, idrid_root: Path, output: Path, valid_fraction=0.3, seed=42):
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    for root in (maples_root, idrid_root):
        if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
            raise ValueError("Write prepared CSVs outside the original dataset folders")
    frames, stats, fingerprint_records = {}, [], {}
    for name, root in (("maples", maples_root), ("idrid", idrid_root)):
        frame, pixels, records = inspect_dataset(root, name)
        train, valid = grouped_split(frame, valid_fraction, seed)
        frames.update({f"{name}_all": frame, f"{name}_train": train, f"{name}_valid": valid})
        fingerprint_records[name] = {"files": records, "train_ids": sorted(train.source_id.unique()),
                                     "valid_ids": sorted(valid.source_id.unique())}
        print(f"{name}: train {train.source_id.nunique()} images / {len(train)} patches; "
              f"valid {valid.source_id.nunique()} images / {len(valid)} patches")
        for split, subframe in (("train", train), ("valid", valid), ("all", frame)):
            selected = [pixels[sid] for sid in sorted(subframe.source_id.unique())]
            total = np.stack(selected).sum(axis=0)
            for class_id, class_name in {**CLASS_NAMES, 255: "ignore"}.items():
                present = sum(int(p[class_id] > 0) for p in selected)
                stats.append({"dataset": name, "split": split, "class_id": class_id,
                              "class_name": class_name, "images": len(selected),
                              "patches": len(subframe), "images_with_class": present,
                              "pixels": int(total[class_id])})
                if split != "all" and class_id not in (0, 255) and not present and any(p[class_id] for p in pixels.values()):
                    print(f"WARNING: {name}/{split} has no marked {class_name}; see class_counts.csv")
    for name, frame in frames.items():
        write_csv(output / f"{name}.csv", frame)
    write_csv(output / "class_counts.csv", pd.DataFrame(stats))
    manifest = {"version": 1, "seed": seed, "valid_fraction": valid_fraction,
                "grouping": "dataset + original source_id; never individual patches",
                "fingerprint": digest({"seed": seed, "valid_fraction": valid_fraction, "data": fingerprint_records}),
                "source_splits": {name: {k: v for k, v in record.items() if k != "files"}
                                  for name, record in fingerprint_records.items()},
                "csv_sha256": {f"{name}.csv": file_digest(output / f"{name}.csv") for name in frames}}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maples-root", type=Path, required=True)
    parser.add_argument("--idrid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.maples_root, args.idrid_root, args.output_dir, args.valid_fraction, args.seed)


if __name__ == "__main__":
    main()
