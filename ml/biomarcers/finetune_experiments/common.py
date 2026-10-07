from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from ml.biomarcers.config import Config

CLASS_NAMES = {0: "background", **{v: k for k, v in Config.CLASS_TO_ID.items()}}
CLASS_IDS = tuple(range(1, Config.NUM_CLASSES))
ANNOTATED_IDS = {"maples": (1, 6, 7, 10, 11, 12), "idrid": (6, 7, 10, 12)}
EXPERIMENTS = {
    1: {"train": ("maples_train",), "valid": ("maples_valid",),
        "test": ("maples_valid", "idrid_all")},
    2: {"train": ("idrid_train",), "valid": ("idrid_valid",),
        "test": ("idrid_valid", "maples_all")},
    3: {"train": ("maples_train", "idrid_train"),
        "valid": ("maples_valid", "idrid_valid"),
        "test": ("maples_valid", "idrid_valid")},
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    os.replace(temporary, path)


def read_split(root: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(root / f"{name}.csv", dtype={"source_id": str, "dataset": str})
    required = {"image", "mask", "source_id", "dataset"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Invalid/empty split: {name}")
    return frame


def source_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame.dataset, frame.source_id))


def validate_partition(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    overlap = source_keys(train) & source_keys(valid)
    if overlap:
        raise ValueError(f"SOURCE LEAKAGE between train/valid: {sorted(overlap)[:10]}")
    for column in ("image", "mask"):
        if set(train[column]) & set(valid[column]):
            raise ValueError(f"Shared {column} files between train/valid")
    for frame in (train, valid):
        if frame[["dataset", "image"]].duplicated().any():
            raise ValueError("Duplicate patches in a split")


def experiment_frames(root: Path, experiment: int):
    spec = EXPERIMENTS[experiment]
    train = pd.concat([read_split(root, n) for n in spec["train"]], ignore_index=True)
    valid = pd.concat([read_split(root, n) for n in spec["valid"]], ignore_index=True)
    validate_partition(train, valid)
    return train, valid


def check_prepared(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["csv_sha256"].items():
        path = root / name
        if file_digest(path) != expected_hash:
            raise ValueError(f"Prepared CSV changed: {path}. Run preparation again.")
    for experiment in EXPERIMENTS:
        train, valid = experiment_frames(root, experiment)
        for column in ("image", "mask"):
            for value in set(train[column]) | set(valid[column]):
                if not Path(value).is_file():
                    raise FileNotFoundError(value)
    return manifest
