"""Small notebook helpers; training/evaluation remain separate CLI processes."""
from __future__ import annotations

import re
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_dataset(csv_name, explicit=None, search_root="/kaggle/input"):
    if explicit:
        path = Path(explicit)
        if not (path / csv_name).is_file():
            raise FileNotFoundError(f"Expected {csv_name} in {path}")
        return path
    candidates = sorted({p.parent for p in Path(search_root).rglob(csv_name)})
    if len(candidates) != 1:
        raise ValueError(f"Set an explicit dataset root for {csv_name}; candidates: {candidates}")
    return candidates[0]


def find_group(group, explicit=None, search_root="/kaggle/input"):
    normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        if normalize(path.name) != normalize(group):
            raise ValueError(f"GROUP={group!r} disagrees with folder {path.name!r}")
        return path
    candidates = sorted({p.parent for p in Path(search_root).rglob("*.pth")
                         if normalize(p.parent.name) == normalize(group)})
    if len(candidates) != 1:
        raise ValueError(f"Set MODELS_DIR explicitly for {group}; candidates: {candidates}")
    return candidates[0]


def run_cli(command, *arguments):
    subprocess.run([sys.executable, "-m", f"ml.biomarcers.finetune_experiments.{command}",
                    *map(str, arguments)], check=True)


def restore_saved_run(source, output_root, expected_name):
    """Copy a previous Kaggle output attached as an input, preserving newer local work."""
    if not source:
        return
    source = Path(source)
    info = json.loads((source / "run.json").read_text(encoding="utf-8"))
    name = f"{info['architecture']}_{info['loss']}{'_smoke' if info['smoke_patches'] else ''}"
    if name != expected_name:
        raise ValueError(f"Previous output is {name}, expected {expected_name}")
    target = Path(output_root) / name
    if target.exists():
        if json.loads((target / "run.json").read_text(encoding="utf-8")) != info:
            raise ValueError(f"Conflicting output already exists: {target}")
        print(f"Keeping existing working output: {target}")
        return
    temporary = target.with_name(target.name + ".restoring")
    shutil.copytree(source, temporary)
    temporary.rename(target)
    print(f"Restored previous output to {target}")
