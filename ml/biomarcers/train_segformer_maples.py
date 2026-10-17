"""Train the original SegFormer-B2 + CE/Tversky recipe on prepared MAPLES-DR.

Run from the repository root on Kaggle:
    python -m ml.biomarcers.train_segformer_maples
"""

import argparse
import json
import math
import os
import time
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from transformers import get_cosine_schedule_with_warmup

from ml.biomarcers.config import Config
from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.metrics import dice_score_fast
from ml.biomarcers.utils_loss import TverskyLoss


DATASET_FORMAT = "maples-segformer-7class-v1"
CLASS_NAMES = {
    0: "background",
    1: "drusen",
    2: "hard_exudates",
    3: "hemorrhages",
    4: "microaneurysms",
    5: "neovascularization_including_IRMA",
    6: "cotton_wool_spots",
}
GRADIENT_ACCUMULATION_STEPS = 2


def read_manifest(root):
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError(f"Unexpected dataset format in {path}")
    if manifest.get("patch_size") != 512 or manifest.get("ignore_index") != 255:
        raise ValueError(f"Unexpected patch size or ignore index in {path}")
    if manifest.get("class_names") != {str(k): v for k, v in CLASS_NAMES.items()}:
        raise ValueError(f"Dataset class mapping differs from this training script: {path}")
    return manifest


def find_data_root(explicit_root):
    if explicit_root:
        root = Path(explicit_root).resolve()
        read_manifest(root)
        return root
    input_root = Path("/kaggle/input")
    candidates = []
    if input_root.is_dir():
        for path in input_root.rglob("manifest.json"):
            try:
                read_manifest(path.parent)
            except (ValueError, FileNotFoundError, json.JSONDecodeError):
                continue
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one prepared MAPLES dataset in /kaggle/input; "
            f"found {len(candidates)}: {candidates}. Pass --data-root explicitly."
        )
    return candidates[0]


def load_split(root, split):
    csv_path = root / f"{split}.csv"
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty or not {"image", "mask", "source_id"}.issubset(df.columns):
        raise ValueError(f"Expected nonempty image, mask, source_id columns: {csv_path}")
    if df[["image", "mask", "source_id"]].isna().any().any():
        raise ValueError(f"Missing values in {csv_path}")
    for column, subfolder in (("image", "images"), ("mask", "masks")):
        resolved = []
        for value in df[column]:
            if "\\" in value:
                raise ValueError(f"CSV path must use forward slashes: {value}")
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != (split, subfolder):
                raise ValueError(f"Invalid relative {column} path: {value}")
            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {column} file: {path}")
            resolved.append(str(path))
        df[column] = resolved
        if df[column].duplicated().any():
            raise ValueError(f"Duplicate {column} paths in {csv_path}")
    return df


def train(train_df, val_df, output_dir, epochs=30, batch_size=8,
          train_workers=8, val_workers=4, patience=5):
    config = Config()
    config.NUM_CLASSES = len(CLASS_NAMES)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Enable GPU in the Kaggle notebook settings.")
    device = torch.device(config.DEVICE)

    train_dataset = ImageMaskDataset(train_df, augment_prob=0.5)
    val_dataset = ImageMaskDataset(val_df, augment_prob=0.0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=train_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=val_workers, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=config.NUM_CLASSES,
        id2label=CLASS_NAMES,
        label2id={label: class_id for class_id, label in CLASS_NAMES.items()},
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.segformer.encoder.parameters(), "lr": 1e-5},
        {"params": model.decode_head.parameters(), "lr": 5e-4},
    ])
    num_training_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX).to(device)
    tversky_loss = TverskyLoss(ignore_index=config.IGNORE_INDEX).to(device)

    def combined_loss(logits, targets):
        return ce_loss(logits, targets) + 2.0 * tversky_loss(logits, targets)

    scaler = torch.cuda.amp.GradScaler()
    best_dice = float("-inf")
    epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        loader_iter = tqdm(train_loader, desc=f"Train | Epoch {epoch + 1}", unit="batch")
        optimizer.zero_grad()
        for i, (imgs, masks) in enumerate(loader_iter):
            imgs = imgs.to(device)
            masks = masks.to(device)
            if epoch == 0 and i == 0:
                print("Unique values in first mask batch:", torch.unique(masks))
            with torch.cuda.amp.autocast():
                logits = model(pixel_values=imgs).logits
                logits = F.interpolate(
                    logits, masks.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = combined_loss(logits, masks)
                final_single_batch = (
                    len(train_loader) % GRADIENT_ACCUMULATION_STEPS == 1
                    and i + 1 == len(train_loader)
                )
                accumulation_divisor = 1 if final_single_batch else GRADIENT_ACCUMULATION_STEPS
                scaled_loss = loss / accumulation_divisor
            scaler.scale(scaled_loss).backward()
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or i + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            running_loss += loss.item()
            loader_iter.set_postfix({"avg_loss": running_loss / (i + 1)})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished, Avg Loss: {epoch_loss:.4f}, "
              f"Time: {time.time() - start_time:.1f}s")

        model.eval()
        all_dice = 0.0
        count = 0
        val_iter = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch")
        for imgs, masks in val_iter:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                logits = model(pixel_values=imgs).logits
                logits = F.interpolate(
                    logits, masks.shape[-2:], mode="bilinear", align_corners=False
                )
                d = dice_score_fast(logits, masks)
                all_dice += d
                count += 1
            val_iter.set_postfix({"dice": all_dice / count})
        avg_dice = all_dice / count
        print(f"Validation Dice: {avg_dice:.4f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "val_dice": avg_dice,
            "class_names": CLASS_NAMES,
        }
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")
        if avg_dice > best_dice:
            best_dice = avg_dice
            epochs_no_improve = 0
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"Best model updated! Dice: {best_dice:.4f}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(no improvement {patience} epochs).")
            break
    print(f"Training finished. Best Dice: {best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/kaggle/working/segformer_maples"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive")
    if args.train_workers < 0 or args.val_workers < 0:
        parser.error("worker counts must be nonnegative")
    root = find_data_root(args.data_root)
    manifest = read_manifest(root)
    train_df = load_split(root, "train")
    val_df = load_split(root, "test")
    if set(train_df.source_id) & set(val_df.source_id):
        raise ValueError("Train and test source IDs overlap")
    for split, df in (("train", train_df), ("test", val_df)):
        expected = manifest["splits"][split]
        if len(df) != expected["patches"] or df.source_id.nunique() != expected["source_images"]:
            raise ValueError(f"{split}.csv does not match manifest")
    print(f"Dataset: {root}")
    print(f"Train: {train_df.source_id.nunique()} images / {len(train_df)} patches")
    print(f"Validation: {val_df.source_id.nunique()} images / {len(val_df)} patches")
    print(f"Checkpoints: {args.output_dir}")
    torch.multiprocessing.set_start_method("spawn", force=True)
    train(train_df, val_df, args.output_dir, args.epochs, args.batch_size,
          args.train_workers, args.val_workers, args.patience)


if __name__ == "__main__":
    main()
