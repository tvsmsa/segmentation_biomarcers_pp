"""Fine tune the seven-class MAPLES SegFormer on prepared original IDRiD.

From the cloned repository root in Kaggle::

    python -m ml.biomarcers.train_segformer_idrid

The prepared IDRiD dataset and MAPLES best_model.pth must be attached as Kaggle
inputs. Pass --data-root and --checkpoint if automatic discovery is ambiguous.
"""

import argparse
import json
import math
import time
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers import get_cosine_schedule_with_warmup

from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.train_segformer_maples import CLASS_NAMES, GRADIENT_ACCUMULATION_STEPS


ANNOTATED_CLASSES = (2, 3, 4, 6)
UNANNOTATED_CLASSES = (1, 5)
IGNORE_INDEX = 255
DATASET_FORMAT = "idrid-maples-segformer-7class-v1"


def find_data_root(explicit):
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [p.parent for p in Path("/kaggle/input").rglob("manifest.json")]
    matches = []
    for root in candidates:
        path = root / "manifest.json"
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("format") == DATASET_FORMAT:
            matches.append((root.resolve(), manifest))
    if len(matches) != 1:
        raise ValueError(f"Expected one prepared IDRiD manifest; found {len(matches)}. Pass --data-root.")
    root, manifest = matches[0]
    if manifest.get("class_names") != {str(k): v for k, v in CLASS_NAMES.items()}:
        raise ValueError("Prepared dataset class names differ from MAPLES checkpoint")
    if manifest.get("patch_size") != 512 or manifest.get("ignore_index") != IGNORE_INDEX:
        raise ValueError("Unexpected IDRiD patch size or ignore index")
    if manifest.get("annotated_classes") != list(ANNOTATED_CLASSES):
        raise ValueError("Unexpected IDRiD annotated classes")
    if set(manifest.get("splits", {})) != {"train", "test"}:
        raise ValueError("Prepared IDRiD must contain only official train and test splits")
    return root, manifest


def find_checkpoint(explicit):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit.resolve()
    matches = sorted(Path("/kaggle/input").rglob("best_model.pth"))
    if not matches:
        previous_run = Path("/kaggle/working/segformer_maples/best_model.pth")
        if previous_run.is_file():
            return previous_run
    if len(matches) != 1:
        raise ValueError(
            f"Expected one MAPLES best_model.pth in /kaggle/input; found {len(matches)}. "
            "Attach the checkpoint dataset and pass --checkpoint if needed."
        )
    return matches[0]


def load_split(root, split, manifest):
    csv_path = root / f"{split}.csv"
    frame = pd.read_csv(csv_path, dtype=str)
    if frame.empty or not {"image", "mask", "source_id"}.issubset(frame.columns):
        raise ValueError(f"Invalid split CSV: {csv_path}")
    if frame[["image", "mask", "source_id"]].isna().any().any():
        raise ValueError(f"Missing image, mask or source_id in {csv_path}")
    for column, folder in (("image", "images"), ("mask", "masks")):
        paths = []
        for value in frame[column]:
            if "\\" in value:
                raise ValueError(f"CSV path must use forward slashes: {value}")
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != (split, folder):
                raise ValueError(f"Invalid relative {column} path: {value}")
            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(str(path))
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate {column} paths in {csv_path}")
        frame[column] = paths
    expected = manifest["splits"][split]
    if len(frame) != expected["patches"] or frame.source_id.nunique() != expected["source_images"]:
        raise ValueError(f"CSV row counts differ from manifest: {csv_path}")
    return frame


def make_b2_config():
    # Exact B2 architecture from nvidia/segformer-b2-finetuned-ade-512-512.
    # The MAPLES checkpoint supplies all weights; no Hub download is needed.
    return SegformerConfig(
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_labels=len(CLASS_NAMES),
        id2label=CLASS_NAMES,
        label2id={name: class_id for class_id, name in CLASS_NAMES.items()},
    )


OLD_TO_NEW_BLOCK = (
    ("layer_norm_1.", "layernorm_before."),
    ("layer_norm_2.", "layernorm_after."),
    ("attention.self.query.", "attention.q_proj."),
    ("attention.self.key.", "attention.k_proj."),
    ("attention.self.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
    ("attention.self.sr.", "attention.sequence_reduction.sequence_reduction."),
    ("attention.self.layer_norm.", "attention.sequence_reduction.layer_norm."),
    ("mlp.dense1.", "mlp.fc1."),
    ("mlp.dense2.", "mlp.fc2."),
)


def convert_state_layout(state, to_new):
    """Translate SegFormer encoder keys between Transformers 4.28 and 5.x."""
    converted = {}
    for key, value in state.items():
        if to_new and key.startswith("segformer.encoder."):
            parts = key.split(".")
            kind, stage = parts[2:4]
            if kind in ("patch_embeddings", "layer_norm"):
                key = f"segformer.stages.{stage}.{kind}." + ".".join(parts[4:])
            elif kind == "block":
                block = parts[4]
                suffix = ".".join(parts[5:])
                for old, new in OLD_TO_NEW_BLOCK:
                    if suffix.startswith(old):
                        suffix = new + suffix[len(old):]
                        break
                key = f"segformer.stages.{stage}.blocks.{block}.{suffix}"
            else:
                raise ValueError(f"Unknown old SegFormer key: {key}")
        elif not to_new and key.startswith("segformer.stages."):
            parts = key.split(".")
            stage, kind = parts[2:4]
            if kind in ("patch_embeddings", "layer_norm"):
                key = f"segformer.encoder.{kind}.{stage}." + ".".join(parts[4:])
            elif kind == "blocks":
                block = parts[4]
                suffix = ".".join(parts[5:])
                for old, new in OLD_TO_NEW_BLOCK:
                    if suffix.startswith(new):
                        suffix = old + suffix[len(new):]
                        break
                key = f"segformer.encoder.block.{stage}.{block}.{suffix}"
            else:
                raise ValueError(f"Unknown new SegFormer key: {key}")
        elif to_new and key.startswith("decode_head.linear_c."):
            key = key.replace("decode_head.linear_c.", "decode_head.linear_projections.", 1)
        elif not to_new and key.startswith("decode_head.linear_projections."):
            key = key.replace("decode_head.linear_projections.", "decode_head.linear_c.", 1)
        if key in converted:
            raise ValueError(f"Duplicate converted checkpoint key: {key}")
        converted[key] = value
    return converted


def load_maples_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if names is None or {int(k): v for k, v in names.items()} != CLASS_NAMES:
        raise ValueError("Checkpoint has no matching seven-class MAPLES class_names")
    if state["decode_head.classifier.weight"].shape[0] != len(CLASS_NAMES):
        raise ValueError("Checkpoint classifier is not seven-class")
    model = SegformerForSemanticSegmentation(make_b2_config())
    model_is_new = any(k.startswith("segformer.stages.") for k in model.state_dict())
    state_is_new = any(k.startswith("segformer.stages.") for k in state)
    if model_is_new != state_is_new:
        state = convert_state_layout(state, to_new=model_is_new)
    model.load_state_dict(state, strict=True)
    print(f"Loaded MAPLES checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}")
    return model.to(device), checkpoint


def partial_cross_entropy(logits, targets):
    """Background may contain IDRiD-unannotated drusen or neovascularization."""
    valid = targets != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0
    log_probs = F.log_softmax(logits.float(), dim=1)
    exact = -log_probs.gather(1, targets.clamp(0, 6).unsqueeze(1)).squeeze(1)
    allowed_background = -torch.logsumexp(log_probs[:, [0, *UNANNOTATED_CLASSES]], dim=1)
    losses = torch.where(targets == 0, allowed_background, exact)
    return losses[valid].mean()


def observed_tversky_loss(logits, targets, alpha=0.7, beta=0.3):
    """Original Tversky formula, scored on the four IDRiD-annotated classes."""
    valid = (targets != IGNORE_INDEX).float()
    probs = torch.softmax(logits.float(), dim=1)
    scores = []
    for class_id in ANNOTATED_CLASSES:
        predicted = probs[:, class_id] * valid
        actual = (targets == class_id).float()
        tp = (predicted * actual).sum()
        fp = (predicted * (1 - actual)).sum()
        fn = ((1 - predicted) * actual).sum()
        scores.append((tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6))
    return 1 - torch.stack(scores).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = torch.zeros(len(ANNOTATED_CLASSES), dtype=torch.float64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    for images, targets in tqdm(loader, desc="Evaluation", unit="batch"):
        images, targets = images.to(device), targets.to(device)
        logits = model(pixel_values=images).logits
        logits = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)
        valid = targets != IGNORE_INDEX
        for index, class_id in enumerate(ANNOTATED_CLASSES):
            predicted = (preds == class_id) & valid
            actual = targets == class_id
            tp[index] += (predicted & actual).sum().item()
            fp[index] += (predicted & ~actual).sum().item()
            fn[index] += (~predicted & actual).sum().item()
    per_class = {}
    for index, class_id in enumerate(ANNOTATED_CLASSES):
        denom = 2 * tp[index] + fp[index] + fn[index]
        per_class[CLASS_NAMES[class_id]] = (2 * tp[index] / denom).item() if denom else 0.0
    return {"dice": per_class, "mean_dice": sum(per_class.values()) / len(per_class)}


def train(train_df, test_df, checkpoint_path, output_dir, args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Enable GPU in Kaggle notebook settings.")
    device = torch.device("cuda")
    model, initial_checkpoint = load_maples_model(checkpoint_path, device)
    train_loader = DataLoader(
        ImageMaskDataset(train_df, augment_prob=0.5), batch_size=args.batch_size,
        shuffle=True, num_workers=args.train_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ImageMaskDataset(test_df, augment_prob=0.0), batch_size=args.batch_size,
        shuffle=False, num_workers=args.test_workers, pin_memory=True,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.segformer.parameters(), "lr": args.encoder_lr},
        {"params": model.decode_head.parameters(), "lr": args.decoder_lr},
    ])
    training_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * training_steps),
        num_training_steps=training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        start = time.time()
        for index, (images, targets) in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch + 1}", unit="batch")):
            images, targets = images.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(pixel_values=images).logits
                logits = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
                loss = partial_cross_entropy(logits, targets) + 2 * observed_tversky_loss(logits, targets)
                final_single = (len(train_loader) % GRADIENT_ACCUMULATION_STEPS == 1
                                and index + 1 == len(train_loader))
                divisor = 1 if final_single else GRADIENT_ACCUMULATION_STEPS
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss. Disable --amp or lower the learning rates."
                )
            scaler.scale(loss / divisor).backward()
            if (index + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: train loss={epoch_loss:.4f}, time={time.time() - start:.1f}s")
        saved = {
            "epoch": epoch + 1,
            "source_maples_epoch": initial_checkpoint.get("epoch"),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "class_names": CLASS_NAMES,
            "annotated_classes": ANNOTATED_CLASSES,
            "loss_policy": "partial CE: background accepts 0,1,5; 2x Tversky on 2,3,4,6",
            "model_config": model.config.to_dict(),
        }
        torch.save(saved, output_dir / "last_epoch.pth")
    torch.save(saved, output_dir / "final_model.pth")
    independent_test = evaluate(model, test_loader, device)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(independent_test, indent=2), encoding="utf-8"
    )
    print(f"Independent official test Dice: {independent_test['mean_dice']:.4f}")
    print(f"Checkpoints and test metrics: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/segformer_idrid"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--test-workers", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--amp", action="store_true",
                        help="Enable fp16 mixed precision only if it is stable on your GPU")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.train_workers < 0 or args.test_workers < 0:
        parser.error("worker counts must be nonnegative")
    if args.encoder_lr <= 0 or args.decoder_lr <= 0:
        parser.error("learning rates must be positive")
    root, manifest = find_data_root(args.data_root)
    checkpoint_path = find_checkpoint(args.checkpoint)
    splits = {name: load_split(root, name, manifest) for name in ("train", "test")}
    ids = {name: set(frame.source_id) for name, frame in splits.items()}
    if ids["train"] & ids["test"]:
        raise ValueError("Source images overlap between train and test")
    for name, frame in splits.items():
        print(f"{name}: {frame.source_id.nunique()} images / {len(frame)} patches")
    print(f"Dataset: {root}\nMAPLES checkpoint: {checkpoint_path}\nOutput: {args.output_dir}")
    if Path("/kaggle/input").is_dir() and Path("/kaggle/working") not in args.output_dir.resolve().parents:
        raise ValueError("Kaggle output-dir must be inside /kaggle/working")
    torch.multiprocessing.set_start_method("spawn", force=True)
    train(splits["train"], splits["test"], checkpoint_path, args.output_dir, args)


if __name__ == "__main__":
    main()
