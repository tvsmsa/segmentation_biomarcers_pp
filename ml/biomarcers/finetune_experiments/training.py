"""Training primitives shared with the small verification suite."""
from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from transformers import get_cosine_schedule_with_warmup

from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.utils_loss import TverskyLoss
from .models import forward_logits


class BudgetExpired(Exception):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("Training time budget exhausted; resume from the last complete epoch")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(frame, batch_size, workers, *, train=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ImageMaskDataset(frame, augment_prob=0.5 if train else 0.0),
                      batch_size=batch_size, shuffle=train, drop_last=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
        else:
            # Only cuda:0 is used; Kaggle may expose a different number of GPUs.
            torch.cuda.set_rng_state(state["cuda"][0].cpu(), 0)


def atomic_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class CombinedLoss(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.overlap = (smp.losses.DiceLoss(mode="multiclass", ignore_index=255) if kind == "dice"
                        else TverskyLoss(alpha=0.7, beta=0.3, ignore_index=255))

    def forward(self, logits, targets):
        if not (targets != 255).any():
            return logits.float().sum() * 0.0
        # Overlap reductions over 512x512 patches must not overflow fp16.
        logits = logits.float()
        return self.ce(logits, targets) + 2.0 * self.overlap(logits, targets)


def make_optimizer(model, architecture, encoder_lr, head_lr, transunet_lr, weight_decay=None):
    if architecture == "transunet":
        groups = [{"params": model.parameters(), "lr": transunet_lr}]
    elif architecture == "segformer":
        groups = [{"params": model.segformer.encoder.parameters(), "lr": encoder_lr},
                  {"params": model.decode_head.parameters(), "lr": head_lr}]
    else:
        groups = [{"params": model.encoder.parameters(), "lr": encoder_lr},
                  {"params": list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), "lr": head_lr}]
    decay = (0.01 if architecture == "segformer" else 1e-4) if weight_decay is None else weight_decay
    return torch.optim.AdamW(groups, weight_decay=decay)


def make_scheduler(optimizer, batches, accumulation, epochs):
    steps = math.ceil(batches / accumulation) * epochs
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(steps * 0.1), num_training_steps=steps)


@torch.inference_mode()
def probe_amp(model, frame, batch_size, device, requested):
    """Check actual checkpoint activations before enabling fp16 training."""
    if not requested or device.type != "cuda":
        return False
    model.eval()
    images, masks = next(iter(make_loader(frame.head(batch_size), batch_size, 0)))
    images = images.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = forward_logits(model, images, masks.shape[-2:])
    if torch.isfinite(logits).all():
        return True
    del logits
    logits = forward_logits(model, images, masks.shape[-2:])
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Initial checkpoint produces non-finite logits even in FP32")
    print("WARNING: FP16 logits are non-finite on this checkpoint/device; using FP32 for this experiment.", flush=True)
    return False


@contextmanager
def batchnorm_for_single_image(model, batch_size):
    # DeepLab's global-pooling branch has BxCx1x1 features and cannot update BN
    # with B=1. Keep affine parameters trainable, use saved running statistics.
    layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
              and m.training] if batch_size == 1 else []
    for layer in layers:
        layer.eval()
    try:
        yield
    finally:
        for layer in layers:
            layer.train()


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, device,
                accumulation, deadline=float("inf"), amp=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, samples, group_samples = 0.0, 0, 0
    progress = tqdm(loader, desc="Train", leave=False, unit="batch")
    for index, (images, masks) in enumerate(progress):
        check_deadline(deadline)
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with batchnorm_for_single_image(model, len(images)), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = forward_logits(model, images, masks.shape[-2:])
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {index + 1}")
        # Normalize by actual samples after accumulation, including a short last group.
        scaler.scale(loss * len(images)).backward()
        group_samples += len(images)
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
        total_loss += float(loss.detach()) * len(images)
        samples += len(images)
        progress.set_postfix(loss=total_loss / samples)
    return total_loss / samples
