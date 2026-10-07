"""Checkpoint-compatible model constructors without pretrained downloads."""
from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torchvision import models as tv_models
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from ml.biomarcers.config import Config
from .segformer_compat import align_segformer_state


class TransUNet(nn.Module):
    """Same layers, names and forward pass as the legacy TransUNet."""

    def __init__(self):
        super().__init__()
        self.img_dim, self.num_classes = 512, Config.NUM_CLASSES
        self.patch_size, self.hidden_dim = 16, 768
        resnet = tv_models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, 768, kernel_size=1)
        layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(64, Config.NUM_CLASSES, 1),
        )

    def forward(self, images):
        projected = self.projection(self.encoder(images))
        batch, channels, height, width = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)
        features = self.transformer(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        return self.decoder(features)


def build_model(architecture: str):
    if architecture == "deeplab":
        return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                               classes=Config.NUM_CLASSES, encoder_output_stride=16,
                               decoder_atrous_rates=(6, 12, 18), activation=None)
    if architecture == "segformer":
        config = SegformerConfig(num_labels=Config.NUM_CLASSES, depths=[3, 4, 6, 3],
                                 hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768,
                                 semantic_loss_ignore_index=255)
        return SegformerForSemanticSegmentation(config)
    if architecture == "transunet":
        return TransUNet()
    raise ValueError(f"Unknown architecture: {architecture}")


def checkpoint_data(path: Path):
    # User-provided tensor dictionaries only; do not unpickle arbitrary modules.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_model(path: Path, architecture: str):
    checkpoint = checkpoint_data(path)
    model = build_model(architecture)
    state = checkpoint.get("model_state_dict", checkpoint)
    if architecture == "segformer":
        state = align_segformer_state(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    return model


def forward_logits(model, images, size):
    result = model(pixel_values=images).logits if isinstance(model, SegformerForSemanticSegmentation) else model(images)
    if result.shape[-2:] != size:
        result = nn.functional.interpolate(result, size=size, mode="bilinear", align_corners=False)
    return result


def discover_group(directory: Path):
    name = directory.name.lower()
    architectures = [a for a in ("deeplab", "segformer", "transunet") if a in name]
    losses = [loss for loss in ("dice", "tversky") if loss in name]
    if len(architectures) != 1 or len(losses) != 1:
        raise ValueError("Select ONE group folder, e.g. 'Deeplab dice', 'Segformer tversky', 'Transunet dice'")
    paths = sorted(directory.glob("*.pth"))
    if len(paths) != 3:
        raise ValueError(f"Expected exactly 3 .pth checkpoints directly inside {directory}; found {len(paths)}")
    mapping = {}
    for path in paths:
        match = re.search(r"(?:fold[_ ]?|model_)([123])(?:\D|$)", path.stem, re.IGNORECASE)
        if not match or int(match.group(1)) in mapping:
            raise ValueError(f"Ambiguous experiment number in checkpoint filename: {path.name}")
        mapping[int(match.group(1))] = path
    if set(mapping) != {1, 2, 3}:
        raise ValueError("Checkpoint numbers must be exactly 1, 2, 3")
    return architectures[0], losses[0], mapping
