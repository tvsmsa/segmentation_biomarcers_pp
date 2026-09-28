import torch
from ml.biomarcers.config import Config
from ml.biomarcers.model_transunet import TransUNet
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
from ml.biomarcers.config_deeplab import DeepLabV3Config

config = Config()
deeplab_config = DeepLabV3Config()

@torch.no_grad()
def load_model(model_path, model_type="transunet"):
    """
    Загружает модель из чекпоинта
    """
    if model_type == "transunet":
        model = TransUNet(
            img_dim=config.PATCH_SIZE,
            num_classes=config.NUM_CLASSES
        ).to(config.DEVICE)
    elif model_type == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=config.NUM_CLASSES,
            ignore_mismatched_sizes=True
        ).to(config.DEVICE)
    elif model_type == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=config.NUM_CLASSES,
            encoder_output_stride=deeplab_config.OUTPUT_STRIDE,
            decoder_atrous_rates=deeplab_config.ATROUS_RATES,
            activation=None,
        ).to(config.DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_dice = checkpoint.get('val_dice', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}, val_dice: {val_dice}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights (no checkpoint metadata)")
    
    model.eval()
    return model