import numpy as np
from ml.biomarcers.config import Config as SegformerConfig


class DeepLabV3Config(SegformerConfig):
    #: игнорируем цвет
    IGNORE_INDEX = 255
    #: размер патча
    PATCH_SIZE = 512
    #: перекрытие патча
    STRIDE = PATCH_SIZE // 2
    #: размер батча
    BATCH_SIZE = 8
    #: количество эпох обучения
    EPOCHS = 5
    #: на чем идет обучение
    DEVICE = "cuda"
    #: Классы
    CLASSES = {
        "ERROR": (211, 255, 5),
        "OD": (250, 250, 55),
        "background": (0, 0, 0),
        "microaneurysms": (250, 50, 83),
        "hemorrhages": (42, 125, 209),
        "hard_exudates": (184, 61, 245),
        "soft_exudates": (61, 245, 61),
    }

    BACKGROUND_CLASSES = ["ERROR", "OD", "background"]
    FOREGROUND_CLASSES = [cls for cls in CLASSES.keys() if cls not in [
        "ERROR", "OD", "background"]]
    NUM_CLASSES = len(FOREGROUND_CLASSES) + 1  # + background
    CLASS_TO_ID = {cls: i + 1 for i, cls in enumerate(FOREGROUND_CLASSES)}

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    CHECKPOINT_DIR = "ml/biomarcers/checkpoints_deeplabv3"

    OUTPUT_STRIDE = 16
    ATROUS_RATES = (6, 12, 18)