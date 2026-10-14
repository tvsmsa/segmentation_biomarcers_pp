import torch
import random
import numpy as np
from torch.utils.data import Dataset
from ml.biomarcers.config import Config

config = Config()

OLD_TO_NEW = {
    0:   0,     # background
    6:   3,     # hard_exudates  (EX)
    7:   4,     # hemorrhages    (HE)
    10:  6,     # microaneurysms (MA)
    12:  7,     # soft_exudates  (SE)
    # Остальные старые значения (если есть) → 255 (ignore)
}

def remap_mask(mask):
    """Перекодирует старые значения в новые ID, остальное → 255."""
    new_mask = np.full(mask.shape, 255, dtype=np.uint8)
    for old, new in OLD_TO_NEW.items():
        new_mask[mask == old] = new
    return new_mask


class ImageMaskDataset(Dataset):
    def __init__(self, df, augment_prob=0.5):
        """
        df: DataFrame с путями к .npy файлам
        augment_prob: вероятность применить flip/rotate
        """
        self.df = df.reset_index(drop=True)
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Загружаем уже предобработанные .npy
        img = np.load(row["image"])           # uint8, HxWx3
        mask = np.load(row["mask"])           # uint8, HxW

        if mask.ndim == 2:
            mask = remap_mask(mask)

        # Геометрическая аугментация случайно

        do_aug = random.random() < self.augment_prob
        if do_aug:
            # горизонтальный flip
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)
            # вертикальный flip
            if random.random() < 0.5:
                img = np.flip(img, axis=0)
                mask = np.flip(mask, axis=0)
            # поворот на 90°
            if random.random() < 0.25:
                img = np.rot90(img, k=1, axes=(0, 1))
                mask = np.rot90(mask, k=1, axes=(0, 1))

        img = img.copy()
        mask = mask.copy()

        # Нормализация изображения

        img = img.astype(np.float32) / 255.0
        for i, m in enumerate(Config.IMAGENET_MEAN):
            img[..., i] = (img[..., i] - m) / Config.IMAGENET_STD[i]

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
