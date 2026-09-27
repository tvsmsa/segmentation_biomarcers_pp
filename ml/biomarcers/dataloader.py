import torch
import random
import numpy as np
from torch.utils.data import Dataset
from ml.biomarcers.config import Config
from pathlib import Path

config = Config()


class ImageMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment_prob=0.5):
        """
        df: DataFrame с путями к .npy файлам
        augment_prob: вероятность применить flip/rotate
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.augment_prob = augment_prob
        self.image_paths = []
        self.mask_paths = []
        
        for img_path in self.images_dir.glob("*.npy"):
            mask_path = self.masks_dir / f"{img_path}.npy"
            if img_path.exists():
                self.image_paths.append(img_path)
        for mask_path in self.masks_dir.glob("*.npy"):
            if mask_path.exists():
                self.mask_paths.append(mask_path)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Загружаем уже предобработанные .npy
        img = np.load(self.image_paths[idx])           # uint8, HxWx3
        mask = np.load(self.mask_paths[idx])           # uint8, HxW

        # Геометрическая аугментация случайно

        # do_aug = random.random() < self.augment_prob
        # if do_aug:
        #     # горизонтальный flip
        #     if random.random() < 0.5:
        #         img = np.flip(img, axis=1)
        #         mask = np.flip(mask, axis=1)
        #     # вертикальный flip
        #     if random.random() < 0.5:
        #         img = np.flip(img, axis=0)
        #         mask = np.flip(mask, axis=0)
        #     # поворот на 90°
        #     if random.random() < 0.25:
        #         img = np.rot90(img, k=1, axes=(0, 1))
        #         mask = np.rot90(mask, k=1, axes=(0, 1))

        img = img.copy()
        mask = mask.copy()

        # Нормализация изображения

        img = img.astype(np.float32) / 255.0
        for i, m in enumerate(Config.IMAGENET_MEAN):
            img[..., i] = (img[..., i] - m) / Config.IMAGENET_STD[i]

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
