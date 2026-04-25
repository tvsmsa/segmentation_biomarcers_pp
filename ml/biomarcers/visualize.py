import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.biomarcers.config import Config
from ml.biomarcers.model_transunet import TransUNet


config = Config()

CLASS_COLORS = {
    1:  [1.0, 0.0, 0.0],
    2:  [0.0, 1.0, 0.0],
    3:  [0.0, 0.0, 1.0],
    4:  [1.0, 1.0, 0.0],
    5:  [1.0, 0.0, 1.0],
    6:  [0.0, 1.0, 1.0],
    7:  [0.5, 0.5, 0.5],
    8:  [1.0, 0.5, 0.0],
    9:  [0.5, 0.0, 0.5],
    10: [0.0, 0.5, 0.0],
    11: [0.5, 0.0, 0.0],
    12: [0.0, 0.0, 0.5],
    13: [0.5, 0.5, 0.0],
    14: [0.0, 0.5, 0.5],
}

ID_TO_CLASS = {v: k for k, v in config.CLASS_TO_ID.items()}
ID_TO_CLASS[0] = "background"


def load_model(model_path, device="cpu"):
    """Загружает модель TransUNet"""
    model = TransUNet(
        img_dim=config.PATCH_SIZE,
        num_classes=config.NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}, val_dice: {checkpoint.get('val_dice', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded")
    return model


def load_sample(df, idx, image_dir, mask_dir):
    """Загружает один снимок и маску из датафрейма, возвращает тензоры."""
    row = df.iloc[idx]

    img_name = row['image'].split('\\')[-1]
    mask_name = row['mask'].split('\\')[-1]

    image = np.load(os.path.join(image_dir, img_name))
    gt_mask = np.load(os.path.join(mask_dir, mask_name))

    # Нормализация как в ImageMaskDataset
    img = image.astype(np.float32) / 255.0
    for i, m in enumerate(config.IMAGENET_MEAN):
        img[..., i] = (img[..., i] - m) / config.IMAGENET_STD[i]

    image_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(gt_mask).long().unsqueeze(0)

    return image_tensor, mask_tensor, img_name


def predict(model, image_tensor, mask_tensor, device="cpu"):
    """Прогоняет изображение через модель, возвращает pred_mask, gt_mask, img_vis."""
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    if logits.shape[-2:] != mask_tensor.shape[-2:]:
        logits = F.interpolate(logits, size=mask_tensor.shape[-2:], mode="bilinear", align_corners=False)

    pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
    gt_mask = mask_tensor.squeeze().cpu().numpy()

    # Денормализация для визуализации
    img_vis = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    for i, (m, s) in enumerate(zip(config.IMAGENET_MEAN, config.IMAGENET_STD)):
        img_vis[..., i] = img_vis[..., i] * s + m
    img_vis = np.clip(img_vis, 0, 1)

    return pred_mask, gt_mask, img_vis


def colorize(mask):
    """Раскрашивает маску (H,W) в (H,W,3)."""
    colored = np.zeros((*mask.shape, 3))
    for cls in np.unique(mask):
        if cls != 0:
            colored[mask == cls] = CLASS_COLORS.get(cls, [1, 1, 1])
    return colored


def visualize_prediction(image, gt_mask, pred_mask, save_path=None):
    """Отображает и сохраняет визуализацию: 5 изображений + текстовый отчёт"""
    fig = plt.figure(figsize=(22, 12))

    # Верхний ряд
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(image)
    ax2.imshow(colorize(gt_mask), alpha=0.5)
    ax2.set_title("Ground Truth (overlay)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(image)
    ax3.imshow(colorize(pred_mask), alpha=0.5)
    ax3.set_title("Prediction (overlay)", fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Нижний ряд
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.imshow(colorize(gt_mask))
    ax4.set_title("Ground Truth Mask", fontsize=14, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(2, 3, 6)
    ax5.imshow(colorize(pred_mask))
    ax5.set_title("Predicted Mask", fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Описание
    ax_desc = fig.add_subplot(2, 3, 4)
    ax_desc.axis('off')
    ax_desc.set_xlim(0, 10)
    ax_desc.set_ylim(0, 10)

    gt_classes = set(np.unique(gt_mask)) - {0}
    pred_classes = set(np.unique(pred_mask)) - {0}
    found = gt_classes & pred_classes
    missed = gt_classes - pred_classes
    extra = pred_classes - gt_classes

    lines = [f"GT classes: {len(gt_classes)}  |  Pred classes: {len(pred_classes)}", ""]

    if found:
        lines.append(f"[FOUND] ({len(found)}):")
        for cls in sorted(found):
            gt_px = (gt_mask == cls).sum()
            pred_px = (pred_mask == cls).sum()
            lines.append(f"  - {ID_TO_CLASS.get(cls, cls)}")
            lines.append(f"    GT: {gt_px} px  |  Pred: {pred_px} px")
    else:
        lines.append("[FOUND]: none")

    lines.append("")
    if missed:
        lines.append(f"[MISSED] ({len(missed)}):")
        for cls in sorted(missed):
            lines.append(f"  - {ID_TO_CLASS.get(cls, cls)}")
    else:
        lines.append("[MISSED]: none")

    lines.append("")
    if extra:
        lines.append(f"[FALSE POS] ({len(extra)}):")
        for cls in sorted(extra):
            lines.append(f"  - {ID_TO_CLASS.get(cls, cls)}")
    else:
        lines.append("[FALSE POS]: none")

    text = "\n".join(lines)
    ax_desc.text(2.0, 9.5,
                 text,
                 fontsize=15,
                 verticalalignment='top',
                 fontfamily='monospace',
                 color='black',
                 bbox=dict(boxstyle='round',
                           facecolor='lightgrey',
                           alpha=0.9,
                           pad=0.8)
                 )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    # === Настройки ===
    MODEL_PATH = "/Users/mamaevalex/aspirantura/PROF/models/best_model.pth"
    TEST_CSV = "/Users/mamaevalex/aspirantura/PROF/npy_article_fold/train_article_fold_1.csv"

    IMAGE_DIR = "/Users/mamaevalex/aspirantura/PROF/npy_article_fold/fold_1/images"
    MASK_DIR = "/Users/mamaevalex/aspirantura/PROF/npy_article_fold/fold_1/masks"

    DEVICE = torch.device("cpu")

    # === Загрузка ===
    model = load_model(MODEL_PATH, DEVICE)
    df = pd.read_csv(TEST_CSV)
    print(f"Samples: {len(df)}")

    # === Выбор снимка ===
    sample_idx = 10 # Указать индекс [0; len(df) - 1]

    image_tensor, mask_tensor, img_name = load_sample(df, sample_idx, IMAGE_DIR, MASK_DIR)
    print(f"Loaded: {img_name}")

    # === Предсказание ===
    pred_mask, gt_mask, img_vis = predict(model, image_tensor, mask_tensor, DEVICE)

    # === Визуализация ===
    visualize_prediction(img_vis, gt_mask, pred_mask, save_path=f"vis_{sample_idx}.png")


if __name__ == "__main__":
    main()
