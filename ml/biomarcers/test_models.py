import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml.biomarcers.config import Config
from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.metrics import print_class_metrics, compute_per_class_metrics
from ml.biomarcers.config_deeplab import DeepLabV3Config
from ml.biomarcers.utils import load_model

config = Config()
deeplab_config = DeepLabV3Config()

def test_model(model, test_loader, model_name="Model", save_results=True):
    """
    Тестирует модель
    """
    model.eval()

    all_preds = []
    all_targets = []

    test_iter = tqdm(test_loader, desc=f"Testing {model_name}", unit="batch")

    with torch.no_grad():
        for imgs, masks in test_iter:
            imgs = imgs.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            if hasattr(model, 'segformer'):
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
            else:
                logits = model(imgs) # TransUNet

            # Интерполяция
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            preds = logits.argmax(dim=1)  # (B, H, W)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_per_class_metrics(
        all_preds,
        all_targets,
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX
    )

    id_to_class = {v: k for k, v in config.CLASS_TO_ID.items()}
    id_to_class[0] = "background"

    # Выводим результаты
    mean_dice = print_class_metrics(metrics, id_to_class, title=f"Metrics for {model_name}")

    if save_results:
        save_test_results(metrics, id_to_class, model_name, mean_dice)

    return metrics, mean_dice


def save_test_results(metrics, id_to_class, model_name, mean_dice):
    """
    Сохранение метрик в CSV
    """
    results_dir = "biomarcers/new_idrid_results"
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    iou_vals, dice_vals, prec_vals, rec_vals = [], [], [], []
    for class_id, class_name in id_to_class.items():
        if class_id == 0:
            continue
        n = metrics.get('count', {}).get(class_id, None)
        if n == 0:
            # класс отсутствует в датасете — не пишем и не учитываем в MEAN
            continue

        iou = metrics['iou'].get(class_id, 0.0)
        dice = metrics['dice'].get(class_id, 0.0)
        precision = metrics['precision'].get(class_id, 0.0)
        recall = metrics['recall'].get(class_id, 0.0)

        iou_vals.append(iou)
        dice_vals.append(dice)
        prec_vals.append(precision)
        rec_vals.append(recall)
        
        rows.append({
            'model': model_name,
            'class_name': class_name,
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'n_images': n,
        })
    
    rows.append({
        'model': model_name,
        'class_name': 'MEAN',
        'iou': float(np.mean(iou_vals)) if iou_vals else 0.0,
        'dice': float(np.mean(dice_vals)) if dice_vals else 0.0,
        'precision': float(np.mean(prec_vals)) if prec_vals else 0.0,
        'recall': float(np.mean(rec_vals)) if rec_vals else 0.0,
        'n_images': '',
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"{model_name}_test_results.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    """
    Тестирование модели
    """
    
    MODEL_PATH = "D:/models/deeplab_dice/deeplab_model_3.pth"
    MODEL_TYPE = "deeplab" 
    TEST_CSV = "D:/combined_dataset/df_test_1.csv"
    MODEL_NAME = "deeplab_ced_fold_3_test_ds"
    #D:\models\segformer_tversky
    
    print(f"\nLoading data from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV)

    test_dataset = ImageMaskDataset(df_test, augment_prob=0.0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = load_model(MODEL_PATH, MODEL_TYPE)

    metrics, mean_dice = test_model(model, test_loader, MODEL_NAME, save_results=True)
    print(f"Model: {MODEL_NAME}")
    print(mean_dice)
    print(f"Results saved to: idrid_results/{MODEL_NAME}.csv")


if __name__ == "__main__":
    main()