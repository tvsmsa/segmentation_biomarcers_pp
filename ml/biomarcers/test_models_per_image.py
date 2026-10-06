import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml.biomarcers.config import Config
from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.metrics import print_class_metrics, compute_per_class_metrics
from ml.biomarcers.config_deeplab import DeepLabV3Config
from pathlib import Path
from ml.biomarcers.utils import load_model

config = Config()
deeplab_config = DeepLabV3Config()
ID_TO_CLASS = {v: k for k, v in config.CLASS_TO_ID.items()}
ID_TO_CLASS[0] = "background"

def test_model(model, test_loader, model_name="Model", save_results=True):
    """
    Тестирует модель. Возвращает метрики по каждому изображению
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    per_image_rows = []
    
    # Для идентификации изображений: если у DataLoader есть sampler с индексами,
    # берём имена из датасета. Иначе — порядковый номер.
    dataset = test_loader.dataset
    image_names = get_image_names(dataset)  # см. функцию ниже
    global_idx = 0
    
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
            
            # === Per-image метрики ===
            B = preds.shape[0]
            for b in range(B):
                pred_b = preds[b]            # (H, W)
                target_b = masks[b]          # (H, W)
                
                img_metrics = compute_per_class_metrics(
                    pred_b.unsqueeze(0),     # (1, H, W)
                    target_b.unsqueeze(0),   # (1, H, W)
                    num_classes=config.NUM_CLASSES,
                    ignore_index=config.IGNORE_INDEX,
                )
                
                name = image_names[global_idx] if global_idx < len(image_names) else f"img_{global_idx:05d}"
                global_idx += 1
                
                row = {"image": name}
                valid_mask = (target_b != config.IGNORE_INDEX)

                for cls_id, cls_name in ID_TO_CLASS.items():
                    if cls_id == 0:
                        continue

                    target_has_cls = ((target_b == cls_id) & valid_mask).any().item()
                    if not target_has_cls:
                        # класса нет на изображении — не пишем метрики для него
                        continue

                    #row[f"iou_{cls_name}"]    = img_metrics["iou"].get(cls_id, float("nan"))
                    row[f"dice_{cls_name}"]   = img_metrics["dice"].get(cls_id, float("nan"))
                    #row[f"prec_{cls_name}"]   = img_metrics["precision"].get(cls_id, float("nan"))
                    row[f"recall_{cls_name}"] = img_metrics["recall"].get(cls_id, float("nan"))
                
                if len(row) > 1:  # только "image" — значит ни одного класса нет
                    per_image_rows.append(row)
                #per_image_rows.append(row)
    
    # === Агрегированные метрики (как было) ===
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_per_class_metrics(
        all_preds,
        all_targets,
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX
    )
    
    per_image_df = pd.DataFrame(per_image_rows)
    
    if save_results:
        save_per_image_results(per_image_df, model_name)
    
    return metrics, per_image_df

def get_image_names(dataset) -> list[str]:
    """
    Пытается извлечь имена файлов из датасета.
    Поддерживает разные варианты структуры ImageMaskDataset.
    """
    # Вариант 1: у датасета есть список путей к изображениям
    for attr in ("image_paths", "images", "img_paths", "paths", "files", "samples"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return [Path(p).stem if isinstance(p, (str, Path)) else str(p) for p in value]
    
    # Вариант 2: датасет — это Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_names = get_image_names(dataset.dataset)
        return [base_names[i] for i in dataset.indices]
    
    # Fallback
    return []

def save_per_image_results(per_image_df: pd.DataFrame, model_name: str):
    """
    Сохранение метрик в CSV
    """
    results_dir = "biomarcers/new_idrid_per_class"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, f"{model_name}_per_image.csv")
    per_image_df.to_csv(csv_path, index=False)
    print(f"Per-image metrics saved to: {csv_path}")
    
    return csv_path

def main():
    """
    Тестирование модели
    """

    MODEL_PATH = "D:/models/transunet_tversky/TransUnet_Fold3.pth"
    MODEL_TYPE = "transunet"
    MODEL_NAME = "transunet_cet_idrid_3"
    
    #images_dir_test = "D:/idrid_blue/patches"
    images_dir_test = "D:/idrid_final/image_patches"
    masks_dir_test = "D:/idrid_test/mask_patches"
    #images_dir_test = "C:/Users/Acer/Desktop/python/IDRiD_processed/images/testing"
    #masks_dir_test  = "C:/Users/Acer/Desktop/python/IDRiD_processed/masks/testing"
    
    test_dataset = ImageMaskDataset(images_dir_test, masks_dir_test, augment_prob=0.0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,          # важно: без shuffle, чтобы индексы совпадали с именами
        num_workers=4,
        pin_memory=True,
    )
    
    model = load_model(MODEL_PATH, MODEL_TYPE)
    
    metrics, per_image_df = test_model(
        model, test_loader, MODEL_NAME, save_results=True
    )
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Per-image results saved: {len(per_image_df)} images")


if __name__ == "__main__":
    main()