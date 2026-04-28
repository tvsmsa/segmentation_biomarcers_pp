import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml.biomarcers.config import Config
from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.model_transunet import TransUNet
from transformers import SegformerForSemanticSegmentation
from ml.biomarcers.metrics import print_class_metrics, compute_per_class_metrics

config = Config()

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
    results_dir = "biomarcers/test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    rows = []
    for class_id, class_name in id_to_class.items():
        if class_id == 0:
            continue
        row = {
            'model': model_name,
            'class_name': class_name,
            'iou': metrics['iou'].get(class_id, 0.0),
            'dice': metrics['dice'].get(class_id, 0.0),
            'precision': metrics['precision'].get(class_id, 0.0),
            'recall': metrics['recall'].get(class_id, 0.0)
        }
        rows.append(row)
    
    # Средние значения
    rows.append({
        'model': model_name,
        'class_name': 'MEAN',
        'iou': np.mean(list(metrics['iou'].values())),
        'dice': mean_dice,
        'precision': np.mean(list(metrics['precision'].values())),
        'recall': np.mean(list(metrics['recall'].values()))
    })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"{model_name}_test_results.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path


def main():
    """
    Тестирование модели
    """
    
    MODEL_PATH = "ml/biomarcers/checkpoints_transunet/fold_3/best_model.pth"
    MODEL_TYPE = "transunet" 
    TEST_CSV = "D:\\aspirantura3\\aspirantura\\PROF\\npy_article_fold\\train_article_fold_1.csv" 
    MODEL_NAME = "TransUNet_Fold2"
    
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
    print(f"Results saved to: test_results/{MODEL_NAME}_test_results.csv")


if __name__ == "__main__":
    main()