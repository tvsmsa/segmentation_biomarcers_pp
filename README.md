# ML Segmentation & Skeletonization Project

## 1. Структура проекта и зоны ответственности

```
root/
├── ml/
│   ├
│   ├── biomarcers/
│   ├   ├── train_segformer_hdd.py
│   ├   ├── train_transuent_hdd.py
│   ├   ├── train_deeplab_hdd.py
│   ├   ├── model_transunet.py
│   ├   ├── .py
│   ├   ├── loss.py
│   ├   └── config.py
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt (для сервиса)

```

---

# 2. Назначение модулей

##  `biomarcers/`
Модуль обучения SegFormer, TransUNet, Deeplab и вузализации результатов.

Отвечает за:
- загрузку датасета
- конфигурацию обучения
- определение loss-функций
- обучение моделей
- сохранение чекпоинтов

Используется для экспериментов и обучения.

---

###  model_transunet.py
Реализация архитектуры TransUNet.

###  config_*.py
Файлы конфигураций для моделей.

###  train_*.py
Обучение моделей сегментации.

###  utils_loss.py
Функции потерь.

###  metrics.py
Подсчет метрик.

###  visualize.py
Визуализация результатов, возвращается .png файл

###  visualize_streamlit.py
Реализация визуализации результатов с помощью streamlit сервиса

---

# 3. Используемые архитектуры

## SegFormer

## TransUNet

## DeeplabV3

# 4. Параметры обучения

Используются:

- Optimizer: Adam / AdamW
- Learning rate: задаётся в config.py
- Batch size: конфигурируемый
- Epochs: задаётся вручную
- Loss functions:
  - Dice Loss
  - CE + Tversky - биомаркеры

Поддерживается:
- k-fold cross validation
- сохранение лучших чекпоинтов

---

# 5. Метрики (сегментация сосудов)

## Dice coefficient

Формула:

Dice = 2TP / (2TP + FP + FN)

Зачем:
- Основная метрика для сегментации
- Устойчива к дисбалансу классов


# 6. Как запускать

## Обучение скелета

---
## Обучение биомаркеров
```
python biomarcers/train_*.py
```
---

## Метрики

```
python biomarcers/metrics.py

```
---

##  Запуск сервиса

```
streamlit run ml.biomarcers.visualize_streamlit

```