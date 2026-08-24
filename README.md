# Object Localization and Detection on Augmented MNIST

**INF265 — Project 2** · Jakob Berg & Tobias Munch

Two convolutional neural network pipelines built with PyTorch on an augmented MNIST dataset
(48×60 grayscale images with randomly positioned, rotated and resized digits on a noisy
background):

1. **Object localization** — at most one digit per image; predict objectness, a bounding box, and the digit class (0–9).
2. **Object detection** — one or more digits per image; YOLO-style 2×3 grid prediction of objectness, box and class (0/1).

Full write-up: **[report.md](report.md)** ([PDF](report.pdf)). Assignment text: [INF265_Project_2_.pdf](INF265_Project_2_.pdf).

---

## Repository layout

| Path | Description |
|------|-------------|
| [Object_localization.ipynb](Object_localization.ipynb) | Part 1 — data loading, `Net` architecture, composite localization loss, training/selection, evaluation |
| [Object_detection.ipynb](Object_detection.ipynb) | Part 2 — grid encoding, three detector architectures, detection loss, hyperparameter search, mAP evaluation |
| [report.md](report.md) / [report.pdf](report.pdf) | Full report: approach, results, discussion |
| [figures/](figures/) | All figures used in the report |
| `data/` | Dataset `.pt` tensors (~1.2 GB, git-ignored — see below) |
| [project_checklist.pdf](project_checklist.pdf) | Course checklist |

### Data

`data/` is not tracked in git. The notebooks expect these files:

```
data/localization_train.pt   data/localization_val.pt   data/localization_test.pt
data/detection_train.pt      data/detection_val.pt      data/detection_test.pt
data/list_y_true_train.pt    data/list_y_true_val.pt    data/list_y_true_test.pt
```

Dataset sizes — localization: 59,400 / 6,600 / 11,000 (train/val/test), ~9.1% with no object.
Detection: 26,874 / 2,967 / 4,981 images, 1–4 objects per image (mean ≈ 1.27).

### Requirements

`torch`, `torchvision`, `torchmetrics`, `matplotlib`, `numpy`

```bash
pip install torch torchvision torchmetrics matplotlib numpy
jupyter lab   # then run Object_localization.ipynb / Object_detection.ipynb top to bottom
```

---

## Part 1 — Object Localization

The model outputs `[pc, cx, cy, w, h, class_logits(10)]`. The loss is the unweighted sum of
BCE-with-logits on objectness (all samples), MSE on the box coordinates (positive samples only),
and cross-entropy on the digit class (positive samples only). Training used SGD (momentum 0.9),
batch size 64, and early stopping with patience 5. Selection metric: `0.5 × (accuracy + mean IoU)`.

![Localization dataset exploration](figures/localization_exploration.png)

**Architectures compared**

| Model | Conv layers (filters, kernel) | FC layers | Dropout |
|-------|-------------------------------|-----------|---------|
| Baseline / Light | (4,3), (8,3) | [64] | 0.0 |
| Deep | (32,3), (64,3), (128,3) | [512, 256] | 0.5 |
| Wide | (32,5), (64,5) | [1024, 512, 256] | 0.3 |

![Localization loss curves](figures/loss_curves_localization.png)

**Validation results**

| Model | Val Accuracy | Val IoU | Val Performance |
|-------|--------------|---------|-----------------|
| Baseline | 0.6933 | 0.4368 | 0.5651 |
| Light | 0.7068 | 0.4193 | 0.5631 |
| Deep | 0.9280 | 0.4497 | 0.6838 |
| **Wide (best)** | **0.8863** | **0.4843** | **0.6953** |

Best model: **Wide**, lr = 0.01, weight decay = 0, dropout = 0.3 →
**test accuracy 0.8878, IoU 0.4860, performance 0.6869**.

Predictions (green = ground truth, red = prediction):

| Train | Validation | Test |
|-------|------------|------|
| ![Train](figures/train_prediction_localization.png) | ![Validation](figures/validation_prediction_localization.png) | ![Test](figures/test_prediction_localization.png) |

---

## Part 2 — Object Detection

Each image is divided into a **2×3 grid**; every cell predicts
`[pc, x_local, y_local, w_local, h_local, logit_c0, logit_c1]`, with an object assigned to the cell
containing its centre (`convert_to_grid`). Decoding applies sigmoid to `(x, y)` and `exp` to `(w, h)`;
`local_to_global` + `xywh_to_xyxy` convert to global corner coordinates for evaluation.
The loss sums objectness BCE (all cells), box MSE and class cross-entropy (object cells only) over the grid.

![Detection samples](figures/sample_images_detection.png)

**Architectures** — all share an `AdaptiveAvgPool2d((2,3))` + `1×1` conv head producing 7 channels:

- **FullyConvDetector** (baseline) — four stride-2/1 conv+BN+ReLU stages up to 256 channels.
- **ResNetDetector** — three downsampling stages, each followed by a residual block, 128 final channels.
- **LightweightDetector** — depthwise separable (MobileNet-style) blocks, 128 final channels.

**Hyperparameter search** — 3 models × 4 configs = 12 runs, Adam + `ReduceLROnPlateau` (patience 3, factor 0.5):

| Config | LR | Epochs | Batch | Weight decay |
|--------|----|--------|-------|--------------|
| Baseline | 1e-3 | 20 | 64 | 0 |
| +WeightDecay | 1e-3 | 20 | 64 | 1e-4 |
| LowerLR | 5e-4 | 30 | 64 | 1e-4 |
| SmallBatch | 1e-4 | 20 | 32 | 1e-4 |

![Detection loss curves](figures/loss_curves.png)

**Validation mAP (best config per model)**

| Model | mAP | mAP@50 | mAP@75 |
|-------|-----|--------|--------|
| FullyConv — +WeightDecay | 0.3696 | 0.8046 | 0.2740 |
| **ResNet — Baseline** | **0.5122** | **0.9423** | **0.5257** |
| Lightweight — +WeightDecay | 0.3295 | 0.7509 | 0.2215 |

Full 12-run table in [report.md](report.md).

Best model: **ResNetDetector, Baseline config** → **test mAP 0.5123, mAP@50 0.9394, mAP@75 0.5145**
(per-cell accuracy 0.9653, IoU 0.7715).

mAP is used for model selection rather than per-cell accuracy/IoU because it is grid-independent
and therefore comparable across architectures.

Detections (green = ground truth, red = prediction, labels show class and confidence):

| Train | Validation | Test |
|-------|------------|------|
| ![Train](figures/train_detections.png) | ![Validation](figures/val_detections.png) | ![Test](figures/test_detections.png) |

---

## Key findings

- Residual connections gave the largest architectural gain in detection (+0.14 mAP over the fully-convolutional baseline); depthwise separable convolutions cost more accuracy than the parameter savings were worth here.
- Learning rate mattered most among the hyperparameters; 1e-3 was best, and weight decay slightly hurt the ResNet while mildly helping the simpler models.
- The gap between mAP@50 (0.94) and mAP@75 (0.51) shows detection and classification are reliable but box precision is limited by the coarse 2×3 grid — a finer grid or anchor boxes are the obvious next step.
- Both best models generalize well: validation and test scores are nearly identical in each part.

## Disclosure of AI use

ChatGPT and Claude were used for language editing, for debugging (mainly the training loops), and
for model-structure templates in the detection task. All output was fact-checked and rewritten by
the authors. See [report.md](report.md) for the full disclosure and division of tasks.
