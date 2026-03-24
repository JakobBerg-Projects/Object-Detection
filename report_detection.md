## Part 2: Object Detection

### Approach and Design Choices

**Grid encoding.** Each 48×60 grayscale image is divided into a **2×3 grid** of cells. Every cell predicts a 7-dimensional vector `[pc, x_local, y_local, w_local, h_local, logit_c0, logit_c1]`, where `pc` is an objectness score, `(x, y)` are the box centre coordinates relative to the cell, `(w, h)` are box dimensions relative to the cell size, and the last two entries are class logits. An object is assigned to the cell that contains its centre. This formulation follows the YOLO paradigm and allows the network to predict multiple objects in one forward pass.

**Coordinate conventions.** During decoding, sigmoid is applied to `(x, y)` to keep the centre within the cell, and `exp` is applied to `(w, h)` so dimensions are always positive. For evaluation, local cell coordinates are converted to global pixel coordinates via `local_to_global`, and then to corner format `(x1, y1, x2, y2)` via `xywh_to_xyxy`.

**Loss function.** The detection loss sums a per-cell localisation loss over all grid cells:

$$L_{\text{detection}} = \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} L_{\text{localization}}[h, w]$$

where each cell's contribution is the unweighted sum of three terms:

$$L_{\text{localization}} = L_A + L_B + L_C$$

| Term | Description | Method |
|------|-------------|--------|
| $L_A$ | Objectness loss — all cells | BCE on predicted vs. true `pc` |
| $L_B$ | Box coordinate loss — object cells only | MSE on decoded `(x, y, w, h)` (sigmoid on x/y, exp on w/h) |
| $L_C$ | Classification loss — object cells only | Cross-entropy on class logits |

**Normalisation.** Pixel values are normalised using per-channel mean and standard deviation computed from the **training set only**, and the same statistics are applied to validation and test sets to avoid data leakage.

---

### Models

Three convolutional architectures share the same head design: `AdaptiveAvgPool2d((2, 3))` followed by a `1×1` convolution that produces 7 output channels (one per prediction per grid cell). The final output is reshaped from `(B, 7, 2, 3)` to `(B, 2, 3, 7)` to align with the label tensor layout.

#### FullyConvDetector
A straightforward four-stage convolutional backbone. Each stage doubles the channel count while halving the spatial resolution via stride-2 convolutions.

```
Input 1×48×60
→ Conv(1→32, 3×3, s=2) + BN + ReLU   →  32×24×30
→ Conv(32→64, 3×3, s=2) + BN + ReLU  →  64×12×15
→ Conv(64→128, 3×3, s=2) + BN + ReLU → 128×6×8
→ Conv(128→256, 3×3, s=1) + BN + ReLU → 256×6×8
→ AdaptiveAvgPool(2×3)                → 256×2×3
→ Conv(256→7, 1×1)                    →   7×2×3
```

#### ResNetDetector
The same three downsampling stages as above but with a **residual block** inserted after each stage to improve gradient flow in deeper architectures. The final feature dimension is 128 (smaller than FullyConv) which reduces parameter count while the skip connections maintain representational capacity.

```
Input 1×48×60
→ Conv(1→32, s=2) + BN + ReLU + ResBlock(32)   → 32×24×30
→ Conv(32→64, s=2) + BN + ReLU + ResBlock(64)  → 64×12×15
→ Conv(64→128, s=2) + BN + ReLU + ResBlock(128)→ 128×6×8
→ AdaptiveAvgPool(2×3)                          → 128×2×3
→ Conv(128→7, 1×1)                              →   7×2×3
```

Each residual block: `Conv → BN → ReLU → Conv → BN` with an identity skip, followed by ReLU.

#### LightweightDetector
Uses **depthwise separable convolutions** (MobileNet-style) to reduce the parameter count substantially. A depthwise separable block first applies one filter per input channel (depthwise), then mixes channels with a `1×1` pointwise convolution — achieving similar receptive fields at a fraction of the FLOPs.

```
Input 1×48×60
→ Conv(1→16, 3×3, s=2) + BN + ReLU         → 16×24×30
→ DWSConv(16→32, s=2)                       → 32×12×15
→ DWSConv(32→64, s=2)                       → 64×6×8
→ DWSConv(64→128, s=1)                      → 128×6×8
→ AdaptiveAvgPool(2×3)                      → 128×2×3
→ Conv(128→7, 1×1)                          →   7×2×3
```

---

### Hyperparameter Search

All three models are trained under the same four hyperparameter configurations (12 runs total). Adam with `ReduceLROnPlateau` (patience=3, factor=0.5) is used in all runs. The best checkpoint (lowest validation loss) is saved and restored before evaluation.

| Config | Learning rate | Epochs | Batch size | Weight decay |
|--------|-------------|--------|------------|--------------|
| Baseline | 1e-3 | 20 | 64 | 0 |
| +WeightDecay | 1e-3 | 20 | 64 | 1e-4 |
| LowerLR | 5e-4 | 30 | 64 | 1e-4 |
| SmallBatch | 1e-4 | 20 | 32 | 1e-4 |

---

### Results

#### mAP Summary (Validation Set)

| Run | mAP | mAP@50 | mAP@75 |
|-----|-----|--------|--------|
| FullyConv — Baseline | 0.3017 | 0.7608 | 0.1516 |
| FullyConv — +WeightDecay | 0.3313 | 0.7697 | 0.2217 |
| FullyConv — LowerLR | 0.3014 | 0.7279 | 0.1906 |
| FullyConv — SmallBatch | 0.2901 | 0.7041 | 0.1670 |
| ResNet — Baseline | 0.3740 | 0.8344 | 0.2754 |
| ResNet — +WeightDecay | 0.4804 | 0.9275 | 0.4447 |
| ResNet — LowerLR | 0.4613 | 0.9094 | 0.4323 |
| ResNet — SmallBatch | 0.4007 | 0.8949 | 0.2782 |
| Lightweight — Baseline | 0.3022 | 0.7446 | 0.1606 |
| Lightweight — +WeightDecay | 0.2973 | 0.7237 | 0.1781 |
| Lightweight — LowerLR | 0.2782 | 0.7063 | 0.1532 |
| Lightweight — SmallBatch | 0.2317 | 0.6335 | 0.0921 |
| **Best model (test set)** | **0.4709** | **0.9165** | **0.4395** |
good

#### Training Curves

<!-- Insert: 3×4 grid of train/val loss plots (one subplot per model × hyperparameter config) -->
<!-- Figure title: "Training & Validation Loss per Model and Hyperparameter Configuration" -->

![Training and Validation Loss](figures/loss_curves.png)

#### Bounding Box Visualisations

Detections are visualised with **green** boxes for ground truth and **red** boxes for predictions. Labels show class id and, for predictions, the detection confidence score (objectness × class probability).

**Validation samples (best model):**

<!-- Insert: show_detection_samples output — best model on val_norm, n=4 -->

![Validation detections](figures/val_detections.png)

**Test samples (best model):**

<!-- Insert: show_detection_samples output — best model on test_norm, n=16 -->

![Test detections](figures/test_detections.png)

---

### Discussion

<!-- Fill in after results are available. Suggested points to cover: -->
<!-- - Which model architecture performed best and why (capacity vs. regularisation) -->
<!-- - Effect of weight decay and learning rate on convergence and generalisation -->
<!-- - Qualitative observations from the bounding box visualisations -->
<!-- - Common failure modes: missed detections, false positives, localisation errors -->
<!-- - Grid resolution limitation: one object per cell means densely packed objects may be missed -->
<!-- - Whether the lightweight model offers a practical speed/accuracy trade-off -->

---

## Conclusion

<!-- Summarise the best configuration, final test mAP, and key takeaways from the detection task -->
