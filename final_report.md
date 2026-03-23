# Object localization and Detection report

## Introduction

## Dataset overview

### Data Analysis
### Challenges
### Preprocessing and Sanity Checks

# Part 1: Object Localization

## Approach and Design Choices

The task is a multi-output localization problem: given a 48×60 grayscale image, the model must simultaneously (1) detect whether a digit is present, (2) predict its bounding box, and (3) classify which digit (0–9) it is. The output vector has six components: `[pc, cx, cy, w, h, label]`.

**Architecture.** A flexible convolutional neural network (`Net`) was implemented, supporting configurable numbers of convolutional layers, filter sizes, and fully connected layers. All models share the same backbone pattern: a stack of Conv2d → ReLU → MaxPool2d blocks, followed by fully connected layers with dropout, and a single output head of size `5 + num_classes` (one objectness logit + four box coordinates + 10 class logits).

**Loss function.** A composite localization loss was used with three terms:

- **Detection loss (Lₐ):** Binary cross-entropy with logits on the objectness score `pc`, computed over all samples.
- **Bounding box loss (L_b):** Mean squared error on the normalized `(cx, cy, w, h)` coordinates, computed only for positive samples (where an object is present).
- **Classification loss (L_c):** Cross-entropy on the 10-class digit logits, again only for positive samples.

The total loss is `α·Lₐ + β·L_b + γ·L_c`, with α = β = γ = 1.0.

**Normalization.** Input images were normalized using the training set mean and standard deviation (per-dataset, not per-channel), and the same statistics were applied to validation and test sets.

**Training.** All models were trained with SGD (lr=0.01, momentum=0.9), batch size 64, and early stopping with patience=5, restoring the best checkpoint by validation performance. The performance metric is defined as `0.5 × (accuracy + mean IoU)`, where accuracy counts a prediction as correct only if the object is detected *and* the digit class is correct.

**Dataset.** The training set contains 59,400 samples, validation 6,600, and test 11,000. Approximately 9.1% of samples have no object (`pc=0`); the remainder are roughly balanced across digits 0–9, with digit 1 slightly overrepresented (~19%).

---

## Models and Hyperparameters

Four architectures were compared:

| Model    | Conv layers (filters, kernel) | FC layers       | Dropout | Parameters (approx.) |
|----------|-------------------------------|-----------------|---------|----------------------|
| Light    | (4,3), (8,3)                  | [64]            | 0.0     | ~65K                 |
| Standard | (6,5), (16,5)                 | [120, 84]       | 0.0     | ~170K                |
| Deep     | (32,3), (64,3), (128,3)       | [512, 256]      | 0.5     | ~2.5M                |
| Wide     | (32,5), (64,5)                | [1024, 512, 256]| 0.3     | ~5M                  |

All models use MaxPool2d (2×2) after each conv block and output 15 values (1 + 4 + 10).

The flat feature sizes after the convolutional stack (computed automatically via a dummy forward pass) are:
- Light: 1,040
- Standard: 1,728
- Deep: 2,560
- Wide: 6,912

---

## Results

*Fill in actual numbers after training completes.*

| Model    | Val Accuracy | Val IoU | Val Performance | Test Accuracy | Test IoU | Test Performance |
|----------|-------------|---------|-----------------|---------------|----------|------------------|
| Light    | —           | —       | —               | —             | —        | —                |
| Standard | —           | —       | —               | —             | —        | —                |
| Deep     | —           | —       | —               | —             | —        | —                |
| Wide     | —           | —       | —               | —             | —        | —                |
| **Best** | —           | —       | —               | —             | —        | —                |

*(Training curves and bounding box visualizations should be inserted here as figures.)*

**Training curves:** Loss and validation performance per epoch for each model. Models with dropout (Deep, Wide) are expected to train more slowly but generalize better.

**Bounding box visualizations:** For qualitative assessment, `pred_vs_actual` overlays the ground-truth box (green) and predicted box (red) on the image. Visualizations from train, validation, and test sets illustrate how well the model localizes unseen digits.

---

## Discussion

**Multi-task learning tradeoffs.** The composite loss forces the network to simultaneously solve detection, localization, and classification. Because all three terms are equally weighted, the model has to balance improvements across tasks. In practice, the classification loss typically dominates early training since it has 10-way cross-entropy, while the MSE bounding box loss benefits from relatively clean targets.

**No-object handling.** Approximately 9% of training images contain no digit. The objectness branch is trained on all samples with BCE, while the box and class branches are only updated on positive examples. This avoids noisy gradients from regressing boxes on blank images. If the model over-predicts objectness, false positives on empty images degrade accuracy.

**Model capacity.** The Light model may underfit due to limited capacity, especially for simultaneously learning detection, localization, and classification. The Deep and Wide models have far more parameters but risk overfitting without sufficient regularization. Early stopping with patience=5 mitigates this; dropout in the Deep and Wide models provides further regularization.

**Class imbalance.** Digit 1 makes up ~19% of samples versus ~9% for most other digits. This could cause the classifier to slightly over-predict 1s. Adjusting class weights in the cross-entropy loss could address this if it proves problematic.

**IoU as a metric.** IoU is only meaningful for positive samples (where an object exists). A model that detects no objects would trivially score zero IoU but achieve high "accuracy" on the no-object class. The composite metric `0.5 × (accuracy + IoU)` balances both concerns, though it still conflates detection and classification into a single accuracy number. Separating precision/recall for detection from digit accuracy would provide a cleaner diagnostic.

**Normalization.** Global normalization (single mean/std over all pixels) was used rather than per-channel normalization. Since images are single-channel, this is equivalent, but using per-feature normalization or 2D spatial normalization could be explored for further improvement.


## Part 2: Object detection

### Approach and design choices

### Models and hyperparameters
* Accuracy, IoU, mean(Accuracy, IoU), mAP
* Training curves
* Bounding box visualizations (train & val & test)


### Discussion

## Conclusion

