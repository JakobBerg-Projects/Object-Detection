# %% [markdown]
# # Detection

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from math import ceil, floor
from datetime import datetime 
torch.manual_seed(123)
import torchvision
from torchvision.ops import box_convert, complete_box_iou_loss, box_iou
import math
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib.patches as patches
import numpy as np

# %% [markdown]
# # Data preperation 

# %% [markdown]
# ## Normalizing

# %%
def normalize(train, val, test):
    train_tensors = train.tensors[0]  # Get the underlying tensor
    val_tensors = val.tensors[0]  
    test_tensors = test.tensors[0]  

    mean = train_tensors.mean()    # Mean per feature
    std  = train_tensors.std()     # Std per feature
    
    train_norm = (train_tensors - mean) / std 
    val_norm = (val_tensors - mean) / std 
    test_norm = (test_tensors - mean) / std 
    return TensorDataset(train_norm, train.tensors[1]), TensorDataset(val_norm, val.tensors[1]), TensorDataset(test_norm, test.tensors[1])


# %% [markdown]
# ### Convert to grid

# %%
def convert_to_grid(list_y_true, Hout, Wout):
    Ntot = len(list_y_true)  # Antall bilder
    
    # Tom tensor for alle bilder
    y_true = torch.zeros(Ntot, Hout, Wout, 6)
    
    for img_idx, objects in enumerate(list_y_true):
        for obj in objects:
            pc, x, y, w, h, c = obj
            
            # Finn hvilken celle objektet tilhører
            col = int(x * Wout)  # Hvilken kolonne (0 til Wout-1)
            row = int(y * Hout)  # Hvilken rad (0 til Hout-1)
            
            # Klem til gyldig indeks
            col = min(col, Wout - 1)
            row = min(row, Hout - 1)
            
            # Konverter til lokale koordinater
            x_local = x * Wout - col  # x relativt til cellen
            y_local = y * Hout - row  # y relativt til cellen
            w_local = w * Wout        # w relativt til cellestørrelse
            h_local = h * Hout        # h relativt til cellestørrelse
            
            # Plasser i riktig celle
            y_true[img_idx, row, col] = torch.tensor([pc, x_local, y_local, w_local, h_local, c])
    
    return y_true

# %% [markdown]
# ### Load and analyze

# %%
def load_data(H_out, W_out, visualize=True):
    list_train = torch.load('data/list_y_true_train.pt', weights_only=False)
    list_val = torch.load('data/list_y_true_val.pt', weights_only=False)
    list_test = torch.load("data/list_y_true_test.pt", weights_only=False)
    label_sets = [list_train, list_val, list_test]

    imgs_train = torch.load("data/detection_train.pt", weights_only=False)
    imgs_val = torch.load("data/detection_val.pt", weights_only=False)
    imgs_test = torch.load("data/detection_test.pt", weights_only=False)
    img_sets = [imgs_train, imgs_val, imgs_test]

    # Sanity check / analysis
    for name, label_set in [("Train", list_train), ("Val", list_val), ("Test", list_test)]:
        counts = [len(objs) for objs in label_set]
        print(f"{name}: {len(label_set)} images, objects/image: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.2f}")

    classes = [int(obj[5]) for objs in list_train for obj in objs]
    unique, counts = np.unique(classes, return_counts=True)
    print(f"\nClass distribution (train, {sum(counts)} objects):")
    for c, n in zip(unique, counts):
        print(f"  Class {c}: {n} ({100*n/sum(counts):.1f}%)")

    
    if visualize:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            img, _ = imgs_train[i]
        
            h_img, w_img = img.shape[1], img.shape[2]
            ax.imshow(img.permute(1, 2, 0))
            for obj in list_train[i]:
                pc, x, y, w, h, c = obj
                x1 = (x - w/2) * w_img
                y1 = (y - h/2) * h_img
                rect = patches.Rectangle((x1, y1), w * w_img, h * h_img,
                                         linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 2, f'c={int(c)}', color='lime', fontsize=9, weight='bold')
            ax.set_title(f'Image {i} ({len(list_train[i])} obj)')
            ax.axis('off')
        plt.suptitle('Training Samples with Bounding Boxes')
        plt.tight_layout()
        plt.savefig("figures/sample_images_detection.png", dpi=150, bbox_inches="tight")
        plt.show()

    output_datasets = []
    for img_set, label_set in zip(img_sets, label_sets):
        labels_tensor = convert_to_grid(label_set, H_out, W_out)
        imgs = [img for img, _ in img_set]
        imgs_tensor = torch.stack(imgs, dim=0)
        output_datasets.append(TensorDataset(imgs_tensor, labels_tensor))

    return tuple(output_datasets)

train, val, test = load_data(2, 3)
train_norm, val_norm, test_norm = normalize(train, val, test)

# %% [markdown]
# # Models

# %%
class FullyConvDetector(nn.Module):
    def __init__(self, num_classes=2, Hout=2, Wout=3):
        super().__init__()
        self.Hout = Hout
        self.Wout = Wout
        num_outputs = 5 + num_classes  # pc, x, y, w, h + class logits

        self.features = nn.Sequential(
            # 1x48x60 -> 32x24x30
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x24x30 -> 64x12x15
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x12x15 -> 128x6x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128x6x8 -> 256x6x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Adaptive pooling to match grid dimensions
        self.pool = nn.AdaptiveAvgPool2d((Hout, Wout))  # 256x6x8 -> 256x2x3
        # 1x1 conv replaces the fully connected layer
        self.head = nn.Conv2d(256, num_outputs, kernel_size=1)  # 256x2x3 -> 7x2x3

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)  # (N, 7, 2, 3)
        return x


# Verify output shape
model = FullyConvDetector(num_classes=2, Hout=2, Wout=3)
dummy = torch.randn(4, 1, 48, 60)
out = model(dummy)
print(f"Output shape: {out.shape}")  # Expected: (4, 7, 2, 3)
print(f"After permute: {out.permute(0, 2, 3, 1).shape}")  # Expected: (4, 2, 3, 7)

# %%
class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)
    
class ResNetDetector(nn.Module):
    """Deeper detector with residual connections for better gradient flow."""
    def __init__(self, num_classes=2, Hout=2, Wout=3):
        super().__init__()
        self.Hout = Hout
        self.Wout = Wout
        num_outputs = 5 + num_classes

        self.features = nn.Sequential(
            # 1x48x60 -> 32x24x30
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            # 32x24x30 -> 64x12x15
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            # 64x12x15 -> 128x6x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.pool = nn.AdaptiveAvgPool2d((Hout, Wout))
        self.head = nn.Conv2d(128, num_outputs, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x


# Verify output shape
model_res = ResNetDetector(num_classes=2, Hout=2, Wout=3)
dummy = torch.randn(4, 1, 48, 60)
out = model_res(dummy)
print(f"ResNetDetector output shape: {out.shape}")  # Expected: (4, 7, 2, 3)
print(f"ResNetDetector params: {sum(p.numel() for p in model_res.parameters()):,}")


# %%
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNet-style) for efficiency."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise: one filter per input channel
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            # Pointwise: 1x1 conv to mix channels
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class LightweightDetector(nn.Module):
    """Lightweight detector using depthwise separable convolutions (fewer parameters, faster)."""
    def __init__(self, num_classes=2, Hout=2, Wout=3):
        super().__init__()
        self.Hout = Hout
        self.Wout = Wout
        num_outputs = 5 + num_classes

        self.features = nn.Sequential(
            # Initial standard conv: 1x48x60 -> 16x24x30
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16x24x30 -> 32x12x15
            DepthwiseSeparableConv(16, 32, stride=2),
            # 32x12x15 -> 64x6x8
            DepthwiseSeparableConv(32, 64, stride=2),
            # 64x6x8 -> 128x6x8
            DepthwiseSeparableConv(64, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((Hout, Wout))
        self.head = nn.Conv2d(128, num_outputs, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x


# Verify output shape
model_light = LightweightDetector(num_classes=2, Hout=2, Wout=3)
dummy = torch.randn(4, 1, 48, 60)
out = model_light(dummy)
print(f"LightweightDetector output shape: {out.shape}")  # Expected: (4, 7, 2, 3)
print(f"LightweightDetector params: {sum(p.numel() for p in model_light.parameters()):,}")

# %% [markdown]
# ## Loss function

# %%
def detection_loss(predictions, targets):
    """
    Detection loss: sum over grid cells of L_localization[h, w].
    L_localization = L_A (objectness) + L_B (box coords) + L_C (classification)
    predictions: (B, 7, Hout, Wout) - raw model output
    targets: (B, Hout, Wout, 6) - [pc, x, y, w, h, class]
    """
    pred = predictions.permute(0, 2, 3, 1)  # (B, Hout, Wout, 7)

    obj_mask   = targets[..., 0] == 1
    noobj_mask = targets[..., 0] == 0

    # L_A: objectness loss (BCE, both object and background cells)
    obj_loss   = F.binary_cross_entropy_with_logits(pred[..., 0][obj_mask],   targets[..., 0][obj_mask])
    noobj_loss = F.binary_cross_entropy_with_logits(pred[..., 0][noobj_mask], targets[..., 0][noobj_mask])

    # L_B + L_C: box and class loss (only cells with objects)
    if obj_mask.sum() > 0:
        pred_boxes = pred[..., 1:5][obj_mask]
        pred_xy    = torch.sigmoid(pred_boxes[..., :2])
        pred_wh    = torch.exp(pred_boxes[..., 2:4])

        # L_B: box coordinate loss (MSE)
        box_loss = F.mse_loss(torch.cat([pred_xy, pred_wh], dim=-1), targets[..., 1:5][obj_mask])

        # L_C: classification loss (cross entropy)
        class_loss = F.cross_entropy(pred[..., 5:][obj_mask], targets[..., 5][obj_mask].long())
    else:
        box_loss   = torch.tensor(0.0, device=predictions.device)
        class_loss = torch.tensor(0.0, device=predictions.device)

    return obj_loss + noobj_loss + box_loss + class_loss

# %% [markdown]
# ## Train

# %%
def train_model(model, train_set, val_set, epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = detection_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_set)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = detection_loss(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
        
        val_loss = running_val_loss / len(val_set)
        val_losses.append(val_loss)
        scheduler.step(val_loss)


        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

# %% [markdown]
# ## Model selection and evaluation

# %%
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def decode_boxes(raw_box):
    """Apply activations to raw model box output: sigmoid on x,y and exp on w,h."""
    xy = torch.sigmoid(raw_box[..., :2])
    wh = torch.exp(raw_box[..., 2:4])
    return torch.cat([xy, wh], dim=-1)

def get_map_results(model, eval_loader, device):
    '''
        Helper functions to get predictions and targets in the format required for mAP calculation.
        Depending on your data processing and model architecture this function can either be used as is, 
        modified to fit your needs or used as a blue print for a rewrite.
        Here it is assussmed that the image has been divide into a 2 x 3 grid.
        ----------------------------------------------------------
        Run through the data in the dataloader and collect predicitions and targets for mAP calculation.

        torchmetric mAP expects predictions and targets in the format:
        preds = [
           { "boxes": tensor([[x1, y1, x2, y2], ...]), "scores": tensor([score1, score2, ...]), "labels": tensor([label1, label2, ...])},
            ...   ]
        and targets = [
            { "boxes": tensor([[x1, y1, x2, y2], ...]), "labels": tensor([label1, label2, ...])},
            ...   ]
        where each dict in the list corresponds to one image in the dataset and contains the predicted and true results
    '''
    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(0, 2, 3, 1)                               # (B, 7, 2, 3) → (B, 2, 3, 7)
            outputs = outputs.reshape(outputs.shape[0], -1, outputs.shape[-1])  # (B, 2, 3, 7) → (B, 6, 7)
            labels = labels.reshape(labels.shape[0], -1, labels.shape[-1])      # (B, 2, 3, 6) → (B, 6, 6)
            for output, label in zip(outputs, labels):
                pred_boxes = []
                pred_scores = []
                pred_labels = []
                target_boxes = []
                target_labels = []
                # collect predicted boxes, scores and labels for the current image
                for i, cell_output in enumerate(output):
                    pred_object_presence = (torch.sigmoid(cell_output[0]) > 0.5) * 1.0
                    if pred_object_presence == 1:
                        # get propability of object presence and class probabilities to compute detection score for mAP calculation
                        obj_prop = torch.sigmoid(cell_output[0]).item()
                        class_prop = F.softmax(cell_output[5:], dim=0)
                        pred_label = torch.argmax(class_prop)
                        detect_score = obj_prop * class_prop[pred_label]
                        # decode raw box predictions before converting to global
                        decoded_box = decode_boxes(cell_output[1:5])
                        bbox_global = local_to_global(i // 3, i % 3, decoded_box)
                        bbox_xyxy = xywh_to_xyxy(bbox_global)
                        bbox_xyxy = torch.stack(bbox_xyxy)
                        # collect predicted boxes, scores and labels for the current image
                        pred_boxes.append(bbox_xyxy)
                        pred_scores.append(detect_score)
                        pred_labels.append(pred_label)
                # collect true boxes and labels for the current image
                for i, cell_label in enumerate(label):
                    true_object_presence = cell_label[0]
                    if true_object_presence == 1:
                        bbox_global = local_to_global(i // 3, i % 3, cell_label[1:5])
                        bbox_xyxy = xywh_to_xyxy(bbox_global)
                        bbox_xyxy = torch.stack(bbox_xyxy)
                        target_boxes.append(bbox_xyxy)
                        target_labels.append(int(cell_label[-1]))
                # store predictions and targets for the current image in the format required for mAP calculation
                # if there are no predicted boxes, we need to create an empty tensor for the boxes, scores and labels to avoid errors in the mAP calculation
                if len(pred_boxes) == 0:
                    pred_dict = {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                    }
                    preds.append(pred_dict)
                else:
                    pred_dict = {
                        "boxes": torch.stack(pred_boxes),
                        "scores": torch.tensor(pred_scores, device=device),
                        "labels": torch.tensor(pred_labels, device=device),
                    }
                    preds.append(pred_dict)
                # if there are no true boxes, we need to create an empty tensor for the boxes and labels to avoid errors in the mAP calculation            
                if len(target_boxes) == 0:
                    target_dict = {
                        "boxes": torch.zeros((0, 4), device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                    }
                    targets.append(target_dict)
                else:
                    target_dict = {
                        "boxes": torch.stack(target_boxes),
                        "labels": torch.tensor(target_labels, device=device),
                    }
                    targets.append(target_dict)
    
    # compute mAP using torchmetrics
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    results = metric.compute()
    # results is a dict with the mAP results for different IoU thresholds and the overall mAP
    return results        

def local_to_global(i, j, bb, width=60, height=48, cols=3, rows=2):
    x, y, w, h = bb
    # get the dimensions of a single grid cell
    cell_width, cell_height = width / cols, height / rows
    # convert from local to global coordinates
    global_x = x * cell_width + j * cell_width
    global_y = y * cell_height + i * cell_height
    global_w = w * cell_width
    global_h = h * cell_height

    return global_x, global_y, global_w, global_h

def xywh_to_xyxy(bb):
    # convert from center format to box format
    x_center, y_center, w, h = bb
    x1 = x_center - w/2
    y1 = y_center - h/2
    x2 = x_center + w/2
    y2 = y_center + h/2
    return x1, y1, x2, y2   

# %% [markdown]
# ## Model selection and evaluation

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store model constructors (lambdas) so we can create fresh instances per run
model_factories = {
    "FullyConv": lambda: FullyConvDetector(),
    "ResNet": lambda: ResNetDetector(),
    "Lightweight": lambda: LightweightDetector(),
}

# Define hyperparameter combinations to try
hyperparam_grid = [
    {"lr": 1e-3, "epochs": 20, "batch_size": 64, "weight_decay": 0},      # baseline
    {"lr": 1e-3, "epochs": 20, "batch_size": 64, "weight_decay": 1e-4},   # + weight decay
    {"lr": 5e-4, "epochs": 30, "batch_size": 64, "weight_decay": 1e-4},   # lower lr, more epochs
    {"lr": 1e-4, "epochs": 20, "batch_size": 32, "weight_decay": 1e-4},   # lowest lr, smaller batch
]

results_table = {}

for name, make_model in model_factories.items():
    for hparams in hyperparam_grid:
        run_name = f"{name}_lr{hparams['lr']}_ep{hparams['epochs']}_bs{hparams['batch_size']}_wd{hparams['weight_decay']}"

        print(f"\n--- Training {run_name} ---")
        model = make_model()
        trained, train_losses, val_losses = train_model(
            model, train_norm, val_norm,
            epochs=hparams["epochs"],
            batch_size=hparams["batch_size"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        val_loader = DataLoader(val_norm, batch_size=hparams["batch_size"])
        res = get_map_results(trained, val_loader, device)
        results_table[run_name] = {
            "model_name": name,
            "hparams": hparams,
            "trained_model": trained,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "map": res["map"],
            "map_50": res["map_50"],
            "map_75": res["map_75"],
        }
        print(f"mAP:    {res['map']:.4f}")
        print(f"mAP@50: {res['map_50']:.4f}")
        print(f"mAP@75: {res['map_75']:.4f}")

# Summary
print("\n=== Results Summary ===")
print(f"{'Run':<45} {'mAP':>8} {'mAP@50':>8} {'mAP@75':>8}")
print("-" * 71)
for run_name, r in sorted(results_table.items(), key=lambda x: x[1]["map"], reverse=True):
    print(f"{run_name:<45} {r['map']:>8.4f} {r['map_50']:>8.4f} {r['map_75']:>8.4f}")

best_run = max(results_table, key=lambda k: results_table[k]["map"])
best_model = results_table[best_run]["trained_model"]
print(f"\nBest run: {best_run} (mAP: {results_table[best_run]['map']:.4f})")

# %% [markdown]
# ## Train and validation loss

# %%
fig, axes = plt.subplots(len(model_factories), len(hyperparam_grid), figsize=(5 * len(hyperparam_grid), 4 * len(model_factories)), squeeze=False, sharey='row')

for i, model_name in enumerate(model_factories):
    for j, hparams in enumerate(hyperparam_grid):
        run_name = f"{name}_lr{hparams['lr']}_ep{hparams['epochs']}_bs{hparams['batch_size']}_wd{hparams['weight_decay']}"
        r = results_table[run_name]
        ax = axes[i][j]
        ax.plot(r["train_losses"], label="Train")
        ax.plot(r["val_losses"], label="Val")
        ax.set_title(f"{model_name}\nlr={hparams['lr']}, bs={hparams['batch_size']}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

fig.suptitle("Training & Validation Loss", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/loss_curves.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Prediction on test-set

# %%
test_loader = DataLoader(test_norm, batch_size=64)
test_res = get_map_results(best_model, test_loader, device)
print(f"=== Test Set Results ({best_run}) ===")
print(f"mAP:    {test_res['map']:.4f}")
print(f"mAP@50: {test_res['map_50']:.4f}")
print(f"mAP@75: {test_res['map_75']:.4f}")

# %% [markdown]
# ## Predicted vs actual on validation- and test-set

# %%
def pred_vs_actual_detection(dataset, preds, i, width=60, height=48, cols=3, rows=2, display_ds=None):
    # Use display_ds for the image if provided, otherwise use dataset
    img_source = display_ds if display_ds is not None else dataset
    img = img_source.tensors[0][i]         # [1, H, W]
    y   = dataset.tensors[1][i]            # [Hout, Wout, 6]

    pred = preds[i].permute(1, 2, 0)      # [7, Hout, Wout] → [Hout, Wout, 7]

    img_u8 = (img.clamp(0, 1) * 255).to(torch.uint8).repeat(3, 1, 1)

    gt_boxes, gt_labels = [], []
    pr_boxes, pr_labels, pr_scores = [], [], []

    for r in range(rows):
        for c in range(cols):
            cell_target = y[r, c]
            cell_pred   = pred[r, c]

            if cell_target[0] == 1:
                gx, gy, gw, gh = local_to_global(r, c, cell_target[1:5].tolist(), width, height, cols, rows)
                x1, y1, x2, y2 = xywh_to_xyxy((gx, gy, gw, gh))
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(int(cell_target[5].item()))

            pc = torch.sigmoid(cell_pred[0]).item()
            if pc > 0.5:
                xy = torch.sigmoid(cell_pred[1:3])
                wh = torch.exp(cell_pred[3:5])
                decoded = torch.cat([xy, wh]).tolist()
                px, py, pw, ph = local_to_global(r, c, decoded, width, height, cols, rows)
                x1, y1, x2, y2 = xywh_to_xyxy((px, py, pw, ph))
                pr_boxes.append([x1, y1, x2, y2])

                class_probs = F.softmax(cell_pred[5:], dim=0)
                pr_labels.append(int(torch.argmax(class_probs).item()))
                pr_scores.append(pc * class_probs.max().item())

    # Draw GT boxes (no labels - we'll add text via matplotlib)
    if gt_boxes:
        gt_t = torch.tensor(gt_boxes).clamp(min=0)
        gt_t[:, 2] = gt_t[:, 2].clamp(max=width)
        gt_t[:, 3] = gt_t[:, 3].clamp(max=height)
        img_u8 = torchvision.utils.draw_bounding_boxes(img_u8, gt_t, colors=["green"] * len(gt_boxes), width=1)

    # Draw pred boxes (no labels)
    if pr_boxes:
        pr_t = torch.tensor(pr_boxes).clamp(min=0)
        pr_t[:, 2] = pr_t[:, 2].clamp(max=width)
        pr_t[:, 3] = pr_t[:, 3].clamp(max=height)
        img_u8 = torchvision.utils.draw_bounding_boxes(img_u8, pr_t, colors=["red"] * len(pr_boxes), width=1)

    return img_u8.permute(1, 2, 0).numpy(), gt_boxes, gt_labels, pr_boxes, pr_labels, pr_scores

# %%
def show_detection_samples(model, ds, preprocessor, n=8, device="cpu", display_ds=None, save_path=None):
    model.eval()
    images = ds.tensors[0][:n].to(device)

    with torch.no_grad():
        try:
            x = preprocessor(images)
        except TypeError:
            x = preprocessor(images, None)
        preds = model(x).cpu()

    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i in range(n):
        img, gt_boxes, gt_labels, pr_boxes, pr_labels, pr_scores = pred_vs_actual_detection(ds, preds, i, display_ds=display_ds)
        axes[i].imshow(img)

        # Add GT labels with white text on dark background
        for box, label in zip(gt_boxes, gt_labels):
            axes[i].text(box[0], box[1] - 1, f"GT:{label}",
                         fontsize=7, color="white", fontweight="bold",
                         bbox=dict(facecolor="darkgreen", alpha=0.8, edgecolor="none", pad=0.5))

        # Add pred labels with white text on dark background
        for box, label, score in zip(pr_boxes, pr_labels, pr_scores):
            axes[i].text(box[0], box[3] + 1, f"P:{label} ({score:.2f})",
                         fontsize=7, color="white", fontweight="bold", verticalalignment="top",
                         bbox=dict(facecolor="darkred", alpha=0.8, edgecolor="none", pad=0.5))

        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

show_detection_samples(best_model, val_norm, lambda x: x.to(device), n=8, device=device, display_ds=val, save_path="figures/val_detections.png")
show_detection_samples(best_model, test_norm, lambda x: x.to(device), n=16, device=device, display_ds=test, save_path="figures/test_detections.png")


