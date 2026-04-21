import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import BatchNorm1d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
classes = ["square", "circle", "triangle"]


class ShapesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        for cls_name in classes:
            img_dir = root / cls_name / "images"
            meta_dir = root / cls_name / "labels"
            for img_path in sorted(img_dir.glob("*.png")):
                labels_path = meta_dir / (img_path.stem + ".txt")
                self.images.append((img_path, labels_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, labels_path = self.images[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            tensor = self.transform(Image.fromarray(img))
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        cls, x, y, w, h = map(float, labels_path.read_text().split())
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        return tensor, int(cls), bbox


class SimpleDetector(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),           
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),           
            
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.flatten = nn.Flatten()

        self.cls_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        features = self.backbone(x)
        
        features = self.flatten(features)
        
        cls_preds = self.cls_head(features)
        bbox_preds = torch.sigmoid(self.bbox_head(features))
        
        return cls_preds, bbox_preds


def calc_giou(pred, target, loss=False):
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = area_p + area_t - inter

    iou = inter / (union + 1e-7)

    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)

    giou = iou - (area_c - union) / (area_c + 1e-7)
    
    if loss:
        return (1 - giou).mean()
        
    return giou.mean()


def detection_loss(cls_pred, bbox_pred, cls_targets, bbox_targets, lambda_bbox=10.0):
    loss_cls = F.cross_entropy(cls_pred, cls_targets)
    loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_targets)
    loss_bbox += calc_giou(bbox_pred, bbox_targets, loss=True)
    return loss_cls + lambda_bbox * loss_bbox, loss_cls, loss_bbox


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

root = Path("shapes_dataset")
# root = Path("shapes_dataset_bg")
# root = Path("shapes_dataset_random")
train_ds = ShapesDataset(root / "train", transform=transform)
val_ds = ShapesDataset(root / "val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4)

model = SimpleDetector(num_classes=len(classes)).to(device)
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.cls_head.parameters(), 'lr': 1e-3},
    {'params': model.bbox_head.parameters(), 'lr': 1e-3}
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 50
save_path = root / "best.pt"
accuracy_threshold = 0.95
giou_threshold = 0.90
no_improve_count = 10

history = defaultdict(list)
best_acc = 0.0
best_giou = -1.0
no_improve = 0

if save_path.exists():
    model.load_state_dict(torch.load(save_path))
else:
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_cls = train_box = 0.0
        for images, cls_t, bbox_t in train_loader:
            images, cls_t, bbox_t = (
                images.to(device),
                cls_t.to(device),
                bbox_t.to(device),
            )
            optimizer.zero_grad()
            cls_pred, bbox_pred = model(images)
            loss, lc, lb = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t, 10)
            # print(f'{lc=} {lb=}')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_cls += lc.item()
            train_box += lb.item()

        n = len(train_loader)
        history["train_loss"].append(train_loss / n)
        history["train_cls"].append(train_cls / n)
        history["train_box"].append(train_box / n)

        model.eval()
        val_loss = correct = total = 0.0
        val_giou_hist = []
        with torch.no_grad():
            for images, cls_t, bbox_t in val_loader:
                images, cls_t, bbox_t = (
                    images.to(device),
                    cls_t.to(device),
                    bbox_t.to(device),
                )
                cls_pred, bbox_pred = model(images)
                print(bbox_pred[0], bbox_t[0])
                loss, _, _ = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t, 10)
                val_loss += loss.item()
                val_giou_hist.append(calc_giou(bbox_pred, bbox_t))
                correct += (cls_pred.argmax(1) == cls_t).sum().item()
                total += cls_t.size(0)

        val_acc = correct / total
        val_giou = np.mean(val_giou_hist)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        history["val_giou"].append(val_giou)

        scheduler.step()

        if (val_acc + val_giou >= best_acc + best_giou):
            best_acc = val_acc
            best_giou = val_giou
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or val_acc > best_acc - 0.01 or val_giou > best_giou - 0.01:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={history['train_loss'][-1]:.4f}  "
                f"val={history['val_loss'][-1]:.4f}  "
                f"acc={val_acc:.3f}  "
                f"giou={val_giou:.3f}"
            )

        # if (val_acc >= accuracy_threshold and val_giou > giou_threshold) or no_improve >= no_improve_count:
        #     break

    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(history["val_acc"], label="val acc", color="green")
    plt.plot(history["val_giou"], label="val giou", color="red")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(root / 'hist.png')


def show_predictions(loader, model, n=8):
    model.eval()
    it = iter(loader)
    images, cls_t, bbox_t = next(it)
    images = images.to(device)
    with torch.no_grad():
        cls_pred, bbox_pred = model(images)
    print(bbox_pred[0], bbox_t[0])
    print(bbox_pred[1], bbox_t[1])
    preds = cls_pred.argmax(1).cpu()

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        H, W = img_np.shape[:2]

        cx, cy, bw, bh = bbox_t[i].numpy()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * W,
                bh * H,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
        )

        cx, cy, bw, bh = bbox_pred[i].cpu().numpy()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * W,
                bh * H,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
        )

        gt_name = classes[cls_t[i]]
        pr_name = classes[preds[i]]
        color = "green" if preds[i] == cls_t[i] else "red"
        ax.set_title(f"Real:{gt_name}  Predicted:{pr_name}", color=color, fontsize=9)
        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(root / 'pred.png')
    # plt.show()


show_predictions(val_loader, model)
