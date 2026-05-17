from pathlib import Path
import time

import matplotlib

matplotlib.use("Qt5Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset



class RoadDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images_paths = path / "images"
        self.masks_paths = path / "masks"
        self.images = sorted(list(self.images_paths.glob("*.png")))
        self.masks = sorted(list(self.masks_paths.glob("*.png")))
        self.len = len(self.images)
        self.resize = transforms.Resize((288, 512))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        image = self.resize(image)
        mask = self.resize(mask)

        image = np.array(image, dtype='f4') / 255.0
        mask = np.array(mask, dtype="f4")
        mask = (mask == 82).astype("f4")
        mask = np.expand_dims(mask, axis=0)  # 1, H, W
        
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()
            
        image = torch.from_numpy(image.transpose(2, 0, 1))  # C, H, W
        mask = torch.from_numpy(mask)

        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        # self.pool = nn.MaxUnpool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)

        for n in features:
            self.downscale.append(DoubleConv(in_channels, n))
            in_channels = n

        for n in reversed(features):
            self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
            self.upscale.append(DoubleConv(n * 2, n))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.result = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []

        for ds in self.downscale:
            x = ds(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for idx in range(0, len(self.upscale), 2):
            x = self.upscale[idx](x)
            skip = skips[idx // 2]
            cx = torch.cat((skip, x), dim=1)
            x = self.upscale[idx + 1](cx)

        return self.result(x)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        pred_area = pred_sig.view(-1)
        tar_area = target.view(-1)
        intersection = (pred_area * tar_area).sum()

        return 1 - (2 * intersection + 1) / (pred_area.sum() + tar_area.sum() + 1)


if __name__ == "__main__":
    path = Path("roads")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    epochs = 10
    

    train_ds = RoadDataset(path)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    print(len(train_ds[0][0][0]))

    model = UNet().to(device)
    # print(sum(p.numel() for p in model.parameters())) # out: 31 million

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    loss_hist = []
    for epoch in range(epochs):
        batch_loss = []
        start = time.perf_counter()
        for image, mask in train_dl:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            result = model(image)
            loss = criterion(result, mask)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss = sum(batch_loss) / len(batch_loss)
        loss_hist.append(epoch_loss)
        
        print(f'Epoch {epoch+1}: el_time={time.perf_counter() - start}, {epoch_loss=:2f}')
        
    torch.save(model.state_dict(), 'model.pth')
            