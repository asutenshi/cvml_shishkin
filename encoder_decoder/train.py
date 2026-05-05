import torch
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


class ImageDataset(Dataset):
    def __init__(
        self,
        n=200,
        size=128,
        x=None,
        y=None,
        text=None,
        text_size=None,
        val=False,
        seed=42,
    ):
        super().__init__()
        self.n = n
        self.size = size
        self.x = x
        self.y = y
        self.text = text
        self.text_size = text_size
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.val = val
        self.rng = np.random.RandomState(seed)

        if self.val:
            self.images = []
            for _ in range(self.n):
                image = Image.new("L", (self.size, self.size), color=255)
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()

                x = self.rng.randint(10, self.size - 40) if self.x is None else self.x
                y = self.rng.randint(10, self.size - 40) if self.y is None else self.y

                text_size = (
                    self.rng.randint(3, 10)
                    if self.text_size is None
                    else self.text_size
                )
                text = (
                    "".join(self.rng.choice(self.alphabet) for _ in range(text_size))
                    if self.text is None
                    else self.text
                )

                draw.text((x, y), text, fill=0, font=font)

                tensor = self.transform(image)
                self.images.append(tensor)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.val:
            tensor = self.images[idx]
            return tensor, tensor

        image = Image.new("L", (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        x = self.rng.randint(10, self.size - 40) if self.x is None else self.x
        y = self.rng.randint(10, self.size - 40) if self.y is None else self.y

        text_size = (
            self.rng.randint(3, 10) if self.text_size is None else self.text_size
        )
        text = (
            "".join(self.rng.choice(self.alphabet) for _ in range(text_size))
            if self.text is None
            else self.text
        )

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)

        return tensor, tensor


class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (4, 4), stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # self, n=200, size=128, x=None, y=None, text=None, text_size=None, val=False, seed=42
    configs = [
        (
            "first",
            ImageDataset(2000, 256, val=False, text="ABCDE"),
            ImageDataset(500, 256, val=True, text="ABCDE"),
        ),
        (
            "second",
            ImageDataset(2000, 256, val=False, x=30, y=30, text_size=5),
            ImageDataset(500, 256, val=True, x=30, y=30, text_size=5),
        ),
        (
            "third",
            ImageDataset(2000, 256, val=False, x=30, y=30),
            ImageDataset(500, 256, val=True, x=30, y=30),
        ),
        (
            "fourth",
            ImageDataset(2000, 256, val=False),
            ImageDataset(500, 256, val=True),
        ),
    ]

    for conf_name, train_ds, val_ds in configs:
        print("-" * 50)
        print(f"RUN {conf_name} config")
        print("-" * 50)

        encoder = Encoder()
        decoder = Decoder()

        encoder.to(device)
        decoder.to(device)

        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=2)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

        epochs = 10
        hist = defaultdict(list)

        for epoch in range(epochs):
            start = time.perf_counter()
            epoch_loss = 0.0
            encoder.train()
            decoder.train()

            for imgs, _ in train_dl:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss_avg = epoch_loss / len(train_dl)
            hist["train_mse"].append(train_loss_avg)
            elapsed_time = time.perf_counter() - start

            encoder.eval()
            decoder.eval()
            val_mse_sum = 0.0

            with torch.no_grad():
                for imgs, _ in val_dl:
                    imgs = imgs.to(device)
                    latent = encoder(imgs)
                    out = decoder(latent)
                    val_mse_sum += ((out - imgs) ** 2).mean().item()

            val_mse_avg = val_mse_sum / len(val_dl)
            hist["val_mse"].append(val_mse_avg)

            print(
                f"{epoch=}, {train_loss_avg=:2f}, {val_mse_avg=:2f}, {elapsed_time=:2f}"
            )

        plt.plot(hist["train_mse"], label="train_mse")
        plt.plot(hist["val_mse"], label="val_mse")
        plt.title(f"{conf_name} config")
        plt.legend()
        plt.savefig(f"{conf_name}_hist.png")
        plt.close()

        batch = next(iter(val_dl))
        imgs = batch[0]
        imgs = imgs[:4]
        imgs = imgs.to(device)

        with torch.no_grad():
            latent = encoder(imgs)
            result = decoder(latent)

        orig = imgs.squeeze().cpu().numpy()
        res = result.squeeze().cpu().detach().numpy()
        diff = orig - res

        fig, axs = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f"{conf_name} config")

        for i in range(4):
            axs[0, i].imshow(orig[i], vmin=0, vmax=1, cmap="gray")
            axs[0, i].set_title("Original")
            axs[0, i].axis("off")

            axs[1, i].imshow(res[i], vmin=0, vmax=1, cmap="gray")
            axs[1, i].set_title("Reconstructed")
            axs[1, i].axis("off")

            axs[2, i].imshow(diff[i], vmin=-1, vmax=1, cmap="RdBu")
            axs[2, i].set_title("Difference")
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(f"{conf_name}_diffs.png")
        plt.close()

        torch.save(encoder.state_dict(), f"{conf_name}_encoder.pth")
        torch.save(decoder.state_dict(), f"{conf_name}_decoder.pth")

