from pathlib import Path
import time

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CyrillicMNISTDataset(Dataset):
  def __init__(self, path, is_train=False):
    self.path = Path(path)
    self.samples = []
    self.ncls_to_label = {}

    for ncls, cls in enumerate(sorted(self.path.glob('*'))):
      label = str(cls.name)
      self.ncls_to_label[ncls] = label

      for image_path in sorted(cls.glob('*.png')):
        self.samples.append((image_path, ncls))

    train_transforms = transforms.Compose([
      transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.RandomAffine(
         degrees=(-10, 10),
         translate=(0.05, 0.05),
         scale=(0.9, 1.1)
      ),
      transforms.ToTensor(),
    ])
    
    test_transforms = transforms.Compose([
      transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.ToTensor(),
    ])
    
    self.transforms = train_transforms if is_train else test_transforms

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    image_path, ncls = self.samples[idx]
    with Image.open(image_path) as img:
      mask = img.getchannel('A')
    mask = self.transforms(mask)
    mask = (mask > 0).float()
    mask = (mask - 0.5) / 0.5
      
    return mask, ncls

class CyrillicCNN(nn.Module):
  def __init__(self):
    super(CyrillicCNN, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 2) # 64 -> 32
    
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2, 2) # 32 -> 16

    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.relu3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(2, 2) # 16 -> 8
    
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(128 * 8 * 8, 256)
    self.relu4 = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(256, 34)
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.pool2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.pool3(x)
    
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu4(x)
    x = self.dropout(x)
    x = self.fc2(x)
    
    return x

if __name__ == '__main__':
  path = Path(__file__).parent
  data_path = path / 'Cyrillic'
  save_path = path
  model_path = save_path / "model.pth"

  all_dataset = CyrillicMNISTDataset(data_path)

  train_samples, test_samples = train_test_split(
    all_dataset.samples,
    test_size=0.2,
    random_state=42,
    stratify=[s[1] for s in all_dataset.samples]
  )

  train_dataset = CyrillicMNISTDataset(data_path, is_train=True)
  train_dataset.samples = train_samples

  test_dataset = CyrillicMNISTDataset(data_path, is_train=False)
  test_dataset.samples = test_samples

  # batch_size = 64
  batch_size = 256
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch.device('cpu')

  model = CyrillicCNN().to(device)

  num_epochs = 100
  patience = 4

  train_loss = []
  train_acc = []
  test_loss = []
  test_acc = []

  best_val_loss = float('inf')
  epochs_no_imporve = 0

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)


  if not model_path.exists():
    for epoch in range(num_epochs):
      start = time.time()
      model.train()
      run_loss = 0.0
      total = 0
      correct = 0

      for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = (images.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      epoch_loss = run_loss / len(train_loader)
      epoch_acc = 100 * (correct / total)
      train_loss.append(epoch_loss)
      train_acc.append(epoch_acc)

      model.eval()
      val_loss, val_correct, val_total = 0.0, 0, 0

      with torch.no_grad():
        for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          val_loss += loss.item()
          
          _, predicted = torch.max(outputs.data, 1)
          val_total += labels.size(0)
          val_correct += (predicted == labels).sum().item()

      val_epoch_loss = val_loss / len(test_loader)
      val_epoch_acc = 100 * (val_correct / val_total)

      scheduler.step(val_epoch_loss)

      test_loss.append(val_epoch_loss)
      test_acc.append(val_epoch_acc)

      if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_imporve = 0       
        torch.save(model.state_dict(), model_path)
      else:
        epochs_no_imporve += 1
        if epochs_no_imporve >= patience:
          print("Early stopping triggered")
          break
        
      print(f"Epoch {epoch}: Train Loss {epoch_loss:.3f}, Train Acc {epoch_acc:.2f}%\nVal Loss {val_epoch_loss:.3f}, Val Acc {val_epoch_acc:.2f}%")

      end = time.time()
      print(f'time={end - start:.2f}')

    plt.figure()
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()

    plt.subplot(122)
    plt.title("Acc")
    plt.plot(train_acc, label='Train acc')
    plt.plot(test_acc, label='Test acc')
    plt.legend()

    plt.savefig(save_path / 'train.png')
    plt.show()

    del train_loader
    del test_loader
  else:
    model.load_state_dict(torch.load(model_path))
