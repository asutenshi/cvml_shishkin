from train_model import CyrillicCNN, CyrillicMNISTDataset

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256

path = Path(__file__).parent
data_path = path / 'Cyrillic'
model_path = path / 'model.pth'
save_path = path / 'pred.png'

if __name__ == '__main__':
  if not model_path.exists():
    print("Model doesn't exist. Please run train_model.py")
  else:
    all_dataset = CyrillicMNISTDataset(data_path, is_train=False)

    # Можно было бы на диске разделять данные по папкам:
    # Train и Test, но я схитрю с помощью random_state
    _, test_samples = train_test_split(
      all_dataset.samples,
      test_size=0.2,
      random_state=42,
      stratify=[s[1] for s in all_dataset.samples]
    )

    test_dataset = CyrillicMNISTDataset(data_path, is_train=False)
    test_dataset.samples = test_samples
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    model = CyrillicCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      images_show, labels_show  = next(iter(test_loader))
      images_show = images_show.to(device)
      outputs = model(images_show)
      _, pred_show = torch.max(outputs, 1)
      images_show = images_show.to('cpu')

      fig, axs = plt.subplots(4, 4, figsize=(12, 12))
      axs = axs.flatten()

      for i in range(16):
        axs[i].imshow(images_show[i].squeeze(), cmap='gray')
        label_class = test_dataset.ncls_to_label[labels_show[i].item()]
        label_class_pred = test_dataset.ncls_to_label[pred_show[i].item()]
        axs[i].set_title(f'pred={label_class_pred} true={label_class}')
        axs[i].axis('off')

      plt.tight_layout()
      print(f"Saving to: {save_path}")  # покажет куда сохраняет
      plt.savefig(save_path)
      print(f"Saved: {save_path.exists()}")  # покажет сохранился ли файл
      plt.show()

      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f'Test accuracy: {acc:.2f}%')

    # из-за worker=8
    del test_loader
