# %%
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
def build_model(model_path=None):
  # EffNet
  weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
  model = torchvision.models.efficientnet_b2(weights=weights)
  
  for param in model.features.parameters():
    param.requires_grad = False
  
  in_features = model.classifier[1].in_features
  
  # У effnet нет "глубокой обученно головы", как у alexnet
  # Поэтому нужно сделать эту сложную голову и натренировать ее
  model.classifier = nn.Sequential(
      nn.Linear(in_features, 128),
      nn.ReLU(),
      nn.Linear(128, 1)
  )
  
  # AlexNet
  # weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
  # model = torchvision.models.alexnet(weights=weights)
    
  # for param in model.features.parameters():
  #   param.requires_grad = False
    
  # features = model.classifier[6].in_features
  # model.classifier[6] = nn.Linear(features, 1)
  
  if model_path != None and model_path.exists():
    model.load_state_dict(torch.load(model_path))
    print(f'Model loaded from {model_path}')
  
  return model


# %%
def train(buffer):
  if len(buffer) < 10:
    return None
  
  # Нейронка говорит, что простое выключение requires_grad не полностью действует на batchnorm
  for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
      module.eval()
    
  model.train()
  images, labels = buffer.get_batch()
  
  # optimizer.zero_grad()
  # predictions = model(images).squeeze(1)
  # loss = criterion(predictions, labels)
  # loss.backward()
  # optimizer.step()
  # return loss.item()
  
  # Чтобы натренировать голову хоть как-то буду прогонять батч 5 раз
  final_loss = 0
  for _ in range(5):
    optimizer.zero_grad()
    predictions = model(images).squeeze()
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    final_loss = loss.item()
      
  return final_loss
  

def predicted(frame, conf_level=0.5):
  model.eval()
  tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  tensor = tensor.unsqueeze(0)
  
  with torch.no_grad():
    predicted = model(tensor).squeeze()
    prob = torch.sigmoid(predicted).item()
  
  label = 'person' if prob > conf_level else 'no person'
  return label, prob

class Buffer():
  def __init__(self, maxsize=16):
    self.frames = deque(maxlen=maxsize)
    self.labels = deque(maxlen=maxsize)
  
  def append(self, tensor, label):
    self.frames.append(tensor)
    self.labels.append(label)
  
  def __len__(self):
    return len(self.frames)
  
  def get_batch(self):
    images = torch.stack(list(self.frames))
    labels = torch.tensor(list(self.labels), dtype=torch.float32)
    
    return images, labels
    
if __name__ == '__main__':
  path = Path(__file__).parent
  model_path = path / 'model.pth'
  img_path = path / 'graphic.png'
  
  model = build_model(model_path)
  print(model)
  
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    # увеличил lr в 10 раз
    lr = 0.001
  )
  
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  cap = cv2.VideoCapture(0)
  cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
  
  buffer = Buffer(16)
  count_labeled = 0
  
  loss_history = []
  pred_history = []
  
  while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if key == ord('q'):
      cv2.destroyAllWindows()
      break
    elif key == ord('1'): # person
      tensor = transform(image)
      buffer.append(tensor, 1.0)
      count_labeled += 1
      print(f'You label image as person, {count_labeled=}')
    elif key == ord('2'): # no person
      tensor = transform(image)
      buffer.append(tensor, 0.0)
      count_labeled += 1
      print(f'You label image as NO person, {count_labeled=}')
    elif key == ord('p'): # predict
      t = time.perf_counter()
      label, conf = predicted(frame)
      print(label, conf)
      pred_history.append((label, conf))
    elif key == ord('s'): # save model
      torch.save(model.state_dict(), model_path)
      
      fig, axs = plt.subplots(1, 2, figsize=(12, 5))
      axs[0].plot(range(len(loss_history)), loss_history)
      axs[0].set_xlabel('Loss')
      axs[0].set_ylabel('Epoch')
      axs[0].set_title('Loss history')
      
      indices = np.arange(len(pred_history))
      confs = [item[1] for item in pred_history]
      labels = [item[0] for item in pred_history]
      
      axs[1].plot(indices, confs, color='gray', linestyle='--', alpha=0.4, label='Confidence trend')
      
      person_idx = [i for i, l in enumerate(labels) if l == 'person']
      person_conf = [confs[i] for i in person_idx]
      
      no_person_idx = [i for i, l in enumerate(labels) if l == 'no person']
      no_person_conf = [confs[i] for i in no_person_idx]
      
      axs[1].scatter(person_idx, person_conf, color='green', label='Person')
      axs[1].scatter(no_person_idx, no_person_conf, color='red', label='No Person')
      
      axs[1].axhline(y=0.5, color='black', linestyle=':', label='Threshold (0.5)')
      
      axs[1].set_xlabel('Prediction Attempt')
      axs[1].set_ylabel('Confidence Level')
      axs[1].set_title('Predictions (Confidence & Labels)')
      axs[1].legend()
      axs[1].grid(True)
      
      
      plt.savefig(img_path)
    
    if count_labeled >= buffer.frames.maxlen:
      loss = train(buffer)
      if loss:
        loss_history.append(loss)
        print('=' * 50)
        print(f'{loss=}')
        print('=' * 50)
      count_labeled = 0