from train_model import build_model

from pathlib import Path

import torch
from torchvision import transforms
import cv2


def predicted(frame, conf_level=0.5):
  model.eval()
  tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  tensor = tensor.unsqueeze(0)
  
  with torch.no_grad():
    predicted = model(tensor).squeeze()
    prob = torch.sigmoid(predicted).item()
  
  label = 'person' if prob > conf_level else 'no person'
  return label, prob

if __name__ == '__main__':
  path = Path(__file__).parent
  model_path = path / 'model.pth'
  
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  if not model_path.exists():
    print("Model doesn't exist. Please run train_model.py")
  else:
    model = build_model(model_path)
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
    
    curr_label = ''
    curr_conf = 0.0
    
    while True:
      _, frame = cap.read()
      frame_copy = frame.copy()
      
      key = cv2.waitKey(1) & 0xFF
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
      if key == ord('q'):
        cv2.destroyAllWindows()
        break
      elif key == ord('p'): # predict
        curr_label, curr_conf = predicted(frame)
        
      if curr_label:
        cv2.putText(frame_copy, f"{curr_label} {float(curr_conf):.2f}",
          (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

      cv2.imshow("Camera", frame_copy)