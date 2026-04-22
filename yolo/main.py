import cv2
from ultralytics import YOLO

model = YOLO('./runs/detect/figures/yolo/weights/best.pt')
model.to('cuda')

cv2.namedWindow('Camera', cv2.WINDOW_GUI_NORMAL)
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    
    key = cv2.waitKey(10) &0xFF
    if key == ord('q'):
        break
        
    results = model(frame)
    for result in results:
        boxes = result.boxes
        
        if len(boxes) == 0:
            continue
        
        for box in boxes:
            x0, y0, x1, y1 = box.xyxy[0].cpu().numpy()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            
            conf = float(box.conf)
            cls = model.names[int(box.cls)]
            
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls} {conf:.2f}', (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow('Camera', frame)