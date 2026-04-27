import time

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def check_horizontal(shoulder, hip, threshold):
    if shoulder[2] < threshold or hip[2] < threshold:
        return False
        
    if (shoulder[0] == 0 and shoulder[1] == 0) or (hip[0] == 0 and hip[1] == 0):
        return False

    dx = abs(shoulder[0] - hip[0])
    dy = abs(shoulder[1] - hip[1])
    
    if dx > dy * 1.5: 
        return True
    return False

def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle
 

def detect_push_up(annotated, keypoints, down):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    # print('-' * 100)
    # print(f'{left_shoulder=} \n {right_shoulder=} \n {left_elbow=} \n {right_elbow=} \n {left_wrist=} \n {right_wrist=}')
    
    if check_horizontal(left_shoulder, left_hip, 0.5) or \
       check_horizontal(right_shoulder, right_hip, 0.5):
           
        left_angle = get_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = get_angle(right_shoulder, right_elbow, right_wrist)
        
        # cv2.putText(annotated, f'{left_angle=:.3f}, {right_angle=:.3f}', (10, annotated.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if (left_angle > 145 and right_angle > 145) and down:
            down = False
            return True, False
        
        if (left_angle < 90 and right_angle < 90) and not down:
            down = True
            return False, True
            # cv2.putText(annotated, 'Push up', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
    return False, down

model = YOLO('yolo26n-pose.pt')
model.to('cpu')

cap = cv2.VideoCapture("side.mp4")

ps = None
down = False
count = 0
start = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
        
    if start is not None and (time.perf_counter() - start) > 5:
        count = 0       
    
    results = model(frame)
    result = results[0]
    keypoints = result.keypoints.data.tolist()
    
    if not keypoints or len(keypoints) == 0:
        if start is None:
            start = time.perf_counter()
        cv2.putText(frame, f'Push up count: {count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow('Video', frame)
    else:
        start = None
        
        annotator = Annotator(frame)
        annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
        annotated = annotator.result()
        
        push_up, down = detect_push_up(annotated, keypoints[0], down)
        if push_up:
            count += 1
            
        cv2.putText(annotated, f'Push up count: {count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow('Video', annotated)
    

cap.release()
cv2.destroyAllWindows()