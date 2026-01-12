from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture('Yolo-car-counter/real_traffic/output.mp4') # Video file
# mask = cv2.imread('Yolo-car-counter/real_traffic/mask.png')

limits = [300, 400, 500, 600]
totalCount = set()
currentClass = None
conf = None

# Tracking object using SORT
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

model = YOLO('Yolo-weights/yolov8l.pt') 

classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
              "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    # imgRegion = cv2.bitwise_and(img, mask)
    img = cv2.resize(img, (1280, 720))

    results = model(img, stream=True) # Stream=True because we are using webcam

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0]*100)) / 100
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 5)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        print(result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(
            img, 
            bbox=(int(x1), int(y1), int(w), int(h)), 
            l=9, 
            rt=2, 
            colorR=(0, 255, 0)
        )

        cvzone.putTextRect(
            img,
            f'{currentClass} - {conf} - {int(id)}',
            (max(0, int(x1)), max(55, int(y1)-10)),
            scale=1,
            thickness=2,
            offset=3
        )

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if int(id) not in totalCount:
                totalCount.add(int(id))

    cvzone.putTextRect(
        img,
        f'Count: {len(totalCount)}',
        (50, 50),
        scale=1,
        thickness=2,
        offset=3
    )

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)