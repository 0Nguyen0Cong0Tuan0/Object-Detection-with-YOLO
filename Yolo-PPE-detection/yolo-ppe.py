from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # Webcam
# cap.set(3, 1280) # Width of webcam
# cap.set(4, 720) # Height of webcam

cap = cv2.VideoCapture('Yolo-PPE-detection/video/2.mov') # Video file


model = YOLO('Yolo-weights/best.pt') 

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck', 'truck and trailer', 'van', 'vehicle', 'wheel loader']

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    results = model(img, stream=True) # Stream=True because we are using webcam

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Get coordinates (xyxy) - Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(
            #     img,              # Image
            #     (x1, y1),         # Top-left corner
            #     (x2, y2),         # Bottom-right corner
            #     (255, 0, 255),    # Color
            #     3                 # Thickness
            # ) 
            
            # x1, y1, w, h = box.xywh[0]            
            # x1, y1, w, h = int(x1), int(y1), int(w), int(h)

            w, h = x2 - x1, y2 - y1
            
            # Draw fancy rectangle
            cvzone.cornerRect(
                img,                     # Image
                bbox=(x1, y1, w, h),     # Bounding box
                l=9,                     # Length of corner lines
                rt=2,                    # Thickness of corner lines
                colorR=(255, 0, 255)     # Color of corner lines
            )

            # Confidence
            conf = math.ceil((box.conf[0]*100)) / 100
            
            # Class name
            cls = int(box.cls[0])

            cvzone.putTextRect(
                img,                             # Image
                f'{classNames[cls]} - {conf}',   # Text
                (max(0, x1), max(35, y1-10)),    # Position 
                scale=1,                         # Scale
                thickness=2,                     # Thickness
                offset=3                         # Offset or padding
            )
            
            # Logging
            print(x1, y1, w, h)
            print(conf)
            print(classNames[cls])

    cv2.imshow("Image", img)
    cv2.waitKey(1)