from ultralytics import YOLO
import cv2

model = YOLO('Yolo-weights/yolov8l.pt') 
results = model(r"run-Yolo/images/3.jpg", show=True)

for result in results:
    im_array = result.plot()
    cv2.imshow("YOLO Result", im_array)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()