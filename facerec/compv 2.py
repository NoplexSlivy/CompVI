from ultralytics import YOLO
import cv2

model = YOLO("yolov8s")

results = model.predict(data="facerec/facrec req/WIN_20240618_23_47_45_Pro.mp4", save =True)

print(results[0])
print('============')
for box in results[0].boxes:
    print(box)


