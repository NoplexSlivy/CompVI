import cv2
from ultralytics import YOLO
import time
from datetime import datetime as dt
import os
from openpyxl import Workbook as wb , load_workbook

file_name = "XLlabWatch_.xlsx"
if not os.path.exists(file_name):
    workbook = wb()
    sheet = workbook.active
    sheet['A1'] = "time"
    sheet['B1'] = 'detections'
    sheet['C1'] = "% confident"
else:
    workbook = load_workbook(filename=file_name)
    sheet = wb.active

model = YOLO('yolov8n.pt')
 # create the camera capturing box
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 850)

if not cap.isOpened():
    print("error, camera could not be used")
    exit()

while True:
    work, frame = cap.read()
    if not work:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(rgb_frame, 640)
    #print(str(results))
    fLog = []

    for result in results:
        boxes = result.boxes
        #print(str(result.boxes))
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            confidence = box.conf[0]
            label = model.names[int(box.cls[0].item())]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 90, 225), 2)
            cv2.putText(frame,f'{label}: {confidence * 100:.2f}' '%', (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, .8, (255, 255, 0), 1)
    cv2.imshow('YOLOv8 detection frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cLog = [str(dt.now()), label, int(confidence.item() * 100) ]
    fLog.append(cLog)

    tLog = []
    if fLog:
        tLog.append(fLog)
    for log in tLog:
        sheet.append(log)

    
    print(tLog)
    time.sleep(2)
wb.save(filename=file_name)
cap.release()
cv2.destroyAllWindows()
