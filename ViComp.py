import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
 # create the camera capturing box
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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
cap.release()
cv2.destroyAllWindows()
