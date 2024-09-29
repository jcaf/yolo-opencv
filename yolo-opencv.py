from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
width_middle = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if cap.isOpened():  # try to get the first frame
    rval, frame = cap.read()
else:
    cap = False

while rval:
    rval, frame = cap.read()
    results = model.predict(frame, classes=[39])
    #draw rectangle
    for r in results:
        annotator = Annotator(frame)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls     #clase
            annotator.box_label(b, model.names[int(c)])
            #coordenadas 1
            print('Coordenadas del boundingbox', b)
            print('cord x', b[0])
            print('cord y', b[1])
            print('cord xx', b[2])
            print('cord yy', b[3])

            # coordenadas 2
            # bb = box.xywh[0]
            # print('Coordenadas del boundingbox xywh',bb)
            # print('cord x', bb[0])
            # print('cord y', bb[1])
            # print('cord ancho', bb[2])
            # print('cord alto', int(bb[3]))

            xcycwh = box.xywh[0]
            x_center = int(xcycwh[0])
            y_center = int(xcycwh[1])
            cv2.line(img=frame, pt1=(int(width_middle), int(height)), pt2=(x_center, y_center), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

    frame = annotator.result()  #escrbir el box en el frame

    cv2.line(img=frame, pt1=(int(width_middle), 1), pt2=(int(width_middle), int(height - 1)), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

    cv2.imshow('JETSON Reconocimiento de botellas', frame)

    # added
    # boxes = results[0].boxes
    # for box in boxes:
    #     print('Coordenadas del boundingbox',box.xyxy)
    #print(width_middle, height)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
