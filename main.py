import cv2
import numpy as np

net = cv2.dnn.readNet('yolov4-custom_last.weights', 'yolov4-custom.cfg')
classes = []
with open('classes.txt', 'r') as f:
    classes = f.read().splitlines()
i = int(input("Nhap so anh: "))
if i >300:
    i = int(input("Nhap lai so anh: "))
img = cv2.imread(f"./images/xuoc_{i}.jpg")

# img = cv2.resize(img, None, fx=0.5, fy=0.5)
height, width = img.shape[:2]

if i in range (2, 102):
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = 180, scale = 1)
    img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 2))

if len(indexes) == 0:
    print("t")

if len(indexes) > 0:
    print("f")
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, y), (x + w + 100, y - 30), color, -1)
        cv2.putText(img, label + " " + confidence, (x, y - 4), font, 2, (255, 255, 255), 2)

cv2.imshow('anh sau khi kiem tra', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
