import cv2
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
LABELS = open("coco.names").read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(1)
root = tk.Tk()
root.title("Object Detection")
last_captured = {}
def detect_objects():
    _, image = cap.read()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=1)
            label = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            object_image = image[y:y+h, x:x+w]
            object_label = LABELS[class_ids[i]].lower().replace(" ", "_")
            # Check cooldown period for capturing the same object
            current_time = time.time()
            if object_label not in last_captured or (current_time - last_captured[object_label]) > COOLDOWN_PERIOD:
                image_filename = f"images/{object_label}_{current_time}.jpg"
                cv2.imwrite(image_filename, object_image)
                last_captured[object_label] = current_time
                print(f"Saved {object_label} image as {image_filename}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image=image)
    label_display.config(image=photo)
    label_display.image = photo
    root.after(10, detect_objects)

label_display = tk.Label(root)
label_display.pack()
COOLDOWN_PERIOD = 20
detect_objects()
# start_button = tk.Button(root, text="Start Detection", command=detect_objects)
# start_button.pack()
# exit_button = tk.Button(root, text="Exit", command=root.quit)
# exit_button.pack()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
