import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("D:/opencv/object_det/yolov3.cfg","D:/opencv/object_det/yolov3.weights")


# Load classes
with open("D:/opencv/object_det/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for faster processing
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Forward pass through network
    outputs = net.forward()

    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Save detection information
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Check for waste objects of the same type within close proximity
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = classes[class_id]
            color = colors[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check for close proximity
            for j in range(i+1, len(boxes)):
                if j in indices:
                    x2, y2, _, _ = boxes[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < 50:
                        # Warning for close proximity of same type waste objects
                        cv2.putText(frame, "Warning: Close proximity of same type waste!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        break

    # Display frame
    cv2.imshow("Waste Detection", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
