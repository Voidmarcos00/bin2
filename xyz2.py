import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("D:/opencv/object_det/yolov3.cfg","D:/opencv/object_det/yolov3.weights")
    

# Load class labels
class_labels = ["plastic", "e-waste", "medical waste"]

# Define confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Capture video from webcam (change 0 to the appropriate camera index if using a different camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layer_indices]
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Get object bounding box
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 625, 0), 2)
                cv2.putText(frame, class_labels[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 625, 0), 2)

    # Display the frame with detected objects
    cv2.imshow("Real-time Object Detection", frame)

    # Check for exit key (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
