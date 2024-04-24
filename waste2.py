from cvzone.ClassificationModule import Classifier
import cv2
from urllib.request import urlopen
import numpy as np

# Function to capture image from ESP32-CAM
def capture_image():
    # Replace 'ESP32_CAM_IP' with the IP address of your ESP32-CAM
    url = 'http://ESP32_CAM_IP/capture'
    img_resp = urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    return img

# Initialize the classifier
classifier = Classifier('/opencv/object_det/keras_model.h5', '/opencv/object_det/labels.txt')

while True:
    # Capture image from ESP32-CAM
    img = capture_image()
    
    # Perform prediction
    pred = classifier.getPrediction(img)
    print(pred)
    
    # Display the image
    cv2.imshow("image", img)
    
    # Check for key press to exit
    if cv2.waitKey(2) & 0xFF == ord('a'):
        break

# Release resources
cv2.destroyAllWindows()
