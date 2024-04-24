from cvzone.ClassificationModule import Classifier
import cv2


cap = cv2.VideoCapture(0)
classifier = Classifier('/opencv/object_det/keras_model.h5','/opencv/object_det/labels.txt')
while True:
    _, img = cap.read()
    pred =classifier.getPrediction(img)
    print(pred)
    cv2.imshow("image",img)
    if cv2.waitKey(2) & 0xff == ord('a'):
        break
