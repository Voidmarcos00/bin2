import cv2 
import matplotlib.pyplot as plt
import numpy as np

yolo =cv2.dnn.readNet("D:/opencv/object_det/yolov3.cfg","D:/opencv/object_det/yolov3.weights")
classes= []
with open("D:/opencv/object_det/coco.names",'r') as f:
    classes = f.read().splitlines()
print(len(classes))    
        

