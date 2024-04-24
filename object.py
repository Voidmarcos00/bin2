import cv2 
import numpy as np
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels =[]
file_names = 'labels'
with open(file_names,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n') 
    
# print(classLabels)  

model.setInputSize(320,320)  
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

# img = cv2.imread('image1.jpeg')
# plt.imshow(img)     

# ClassIndex, confidence ,bbox = model.detect(img,confThreshold=0.55)
# print(ClassIndex)

# font_scale =3
# font =cv2.FONT_HERSHEY_PLAIN
# for ClassIndex, conf , boxes in zip(ClassIndex.flatten(), confidence.flatten(),bbox):
#     cv2.rectangle(img,boxes,(255,0,0),2)
#     cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale= font_scale,color=(0,255,0),thickness=3)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))   
# plt.show()

#VIDEO

def rescaleFrame(frame,scale=1.5):
    width=int(frame.shape[1]*scale)                  # try withput int <type>
    height=int(frame.shape[1]*scale)
    dim=(width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)         # for web-camera
# cap = cv2.VideoCapture('video2.mp4')

if not cap.isOpened():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError('cant open the video')

font_scale =3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    
    frame_resize=rescaleFrame(frame)     
    
    
    ClassIndex ,confidence,bbox = model.detect(frame,confThreshold=0.55)
    print(ClassIndex)
    
    if(len(ClassIndex)!=0):
        for ClassIndex,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if (ClassIndex<=80):
              cv2.rectangle(frame_resize,boxes,(255,0,0),2)
              cv2.putText(frame_resize,classLabels[ClassIndex-1],(boxes[0]+10,boxes[1]+40),font,fontScale= font_scale,color=(0,255,0),thickness=3)
 
    # cv2.imshow("objdetection",frame) 
    cv2.imshow('video_resized',frame_resize) 
    
    if cv2.waitKey(2) & 0xff == ord('a'):
        break
cap.release()
cv2.destroyAllWindows()            