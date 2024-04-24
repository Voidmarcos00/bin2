import cv2

thres = 0.5
# img = cv2.imread('abhi.jpeg')
# print("Image shape:", img.shape)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


classNames = []
classFile = r'd:/opencv/object_det/coco.names'


print("classFile:", classFile)  
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)

configPath = r'd:/opencv/object_det/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' 
weightPath = r'd:/opencv/object_det/frozen_inference_graph.pb'   
    
net = cv2.dnn_DetectionModel(weightPath,configPath) 
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

while True:
    ret,video_data = cap.read()
    classIds, confs, bbox =net.detect(video_data,confThreshold=thres)
    print(classIds,bbox)   
   # Inside the loop where you draw the bounding boxes and labels
    print("ClassIds:", classIds)
    print("ClassNames:", classNames)
    if len(classIds) != 0:
     for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        print("ClassId:", classId)
        print("ClassNames Length:", len(classNames))
        if classId - 1 < len(classNames):
          cv2.rectangle(video_data, box, color=(0, 255, 0), thickness=2)
          cv2.putText(video_data, classNames[classId - 1], (box[0] + 10, box[1] + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
          print("Invalid classId:", classId)

    cv2.imshow("video_live",video_data)
    if cv2.waitKey(10) == ord("a"):
        break
cap.release()
cv2.destroyAllWindows()            