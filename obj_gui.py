import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

thre = 0.50
is_detect_runnning = False

cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 70)

className = []
classFile = "D:/opencv/object_det/coco.names"
with open(classFile, 'rt') as f:
    classNames =f.read().rstrip('\n').split('\n')
    
configPath = "D:/opencv/object_det/yolov3.cfg"
weightPath = "D:/opencv/object_det/yolov3.weights"

net= cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

root = tk.Tk()
root.title("object detection")

label=ttk.Label(root)
label.pack(padx=10,pady=10)

cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('output',1280,720)

def start_det():
    global is_detect_runnning
    is_detect_runnning=True
    update()
    
def stop_det():
    global is_detect_runnning
    is_detect_runnning=False

start_button =ttk.Button(root,text="Start Detection",command = start_det)
start_button.pack(side=tk.LEFT,padx=10)

stop_button =ttk.Button(root, text="Stop Detection", command = stop_det)
stop_button.pack(side=tk.LEFT,padx=10)

def on_key_press(event):
    if event.char == 'q':
        root.destroy()
        stop_det()
        
root.bind('<KeyPress>',on_key_press)

def update():
    global is_detect_runnning
    
    success,img =cap.read()
    
    if not success or img is None:
        return
    
    img =cv2.resize(img,(700,500))
    
    if is_detect_runnning:
        classIds,confs,bbox = net.detect(img,confThreshold=thre)
        
        if len(classIds)!=0:
            for classId,confidence,box in  zip(classIds.flatten(),confs.flatten(),bbox):
                if 0<= classId <len(classNames):
                  cv2.rectangle(img,box,color=(0,255,0),thickness =2)
                  cv2.putText(img,classNames[classId -1].upper(),(box[0]+10, box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                  cv2.putText(img,str(round(confidence *100,2)),(box[0]+200, box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                else:
                  print(f"invalid classId : {classId}")   
                  
    img_rgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

    label.img = img_tk
    label.config(image = img_tk)

    if is_detect_runnning:
       root.after(10,update)
       
root.mainloop()

cap.release()
cv2.destroyAllWindows()       

                       
                
                
                