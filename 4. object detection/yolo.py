import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture("test.mp4")
whT = 320
confThreshold =0.5
nmsThreshold= 0.2
 
#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('n').split('n')
print(classNames)
## Model Files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
 
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
 
    for i in indices:
        print("i=", i)
        #i = i[0]
        print("i=", i)
        box = bbox[i]
        print("bbox[i]", bbox[i])
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        
 
while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)
 
    cv.imshow('Image', img)
    cv.waitKey(1)
    if cv.waitKey(1) & 0xff ==ord('q'):
        break
cap.release()

