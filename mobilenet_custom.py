#Import the neccesary libraries
# python mobilenet_ssd_python.py --prototxt MobileNetSSD_deploy.prototxt --weights MobileNetSSD_deploy.caffemodel --video set06_V002.avi
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import imutils

import numpy as np
import argparse
import cv2 

# construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",help='Path to text network file: MobileNetSSD_deploy.prototxt for Caffe model or ' )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel", help='Path to weights: MobileNetSSD_deploy.caffemodel for Caffe model or ' )
parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'pessoa', 2: 'bicycle', 3: 'carro', 4: 'moto',
    5: 'bottle', 6: 'onibus', 7: 'car', 8: 'carro', 9: 'chair',
    10: 'semaforo', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor', 21: 'tvmonitor', 22: 'tvmonitor', 23: 'tvmonitor' }

# Open video file or capture device. 
if args.video:
    ##sem multi thread
    #cap = cv2.VideoCapture(args.video)
    ##com multi thread
    cap = FileVideoStream(args.video).start()
else:
    ##sem multi thread
    #cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    ##com multi thread
    cap = WebcamVideoStream(0).start()#

#Load the Caffe model 
#net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
#net = cv2.dnn.readNetFromTensorflow('models/faster_rcnn_resnet101_kitti_2018_01_28/frozen_inference_graph.pb', 'models/faster_rcnn_resnet101_kitti_2018_01_28/faster_rcnn_resnet101_kitti_2018_01_28.pbtxt')
#net = cv2.dnn.readNetFromTensorflow('models/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb', 'models/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/ssd_mobilenet_v1_ppn_coco.pbtxt')


fps = FPS().start()
#while fps._numFrames < 100:
while True:
    # Capture frame-by-frame
    ##sem multi thread
    #ret, frame = cap.read()
    ##com multi thread
    frame = cap.read()

    #resize frame for prediction
    ##sem multi thread
    #frame_resized = cv2.resize(frame,(300,300))
    ##com multi thread
    frame_resized = imutils.resize(frame, width=300)
    

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size. 
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    #blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    #net.setInput(blob)
    
    net.setInput(cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True, crop=False))
    #Prediction of network
    detections = net.forward()
    
    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        class_id = int(detections[0, 0, i, 1]) # Class label

        # Object location 
        xLeftBottom = int(detections[0, 0, i, 3] * cols) 
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop   = int(detections[0, 0, i, 5] * cols)
        yRightTop   = int(detections[0, 0, i, 6] * rows)
        
        # Factor for scale to original size of frame
        heightFactor = frame.shape[0]/300.0  
        widthFactor = frame.shape[1]/300.0 
        # Scale object detection to frame
        xLeftBottom = int(widthFactor * xLeftBottom) 
        yLeftBottom = int(heightFactor * yLeftBottom)
        xRightTop   = int(widthFactor * xRightTop)
        yRightTop   = int(heightFactor * yRightTop)
        
        if confidence > args.thr: # Filter prediction 
        # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))
        """elif confidence > 0.4:
        # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (255, 0, 0))
        elif confidence > 0.2:
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 0, 255))  
        """             
        # Draw label and confidence of prediction in frame resized
        if class_id in classNames and confidence > args.thr:
            label = classNames[class_id] #+ ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            print(label+ ": " + str(confidence)) #print class and confidence
       

           
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)

    fps.update()
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
        
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))