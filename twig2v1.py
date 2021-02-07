from ctypes import *                                               # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'twig' : [0, 255, 255], 'scissors': [238, 123, 158], 'ok' :[116, 238, 87]
    }
    
    for label, confidence, bbox in detections:
        x, y, w, h = (bbox[0],
             bbox[1],
             bbox[2],
             bbox[3])
        name_tag = label
        #name_tag = str(detection[0].decode())
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val 
                xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 5)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color, 2)
                print("({})position: (Xmin,Xmax,Ymin,Ymax)=({},{},{},{})".format(detection[0].decode(),xmin,xmax,ymin,ymax))
        
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
   
    global metaMain, netMain, altNames
   
    #configPath = "D:/yolov4/darknet2/darknet/darknet/build/darknet/x64/custom_module/soccer/cfg/yolov4-tiny-custom.cfg"                                 # Path to cfg
    #weightPath = "D:/yolov4/darknet2/darknet/darknet/build/darknet/x64/custom_module/soccer/cfg/yolov4-tiny-custom_final.weights"                                 # Path to weights                          # Path to weights
    #metaPath = "D:/yolov4/darknet2/darknet/darknet/build/darknet/x64/custom_module/soccer/cfg/obj.data" 
    configPath = "/home/pi/darknet/twig3/cfg/yolov4-tiny-custom.cfg"                                 # Path to cfg
    weightPath = "/home/pi/darknet/twig3/cfg/yolov4-tiny-custom_1000.weights"                                 # Path to weights
    metaPath = "/home/pi/darknet/twig3/cfg/obj.data"                                        # Path to meta data
    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detecti           # batch size = 1
    if metaMain is None:
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detecti
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
    #cap = cv2.VideoCapture("D:/yolov4/darknet2/darknet/darknet/build/darknet/x64/fail.mp4")                             # Local Stored video detection - Set input video
    cap = cv2.VideoCapture("/home/pi/darknet/twig3/test/26.jpg")                             # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video write
    
    out = cv2.VideoWriter(                                          # Set the Output path for video writer
        "./output0.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (frame_width, frame_height))
    
    

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(1/(time.time()-prev_time))
        cv2.imshow('>_<', image)
        cv2.imwrite("./pre.png", image)                                    # Display Image window
        cv2.waitKey(3)
        out.write(image)                                             # Write that frame into output video
    cap.release()                                                    # For releasing cap and out. 
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":  
    YOLO()                                                           # Calls the main function YOLO()
