#Marker calibration

#Libraries

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial
import time

os.system('v4l2-ctl -c focus_auto=0')
os.system('v4l2-ctl -c focus_absolute=0')
os.system('v4l2-ctl -c brightness=0')

cam = cv2.VideoCapture(0)
#Camera parameters
cam.set(11,0)
cam.set(3,640) #Width
cam.set(4,480) #Height

cv2.namedWindow("Preview")
#Create a window to preview the images


kernel = np.ones((5,5),np.uint8)
###definir los limites

lower_green = np.array([0,200,50])
upper_green = np.array([20,255,255])

#Create database
results = pd.DataFrame()
temp = {}


while True:
    ret, frame = cam.read()#take a picture
    frameHsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frameHsv,lower_green,upper_green)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours,hierarchy = cv2.findContours(opening,cv2.RETR_TREE
                                                  ,cv2.CHAIN_APPROX_SIMPLE)
    #Verify if the image was taken
    if not ret:
        print("failed to grab frame")
        break #Exit the application
    
    cv2.imshow("Preview", frame)
    #Show the image in the created window
    k = cv2.waitKey(1)
    #Wait for a key commans every milisecond
    #If a key is pressed, the following condition are asked
    if k%256 == 27:
        print("Bye")
        break
#     elif k%256 == 32:
#         img_name = "Calibration{}.png".format(img_calibration)
#         cv2.imwrite(img_name, frame)
#         print("{} saved!".format(img_name))
#         img_calibration += 1
    cv2.imshow('0',frame)
    cv2.imshow('1',mask)
    cv2.imshow('2',opening)
        

cam.release()
plt.imshow(frame)

cv2.destroyAllWindows()