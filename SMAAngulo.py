#This programs computes the angle made by the SMA
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import cv2
import os


os.system('v4l2-ctl -c focus_auto=0')#Eliminate autofocus of the camera
os.system('v4l2-ctl -c focus_absolute=0')#Set the focus of the camera to zero


#OpenSerialPort

ser = serial.Serial()
ser.port='/dev/ttyACM0'#Is the port connected
ser.baudrate=115200#Baudrate
ser.timeout=0.1#Time out for waitng an answer
ser.open()#the port is open

#Camera initialization

#An object for modifiyng the camera properties
cap = cv2.VideoCapture(0)
time.sleep(1)#Wait some time 
cap.set(11,0)
cap.set(3,640) #Width
cap.set(4,480) #Height
cap.set(cv2.CAP_PROP_BRIGHTNESS,0)#Brightness of the image
 
#####Find ChessBoard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*10.7#milimeters
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#these are p
with open('mtx.npy', 'rb') as f:
    mtx = np.load(f)
with open('dist.npy', 'rb') as f:
    coef = np.load(f)
FLAG = True
while FLAG:
    _, frame = cap.read() #Take picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, coef)
        R = cv2.Rodrigues(rvecs)[0]
        R = np.concatenate((R[:,0:2],tvecs),axis=1)
        cv2.drawChessboardCorners(frame, (9,6), corners2, ret)
        #Undistort
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, coef, (w,h), 1, (w,h))
        #En Undistort
        #cv2.imshow('img', frame)
        cv2.waitKey(1)
        FLAG = False




os.system('v4l2-ctl -c brightness=0')

########

kernel = np.ones((5,5),np.uint8)
###definir los limites

lower_green = np.array([0,200,50])
upper_green = np.array([20,255,255])
#Create database
results = pd.DataFrame()
temp = {}

#Elementos SuperTwisting
xo = 0
TauST = 0.15
#w1 signal
#w2 derivative
w1_a = 0
w2_a = 0
l1 = 2
l2 = 0.5
subsA = np.zeros((2))
subsB = np.zeros((2))
#####Inicia PuertoSerial
Tau=100
Cycles=10

Tic=time.time()
cameraPointsA = np.zeros((3,1))
cameraPointsB = np.zeros((3,1))

for Cycle in range(Cycles):
    FlagCool=0
    for steps in range(Tau):
        if(steps==0):
            ser.write(b'A4000\n')
        if(steps>(Tau/10) and FlagCool==0):
            ser.write(b'B')
            FlagCool=1
        _, frame = cap.read()
        frame = cv2.undistort(frame, mtx, coef, None, newcameramtx)
        frameHsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frameHsv,lower_green,upper_green)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours,hierarchy = cv2.findContours(opening,cv2.RETR_TREE
                                                      ,cv2.CHAIN_APPROX_SIMPLE)
        #cameraPointsA = np.reshape(np.array([imgpoints[0][1][0,0],imgpoints[0][1][0,1],1]),(-1,1))
        cameraPointsA[0,0] = imgpoints[0][1][0,0]
        cameraPointsA[1,0] = imgpoints[0][1][0,1]
        cameraPointsA[2,0] = 1
        realPointsA = R@np.linalg.inv(mtx)@(cameraPointsA)
        if len(contours)==2:
            M1 = contours[0]
            M2 = contours[1]
            P1 = cv2.moments(M1)
            P2 = cv2.moments(M2)
            if P1['m00'] != 0 and P2['m00'] != 0:
                cX1 = int(P1['m10']/P1['m00'])
                cY1 = int(P1['m01']/P1['m00'])
                cX2 = int(P2['m10']/P2['m00'])
                cY2 = int(P2['m01']/P2['m00'])
                cameraPointsA[0,0] = cX1
                cameraPointsA[1,0] = cY1
                cameraPointsA[2,0] = 1
                realPointsA = np.linalg.inv(mtx@R)@(cameraPointsA)
                cameraPointsB[0,0] = cX2
                cameraPointsB[1,0] = cY2
                cameraPointsB[2,0] = 1
                realPointsB = np.linalg.inv(mtx@R)@(cameraPointsB)
                subsA[0] = realPointsA[0,0]/realPointsA[2,0]
                subsA[1] = realPointsA[1,0]/realPointsA[2,0]
                subsB[0] = realPointsB[0,0]/realPointsB[2,0]
                subsB[1] = realPointsB[1,0]/realPointsB[2,0]
                #Find the angle in degrees
                dist = math.degrees(math.atan2((subsB[1]-subsA[1]),(subsB[0]-subsA[0])))
                temp['Angulo']=dist
                temp['Tiempo']=time.time()-Tic
                #SuperTwisting
                e1 = w1_a-dist
                w1 = TauST*(w2_a - l1*(np.sqrt(np.abs(e1)))*np.sign(e1)) + w1_a
                w2 = TauST*(-l2*np.sign(e1)) + w2_a
                w1_a = w1
                w2_a = w2
                temp['Aprox']=w1
                temp['Derivative']=w2
                temp['Norma'] = np.abs(e1)
                results = results.append(temp, ignore_index=True)
            for c in contours:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cX = int(M['m10']/M['m00'])
                    cY = int(M['m01']/M['m00'])
                    cv2.circle(opening, (cX, cY), 5, (0, 0, 0), -1)
                    cv2.putText(opening, "centroid", (cX - 25, cY - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('2',opening)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            time.sleep(1)
            break
    
ser.write(b'B')
results.to_csv('Resultados de Mediciones/SuperTwistingBeta10iter.csv')
#cv2.imwrite('Original.png', frame)
#cv2.imwrite('Mask.png', mask)
#cv2.imwrite('Opening.png', opening)

fig, axes = plt.subplots(nrows=3,ncols=1)

results.plot(x='Tiempo',y=['Angulo','Aprox'],ax=axes[0])
results.plot(x='Tiempo',y=['Derivative'],ax=axes[1])
results.plot(x='Tiempo',y=['Norma'],ax=axes[2])
plt.show()
cap.release()
cv2.destroyAllWindows()
