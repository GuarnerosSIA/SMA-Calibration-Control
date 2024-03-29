import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import cv2

cap = cv2.VideoCapture(0)
time.sleep(1)
cap.set(11,0)
cap.set(3,640) #Width
cap.set(4,480) #Height
cap.set(cv2.CAP_PROP_BRIGHTNESS,0)
#IMPORTANTE. SIEMPRE QUITAR EL AUTOFOCUS v4l2-ctl -c focus_auto=0
 
#####Obtener posicion del tablero
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*24.3#milimetros
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

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
        cv2.imshow('img', frame)
        cv2.waitKey(1)
        FLAG = False






########

kernel = np.ones((5,5),np.uint8)
###definir los limites

lower_green = np.array([40,100,100])
upper_green = np.array([65,255,255])

#Create database
results = pd.DataFrame()
temp = {}



for steps in range(100):
    _, frame = cap.read()
    frame = cv2.undistort(frame, mtx, coef, None, newcameramtx)
    frameHsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frameHsv,lower_green,upper_green)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours,hierarchy = cv2.findContours(opening,cv2.RETR_TREE
                                                  ,cv2.CHAIN_APPROX_SIMPLE)
    cameraPointsA = np.reshape(np.array([imgpoints[0][1][0,0],imgpoints[0][1][0,1],1]),(-1,1))
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
            cameraPointsA = np.reshape(np.array([cX1,cY1,1]),(-1,1))
            realPointsA = np.linalg.inv(mtx@R)@(cameraPointsA)
            cameraPointsB = np.reshape(np.array([cX2,cY2,1]),(-1,1))
            realPointsB = np.linalg.inv(mtx@R)@(cameraPointsB)
            subs = np.array([realPointsA[0,0]/realPointsA[2,0] - realPointsB[0,0]/realPointsB[2,0],realPointsA[1,0]/realPointsA[2,0] - realPointsB[1,0]/realPointsB[2,0]])
            #subs = np.array([cX1 - cX2,cY1 - cY2])
            dist = np.sqrt(subs[0]*subs[0] + subs[1]*subs[1])
            temp['Distancia']=dist
            results = results.append(temp, ignore_index=True)
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
                cv2.circle(opening, (cX, cY), 5, (0, 0, 0), -1)
                cv2.putText(opening, "centroid", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('0',frame)
    cv2.imshow('1',mask)
    cv2.imshow('2',opening)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        time.sleep(1)
        break


results.to_csv('Results.csv')
cv2.imwrite('Original.png', frame)
cv2.imwrite('Mask.png', mask)
cv2.imwrite('Opening.png', opening)
results.plot()
plt.show()
cap.release()
cv2.destroyAllWindows()