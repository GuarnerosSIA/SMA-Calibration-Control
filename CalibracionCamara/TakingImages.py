import cv2
import os

#This properties are not in the videocapture set properties
#Then we need to impement it by the command window
os.system('v4l2-ctl -c focus_auto=0')
os.system('v4l2-ctl -c focus_absolute=0')
os.system('v4l2-ctl -c brightness=133')

#Selection of the camera. Is the only one
cam = cv2.VideoCapture(0)
#Camera parameters

#Resolution
cam.set(3,640) #Width
cam.set(4,480) #Height

cv2.namedWindow("Preview")
#Create a window to preview the images

img_calibration = 0
#Creates the calibration numerator to take images

while True:
    ret, frame = cam.read()#take a picture
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
        print("Calibration Image Adquisition Finished")
        break
    elif k%256 == 32:
        img_name = "Calibration{}.png".format(img_calibration)
        cv2.imwrite(img_name, frame)
        print("{} saved!".format(img_name))
        img_calibration += 1
        

cam.release()

cv2.destroyAllWindows()