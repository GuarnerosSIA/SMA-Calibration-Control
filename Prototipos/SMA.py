import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import cv2
#Se crea el puerto
#ser = serial.Serial()
#Seleccion el com al que pertence
#ser.port = 'COM6'#com6
#ser.baudrate = 115200
#ser.timeout = 0.5
#ser.open()
# Iniciar captura de video
cap = cv2.VideoCapture(0)
cap.set(11,0)
cap.set(3,1280) #Width
cap.set(4,960) #Height
#Kernel
kernel = np.ones((80,80),np.uint8)
###definir los limites
lower_orange = np.array([0,0,180])
upper_orange = np.array([100,255,255])
lower_green = np.array([0,130,0])
upper_green = np.array([80,255,255])
### Estimación de la distancia inicial
mm = 30 #Distancia de marcadores en milimetros
L=0
while(L==0):
    _, frame = cap.read()
    mask = cv2.inRange(frame,lower_green,upper_green)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE
                                                  ,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)==2):
        L = 1
        M1 = contours[0]
        M2 = contours[1]
        P1 = cv2.moments(M1)
        P2 = cv2.moments(M2)
        if P1['m00'] != 0 and P2['m00'] != 0:
            cX1 = int(P1['m10']/P1['m00'])
            cY1 = int(P1['m01']/P1['m00'])
            cX2 = int(P2['m10']/P2['m00'])
            cY2 = int(P2['m01']/P2['m00'])
            subs = np.array([cX1 - cX2,cY1 - cY2])
            dist = np.sqrt(subs[0]*subs[0] + subs[1]*subs[1])
            print(dist)
            alfa = mm/dist
            print(alfa)
########################################################################################################

### Variables valores
tiempo = list()
elong = list()
elong2= list()
ErrA = 0
Contador = 1
t = time.time()
for XXX in range(2500):
    _, frame = cap.read()
        #se crea una mascara
    mask = cv2.inRange(frame,lower_orange,upper_orange)
        #Operaciones morfologicas
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #Calcular los contronos que extisten
    contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE
                                                  ,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2:
                            
        M1 = contours[0]
        M2 = contours[1]
        P1 = cv2.moments(M1)
        P2 = cv2.moments(M2)
        if P1['m00'] != 0 and P2['m00'] != 0:
            cX1 = int(P1['m10']/P1['m00'])
            cY1 = int(P1['m01']/P1['m00'])
            cX2 = int(P2['m10']/P2['m00'])
            cY2 = int(P2['m01']/P2['m00'])
            subs = np.array([cX1 - cX2,cY1 - cY2])
            dist = np.sqrt(subs[0]*subs[0] + subs[1]*subs[1])
            dist = dist*alfa
            distH = np.cos(Contador*np.pi/500.)*5 + 105
            elong2.append(distH)
            Err = dist - distH
            dErr = (Err-ErrA)/0.01
            ErrA = Err
            Err = 85*Err + 15*dErr
            Err = int(Err)
            if Err > 500:
                Err = 500
            elif Err<=0:
                Err = 0
            Err += 10000
            Err = str(Err)
            print(Err)
            Err = 'C'+Err[1:5]
            Err = Err.encode('utf-8')
            A = ser.write(Err)
            elong.append(dist)
            tiempo.append(time.time()-t)
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
                cv2.circle(closing, (cX, cY), 5, (0, 0, 0), -1)
                cv2.putText(closing, "centroid", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('frame',closing)
    Contador += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ser.write(b'C0000')
        time.sleep(1)
        break


# When everything done, release the capture
ser.write(b'C0000')
ser.write(b'C0000')
ser.write(b'C0000')
ser.write(b'C0000')
time.sleep(1)
ser.close()
ser.open()
ser.close()
cap.release()
cv2.destroyAllWindows()
cv2.imshow('frame',frame)
##plt.subplot(211)
##plt.plot(elong)
##plt.subplot(212)
A = np.array(elong)
B = np.array(elong2)
C = A-B
C = np.abs(C)

#Almacenar variables
np.save('DISTANCIAControlPD4',elong)
np.save('TRAYECTORIAControlPD4',elong2)
np.save('NORMAControlPD4',C)
np.save('TIEMPOControlPD4',tiempo)

plt.figure()
plt.subplot(211)
plt.plot(tiempo,elong,'k-')
plt.plot(tiempo,elong2,'r-')
plt.subplot(212)
plt.plot(tiempo,C,'r-')
plt.show()