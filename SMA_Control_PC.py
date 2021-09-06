
import serial
import time

def run():
    #OpenSerialPort
    ser = serial.Serial()
    ser.port='/dev/ttyACM0'
    ser.baudrate=115200
    ser.timeout=0.1
    ser.open()
    time.sleep(5)
    ser.write(b'3000\n')
    time.sleep(5)
    ser.write(b'B')
    time.sleep(0.5)
    # ser.write(b'A0000\n')
    ser.close()
    

if __name__ =='__main__':
    run()