# This program collects weather, supply, and demand data and uses that to train a Machine Learning Algorithm
# to make accurate predictions about future supply and demand. This is used to implement a safe rolling blackout
# sequence using a High Level State Machine.
import ML
import FSM
import serial
import threading

UPPERBOUND = 60.05
LOWERBOUND = 59.95

ser = serial.Serial(
    port=serial.tools.list_ports.comports()[len() - 1],
    baudrate=115200
)

if __name__ == "__main__":
    FSM.init()
    t1 = threading.Thread(target = ML.collect_data_weather,) #Constantly collect weather data
    t2 = threading.Thread(target = ML.collect_data_supplydemand,)
    t3 = threading.Thread(target = ML.train())

    t1.start()
    t2.start()
    t3.start()

    while True: 
        freq = ser.read()
        while  freq < UPPERBOUND and freq > LOWERBOUND: # This blocks until an out-of-bounds frequency is read
            freq = ser.read()

        t4 = threading.Thread(target = FSM.greedyPick()) # Start FSM thread 
        t4.start()
        t4.join() # If FSM thread finishes, we are in safe state. Go back to checking if blackout will happen again
    
    t1.join() #This should never be reached
    t2.join()
    t3.join()
