# This program collects weather, supply, and demand data and uses that to train a Machine Learning Algorithm
# to make accurate predictions about future supply and demand. This is used to implement a safe rolling blackout
# sequence using a High Level State Machine.
import ML
import FSM
import serial
from serial.tools import list_ports
import time
import pandas as pd

UPPERBOUND = 60.05
LOWERBOUND = 59.95

BLACKOUT = 0

ser = serial.Serial(
    port=list_ports.comports()[len() - 1],
    baudrate=115200
)

if __name__ == "__main__":
    # Read from Building CSV and initialize all modules with cluster priorities
    clusters = pd.read_csv("ClusterList.csv").to_dict('records')
    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]
    FSM.init(clusters)
    ML.init(clusters)

    while True: 
        if (time.time() - time15 >= 900): #TODO: This assumes that everything below will run in <15 min.
            time15 = time.time()
            ML.train() # always train no matter what
            if BLACKOUT: BLACKOUT = FSM.fsm() #returns whether blackout is continuing or not
            else: #if not in blackout, check frequency for blackout indication
                freq = ser.read()
                if freq < UPPERBOUND and freq > LOWERBOUND: freq = ser.read() # This blocks until an out-of-bounds frequency is read
                else: BLACKOUT = 1