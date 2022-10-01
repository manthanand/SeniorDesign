# This program collects weather, supply, and demand data and uses that to train a Machine Learning Algorithm
# to make accurate predictions about future supply and demand. This is used to implement a safe rolling blackout
# sequence using a High Level State Machine.
import ML
import FSM
import settings
import serial
from serial.tools import list_ports
import time
import pandas as pd

UPPERBOUND = 60.05
LOWERBOUND = 59.95

BLACKOUT = 1

# ser = serial.Serial(
#     port=list_ports.comports()[len() - 1],
#     baudrate=115200
# )

if __name__ == "__main__":
    # Read from Building CSV and initialize all modules with cluster priorities
    clusters = pd.read_csv(settings.clusterfp).to_dict('records')
    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]
    FSM.init(clusters)

    time15 = time.time() - 901
    runs = 0
    while True:
        FSM.reset
        if (time.time() - time15 >= 900): #NOTE: This assumes that everything below will run in <15 min.
            ML.train() # always train no matter what
            if BLACKOUT: 
                BLACKOUT = FSM.fsm(runs) #returns whether blackout is continuing or not
                runs += 1
                with open('./TheDude/PowerRequirements.csv', "r", encoding="utf-8", errors="ignore") as data:
                    final_line = data.readlines()[-1]
                    final_line = final_line.split(',')
                    BLACKOUT = 0
                    for item in range(1, len(final_line) - 1):
                        if(final_line[item] == '0.0'):
                            BLACKOUT = 1
                            break
                if (not BLACKOUT) : FSM.reset()
            else: #if not in blackout, check frequency for blackout indication
                # freq = ser.read()
                if freq < UPPERBOUND and freq > LOWERBOUND: freq = ser.read() # This blocks until an out-of-bounds frequency is read
                else: BLACKOUT = 1