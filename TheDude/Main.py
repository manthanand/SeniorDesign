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

TIME_HORIZON = 15 #minutes

if __name__ == "__main__":
    # Read from Building CSV and initialize all modules with cluster priorities
    clusters = pd.read_csv(settings.clusterfp).to_dict('records')
    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]
    FSM.init(clusters)
    time_horizon = time.time() - TIME_HORIZON * 60 + 1
    while True:
        if (time.time() - time_horizon >= (TIME_HORIZON * 60)):
            time_horizon = time.time()
            ML.train() # always train no matter what
            if BLACKOUT: 
                BLACKOUT = FSM.fsm() #returns whether blackout is continuing or not
                with open(settings.powerreqscsv, "r", encoding="utf-8", errors="ignore") as data:
                    final_line = data.readlines()[-1]
                    final_line = final_line.split(',')
                    BLACKOUT = 0
                    for item in range(1, len(final_line) - 1):
                        if(final_line[item] == '0.0'):
                            BLACKOUT = 1
                            break
                        else:
                            print("Blackout Ended!")
            else: FSM.reset()