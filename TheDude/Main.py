# This program collects weather, supply, and demand data and uses that to train a Machine Learning Algorithm
# to make accurate predictions about future supply and demand. This is used to implement a safe rolling blackout
# sequence using a High Level State Machine.
import ML
import FSM
import settings
import time
import pandas as pd
import threading

BLACKOUT_STR = "OWIEE"
BLACKOUT = 0

TIME_HORIZON = 15 #minutes
START_INDEX = 100
NUMBER_ITERATIONS = 10


def time_test():
    BLACKOUT = 1
    # Read from Building CSV and initialize all modules with cluster priorities
    clusters = pd.read_csv(settings.clusterfp).to_dict('records')
    ML.init()
    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]
    FSM.init(clusters)
    for i in range(START_INDEX, NUMBER_ITERATIONS + START_INDEX):
        time_horizon = time.time()
        ML.train()  # always train no matter what
        if BLACKOUT:
            FSM.fsm()  # returns whether blackout is continuing or not
            with open(settings.powerreqscsv, "r", encoding="utf-8", errors="ignore") as data:
                final_line = data.readlines()[-1]
                final_line = final_line.split(',')
                BLACKOUT = 0
                for item in range(1, len(final_line) - 1):
                    if (final_line[item] == '0'):
                        BLACKOUT = 1
                        break
        else:
            FSM.reset()

def wait_input():
    global BLACKOUT
    while True:
        if not BLACKOUT:
            print("If entering blackout, type in " + BLACKOUT_STR)
            if input() == BLACKOUT_STR: BLACKOUT = 1

if __name__ == "__main__":
    # time_test()
    # Read from Building CSV and initialize all modules with cluster priorities
    clusters = pd.read_csv(settings.clusterfp).to_dict('records')

    #Start thread to wait for user input for blackout
    t1 = threading.Thread(target=wait_input)
    t1.start()

    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]
    FSM.init(clusters)
    ML.init(100)
    
    time_horizon = time.time() - TIME_HORIZON * 60 + 1
    
    while True:
        if (time.time() - time_horizon >= (TIME_HORIZON * 60)):  
            ML.train() # always train no matter what
            time_horizon = time.time()
            if BLACKOUT: 
                BLACKOUT = FSM.fsm() #returns whether blackout is continuing or not
                with open(settings.powerreqscsv, "r", encoding="utf-8", errors="ignore") as data:
                    final_line = data.readlines()[-1]
                    final_line = final_line.split(',')
                    isblackout = True
                    for item in range(1, len(final_line) - 1):
                        isblackout = isblackout and final_line[item]
                    BLACKOUT = not isblackout #NAND function
            else: FSM.reset()