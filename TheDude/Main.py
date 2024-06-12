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
BLACKOUT = True

TIME_HORIZON = 15 #minutes
START_INDEX = 100
NUMBER_ITERATIONS = 100

def time_test():
    global BLACKOUT
    # Read from Building CSV and initialize all modules with cluster priorities
    clusters = pd.read_csv(settings.clusterfp).to_dict('records')
    ML.init(START_INDEX)
    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]
    FSM.init(clusters)
    for i in range(START_INDEX, NUMBER_ITERATIONS + START_INDEX):
        # wait_input()
        ML.train(i)  # always train no matter what
        with open(settings.powerreqscsv, "r", encoding="utf-8", errors="ignore") as data:
            final_line = data.readlines()[-1]
            final_line = final_line.split(',')
            if final_line[-1] == '0\n' or final_line[-1] == '0.0\n':
                BLACKOUT = True
        with open(settings.outputfp, "r", encoding="utf-8", errors="ignore") as data:
            final_line = data.readlines()[1]
            final_line = final_line.split(',')
            if final_line[-2] == '0' or final_line[-2] == '0.0':
                BLACKOUT = True
        if BLACKOUT:
            on = 0
            off = 0
            FSM.fsm()  # returns whether blackout is continuing or not
            with open(settings.powerreqscsv, "r", encoding="utf-8", errors="ignore") as data:
                final_line = data.readlines()[-1]
                final_line = final_line.split(',')
                BLACKOUT = False
                for item in range(1, len(final_line) - 1):
                    if (final_line[item] == '0.0' or final_line[item] == '0'):
                        BLACKOUT = True
                        off += 1
                    else:
                        on += 1
            print("\n\nON: ", on, "\nOFF: ", off)
        else: FSM.reset()

def wait_input():
    global BLACKOUT
    # while True:
    if not BLACKOUT:
        print("If entering blackout, type in " + BLACKOUT_STR)
        if input() == BLACKOUT_STR:
            BLACKOUT = True

if __name__ == "__main__":
    # time_test()
    # Read from Building CSV and initialize all modules with cluster priorities
    # clusters = pd.read_csv(settings.clusterfp).to_dict('records')
    # Send list of all clusters in dictionary form[{Cluster: Name, Priority: x, CSV: file}, ...]

    t1 = threading.Thread(target=wait_input)
    t1.start()

    # FSM.init(clusters)
    # ML.init()
    time_test()
    time_horizon = time.time() - TIME_HORIZON * 60 + 1
    
    while True:
        if (time.time() - time_horizon >= (TIME_HORIZON * 60) or True):
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