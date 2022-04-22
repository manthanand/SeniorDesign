import pandas
import sys
import threading
import matplotlib.pyplot as plt

Prio = 3 #number of priorities
numPrio1 = 2
numPrio2 = 3
numPrio3 = 4
numClusters = numPrio1 + numPrio2 + numPrio3 #number of clusters we are predicting

#current supply is supply given to us over past hour (same for demand)
#predicted supply is supply we think we will get in next hour (same for demand)

def prediction_ml(supplyfile, demandfile, predictfile):
    dfsupply = pandas.readcsv(supplyfile)
    dfdemand = pandas.readcsv(demandfile)
    return

#input data with columns time, future supply, future demand, current supply, current demand, and success based on current parameters
#demand is demand per cluster (demand columns will have list per row for demand per cluster)
# higher priority clusters will have lower indices
def control_ml(readfile, accuracy):
    strpwr = 100 #(storage starts full)
    b_status = [[] for _ in range(numClusters)] # building status's start at all 0
    df = pandas.readcsv(readfile)
    for idx in range(0, len(df.index)):
        #fancy stuff using cost function ()
        yield b_status, df.iloc[index + 1]['current demand']

def simulation_ml(status, demand):
    total = 0
    for i in numClusters:
        total += status[i] * demand[i]
    accuracy = #some stuff
    yield accuracy

if __name__=="__main__":
    #Run prediction algorithms first to get estimated future data (csv inputs) - only for testing full system
    #prediction_ml(sys.argv[1], sys.argv[2], sys.argv[3])
    accuracy = 100 #starts at 100 and decreases
    for status in control_ml(sys.argv[1], accuracy): #can run without predicted data
        accuracy = simulation_ml(status)