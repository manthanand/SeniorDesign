import pandas as pd
import glob
import settings
import os

CLASSIFICATIONS = {
    "Classroom": 2,
    "Computer": 2,
    "Exhibition": 3,
    "Food Service": 1,
    "Health Care": 1,
    "Laboratory": 1,
    "Library": 3,
    "Lodging": 2,
    "Office": 3,
    "Public Assembly": 3,
    "Service": 2
}

clusters = pd.read_csv(settings.clusterfp, header=0, index_col=False)

def add_priorities():
    priorities = []
    for i, r in clusters.iterrows():
        priorities.append(CLASSIFICATIONS[r["use_type"]])
    clusters["Priority"] = priorities
    clusters.to_csv(settings.clusterfp, index=False)

def add_skips():
    for i, r in clusters.iterrows():
        fp = glob.glob(settings.demandfp + r["Cluster"] + "*")[0]
        print(fp)
        cluster_data = pd.read_csv(fp, header=0, index_col=0)
        demand = cluster_data["value"].tolist()
        old_length = len(demand)
        prev = 1 #flag for whether previous element was 0
        for i, e in reversed(list(enumerate(demand))): #traverse through row from end
            if (i >= 1): 
                if ((demand[i - 1] < 0) or (demand[i] == 0)): #if next value is negative, don't subtract
                    pass
                elif ((demand[i] - demand[i - 1]) > 0): 
                    demand[i] -= demand[i - 1]
                    prev = 1
                elif ((demand[i] - demand[i - 1]) == 0): #if current - previous = 0
                    if (prev == 0): #if this is second time it was 0, set all to -1
                        demand[i:i+3] = [-1,-1,-1]
                    else: #else let it be 0
                        demand[i] -= demand[i - 1]
                        prev = 0
                else: #if negative, keep demand[i] the same
                    pass
        cluster_data["value"] = demand[0 : cluster_data.shape[0]]
        cluster_data.to_csv(fp)    

def skip_zeros():
    for i, r in clusters.iterrows():
        fp = glob.glob(settings.demandfp + r["Cluster"] + "*")[0]
        print(fp)
        cluster_data = pd.read_csv(fp, header=0, index_col=0)
        demand = cluster_data["value"].tolist()
        for i in range(0, len(demand), 24): demand[i] = -1
        cluster_data["value"] = demand[0 : cluster_data.shape[0]]
        cluster_data.to_csv(fp)

def replace_negatives():
    for i, r in clusters.iterrows():
        fp = glob.glob(settings.demandfp + r["Cluster"] + "*")[0]
        print(fp)
        cluster_data = pd.read_csv(fp, header=0, index_col=0)
        demand = cluster_data["value"].tolist()
        for i in range(len(demand)):
            if ((demand[i] == -1) and (i%24 == 0)): #This is if the data is at 6am
                demand[i] = (demand[i-1] + demand[i+1]) / 2
            elif (demand[i] == -1):
                demand[i] = (demand[i-24] + demand[i+24]) / 2
        cluster_data["value"] = demand
        cluster_data.to_csv(fp)

if __name__ == "__main__":
    skip_zeros()
