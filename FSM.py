import sys
import csv

# Time each cluster will be on (highest to lowest) 0 indicates always powered
TIME = [0,60,30]

# List of clusters with varying priorities
P1 = []
P2 = []
P3 = []

PRIORITIES = [P1, P2, P3]
class building:
    def __init__(self,name_item,priority_item):
        self.name = name_item
        self.priority = priority_item
        self.demand = TIME[priority_item - 1]
        self.timeon = 0 #Units are in timesteps (i.e if a P3 building was powered for 1 cycle, this would be 1)
    def reset(self):
        self.timeon = 0

def init(clusters):
    for i in clusters: 
        row_building = building(name_item=i,priority_item=clusters[i]['Priority'])
        match row_building.priority:
                case '1':
                    P1.append(row_building)
                case '2':
                    P2.append(row_building)
                case '3':
                    P3.append(row_building)
                case _:
                    print("Building Omitted: " + row_building.name)        

def fsm():
    
def greedy_pick():
    # read from current_data.csv
    # read from predicted_data.csv
    for priority_list in PRIORITIES:
        sorted_list = sorted(priority_list, key=lambda building: building.demand, reverse=True)
        for item in sorted_list:
            print(item.name + " " + item.demand)

# make class structure for each building--projected demand
