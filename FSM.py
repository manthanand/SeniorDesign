import time
import sys
import csv

TIME = [45,30,15]

P1 = []
P2 = []
P3 = []

class building:
    def __init__(self,name_item,priority_item,demand_item):
        self.name = name_item
        self.priority = priority_item
        self.demand = demand_item

with open('./SeniorDesign/buildingList.csv', newline='') as csvfile:
    building_reader = csv.reader(csvfile, delimiter=',')
    for row in building_reader:
        row_building = building(name_item=row[0],priority_item=row[1],demand_item=row[2])
        match row_building.priority:
            case 'P1':
                P1.append(row_building)
            case 'P2':
                P2.append(row_building)
            case 'P3':
                P3.append(row_building)
            case _:
                print("Building Omitted: " + row_building.name)

PRIORITIES = [P1, P2, P3]

def greedy_pick():
    # read from current_data.csv
    # read from predicted_data.csv
    for priority_list in PRIORITIES:
        sorted_list = sorted(priority_list, key=lambda building: building.demand, reverse=True)
        for item in sorted_list:
            print(item.name + " " + item.demand)

# make class structure for each building--projected demand
