import time
import sys
import csv

TIME = [45,30,15]

P1 = []
P2 = []
P3 = []

class building:
    def __init__(self,nameItem,priorityItem,demandItem):
        self.name = nameItem
        self.priority = priorityItem
        self.demand = demandItem

with open('./SeniorDesign/buildingList.csv', newline='') as csvfile:
    buildingReader = csv.reader(csvfile, delimiter=',')
    for row in buildingReader:
        rowBuilding = building(nameItem=row[0],priorityItem=row[1],demandItem=row[2])
        match rowBuilding.priority:
            case 'P1':
                P1.append(rowBuilding)
            case 'P2':
                P2.append(rowBuilding)
            case 'P3':
                P3.append(rowBuilding)
            case _:
                print("Building Omitted: " + rowBuilding.name)

PRIORITIES = [P1, P2, P3]

def greedyPick(projectedSupply):


# make class structure for each building--projected demand
