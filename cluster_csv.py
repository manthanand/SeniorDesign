# '''
# Step 1: read from CSV data
# Step 2: parse data into various columns
# Cluster, Priority, CSV,
# Demand 15 min, Demand 30 min, Demand 45 min, Demand 1 hour
# Supply 15 min, Supply 30 min, Supply 45 min, Supply 1 hour
# Step 3: do the thing
# Step 4: write to CSV
# '''
# import pandas as pd
#
# #reads csv data
# def read_cluster_csv():
#     df = pd.read_csv("Cluster Data.csv", header=0)
#     return df
#
# #returns clusters
# def get_cluster():
#     csv = read_cluster_csv()
#     return csv['Cluster']
#
# #returns priorities
# def get_priority():
#     csv = read_cluster_csv()
#     return csv['Priority']
#
# #returns name of CSV to look at for data
# def get_csv_name():
#     csv = read_cluster_csv()
#     return csv['CSV']
#
# #returns demand for every interval and every cluster
# def get_demand():
#     try:
#         csv = read_cluster_csv()
#         return csv['Demand 15'], csv['Demand 30'], csv['Demand 45'], csv['Demand 60']
#     except:
#         return None
#
# #returns supply for every interval and every cluster
# def get_supply():
#     try:
#         csv = read_cluster_csv()
#         return csv['Supply 15'], csv['Supply 30'], csv['Supply 45'], csv['Supply 60']
#     except:
#         return None
#
# #takes in array of new demand predictions and cluster index?
# def write_demand(demand, time, index):
#     csv = read_cluster_csv()
#     csv['Demand %s' %time][index] = demand
#     csv.to_csv("Cluster Data.csv", index = False)
#
# def write_supply(demand, time, index):
#     csv = read_cluster_csv()
#     csv['Supply %s' %time][index] = demand
#     csv.to_csv("Cluster Data.csv", index = False)