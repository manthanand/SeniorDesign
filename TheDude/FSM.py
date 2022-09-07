from pickle import FALSE, TRUE
import csv

# Time each cluster will be on (highest to lowest) 0 indicates always powered
TIME = [0,4,2]

powered_clusters = []

# List of clusters with varying priorities
P1 = []
P2 = []
P3 = []

PRIORITIES = [P1, P2, P3]
class cluster:
    def __init__(self,name_item,priority_item):
        self.name = name_item
        self.priority = priority_item
        self.demand_horizon = 0
        self.time_needed = TIME[priority_item - 1]
        self.timeon = 0 #Units are in timesteps (i.e if a P3 cluster was powered for 1 cycle, this would be 1)
    def reset(self):
        self.timeon = 0

def init(clusters):
    for i in clusters: 
        row_building = cluster(name_item=i['Cluster'],priority_item=i['Priority'])
        match row_building.priority:
                case 1:
                    P1.append(row_building)
                case 2:
                    P2.append(row_building)
                case 3:
                    P3.append(row_building)
                case _:
                    print("Building Omitted: " + row_building.name)        

def updated_record(list_of_clusters, cluster_item):
    for old_item in list_of_clusters:
        if cluster_item.name == old_item.name:
            cluster_item.time_on = old_item.time_on
            list_of_clusters.remove(old_item)
            list_of_clusters.append(cluster_item)
            break

    return list_of_clusters

def verify_horizon_compatibility(supply_horizon, demand_horizon):
    list3 = pointwise_subtraction(supply_horizon, demand_horizon)
    for i in list3:
        if i < 0:
            return FALSE
    
    return TRUE

def pointwise_subtraction(list1, list2):
    list3 = []
    for i in range(len(list1)):
        list3.append(list1[i] - list2[i])

    return list3

def pointwise_addition(list1, list2):
    list3 = []
    for i in range(len(list1)):
        list3.append(list1[i] + list2[i])

    return list3

def clear_lower_priorities(level):
    remaining_supply_horizon = [0] * 8
    for item in powered_clusters:
        if item.priority <= level:
            powered_clusters.remove(item)

def cluster_reader(filepath):
    with open(filepath, newline='') as csvfile:
        cluster_reader = csv.reader(csvfile, delimiter=',')
        remaining_supply_horizon = cluster_reader[0]
        for row in cluster_reader:
            row_cluster = cluster(name_item=row[0],priority_item=row[1],demand_set=row[2:5])
            powered_clusters = updated_record(powered_clusters,row_cluster)
            if row_cluster.priority == '1': P1 = updated_record(P1, row_cluster)
            elif row_cluster.priority == '2': P2 = updated_record(P2, row_cluster)
            elif row_cluster.priority == '3': P3 = updated_record(P3, row_cluster)
            else: print("Cluster Omitted: " + row_cluster.name)

    return remaining_supply_horizon

# this function should be called every time step. However, depending on how many time steps constitute each time horizon
# some details on implementation may be slightly different.

# general procedure:
# increase the time_on statistic for currently powered clusters
# read the supply horizons, and update any important records i.e. priority lists and lists of powered clusters
# allocate power to Tier 1 clusters as much as possible
def fsm(time_step):
    for item in powered_clusters:
            item.time_on += 1

    remaining_supply_horizon = cluster_reader('./SeniorDesign/clusterInit.csv')

    '''
    P1 SETUP ---------------------
    '''

    # this stores the total P1 demand and a flag for whether or not the program should add more clusters.
    P1_demand = [0] * 8
    additions_permitted_after_P1 = TRUE

    # calculate the total P1 demand and remove P1 clusters from the list of clusters that should be powered, temporarily
    for item in P1:
        P1_demand = pointwise_addition(P1_demand, item.demand_horizon)
        if item in powered_clusters:
            powered_clusters.remove(item)

    # if you can power all the P1 clusters, do so and continue
    # if not, exit the function and return the list of clusters that should be powered
    if verify_horizon_compatibility(remaining_supply_horizon, P1_demand):
        remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P1_demand)
        for item in P1:
            powered_clusters.append(item)
    else:
        additions_permitted_after_P1 = FALSE
        for item in P1:
            if verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon):
                powered_clusters.append(item)
        return powered_clusters
    
    '''
    P2 and P3 SETUP ---------------------
    '''

    # if time step equals tier 2 + 3 time horizons
    # increment timeon for the previously powered clusters
    # depower all clusters first
    # find a tier 2 cluster to power along the time horizon
    # then find a tier 3 cluster to power along the shorter time horizon
    if time_step % TIME[1] == 0:
        # find the maximum time_on in each priority list. remove each non-P1 item from the list of powered_clusters as well
        tier2standard = 0
        tier3standard = 0
        for item in P2:
            if item.time_on > tier2standard: tier2standard = item.time_on

        for item in P3:
            if item.time_on > tier3standard: tier3standard = item.time_on

        clear_lower_priorities(2)

        ran_out = FALSE
        # while we have not "run out" of power for clusters,
        while not ran_out:
            # iterate through the items in P2. if it can be powered, and has been on for 
            # less than the most-powered cluster that was powered in the last time horizon, power it
            ran_out = TRUE
            for item in P2:
                functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon)
                if functional_item and item.time_on < tier2standard and item not in powered_clusters:
                    powered_clusters.append(item)
                    remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                    ran_out = FALSE
                    break                    # if all items in the list were checked and none could be picked, then ran_out continues to be true

            # ditto for P3. if you couldnt power a P2 cluster, don't power a P3 clusters
            if not ran_out:
                for item in P3:
                    functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon)
                    if functional_item and item.time_on < tier3standard and item not in powered_clusters:
                        powered_clusters.append(item)
                        remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                        ran_out = FALSE
                        break
                

    elif time_step % TIME[2] == 0:
        tier3standard = 0
        power_P3_clusters = TRUE
        # this stores the total demand horizon of the P2 clusters
        P2_used = [0] * 8

        for item in P3:
            if item.time_on > tier3standard: tier3standard = item.time_on

        # two cases:
            # the P2 clusters are sustainable if no additional P3 clusters are powered
            # the P2 clusters are not sustainable

        # if the latter, cut power to all P2 clusters in preparation for severe issues
        for item in powered_clusters:
            if item in P2:
                P2_used = pointwise_addition(P2_used, item.demand_horizon)

        clear_lower_priorities(3)
                
        
        # if the P2 clusters require more power than the supply can provide, shut everything down
        if not verify_horizon_compatibility(remaining_supply_horizon, P2_used):
            power_P3_clusters = FALSE
            clear_lower_priorities(2)
        else:
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P2_used)
        
        # if you have enough power for all the P2 clusters, then you can start powering P3 clusters
        if power_P3_clusters:
            ran_out = FALSE
            # while we have not "run out" of clusters,
            while not ran_out:
                ran_out = TRUE
                # if an item is compatible with the remaining supply horizon, and the cluster wasn't powered before, add the 
                # cluster to the list of powered clusters
                # if there is no such cluster, then break. that's the maximum amount of P3 clusters.
                for item in P3:
                    functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon)
                    if functional_item and item.time_on < tier3standard and item not in powered_clusters:
                        powered_clusters.append(item)
                        remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                        ran_out = FALSE
                        break


    # ensure that current predictions are matching previous predictions
    else:
        # these store the total demand horizon of the P2 and P3 clusters
        P2_used = [0] * 8
        P3_used = [0] * 8

        # accumulate the total power used by P2 and ensure that those clusters can still be powered
        for item in powered_clusters:
            if item in P2: P2_used = pointwise_addition(P2_used, item.demand_horizon)
            if item in P3: P3_used = pointwise_addition(P3_used, item.demand_horizon)

        # if remaining_supply_horizon can't support P2_used, clear all P2 and P3 clusters out of the powered list
        if not verify_horizon_compatibility(remaining_supply_horizon, P2_used):
            clear_lower_priorities(2)
        elif not verify_horizon_compatibility(pointwise_subtraction(remaining_supply_horizon, P2_used), P3_used):
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P2_used)
            clear_lower_priorities(3)
        # if everything can be powered,
        else:
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P2_used)
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P3_used)
        

    return powered_clusters, remaining_supply_horizon
