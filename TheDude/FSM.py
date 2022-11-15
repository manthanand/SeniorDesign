from __future__ import print_function
import pandas as pd
import settings

# CONSTANTS
DEMAND_HORIZON_LENGTH = 5
CSV_ENTRY_LENGTH = 15
BATTERY_CAPACITY = 1000
BATTERY_USE_RATIO_2 = .2
BATTERY_USE_RATIO_3 = .7

# Time each cluster will be on (highest to lowest) 0 indicates always powered
TIME = [0,4,2]
powered_clusters = []
TOTAL_CLUSTERS = []
P1 = []
P2 = []
P3 = []
total_battery = 0
run_count = 0

# List of clusters with varying priorities
PRIORITIES = [P1, P2, P3]
class cluster:
    def __init__(self,name_item,priority_item):
        self.name = name_item
        self.priority = priority_item
        self.demand_horizon = [0] * DEMAND_HORIZON_LENGTH
        self.time_needed = TIME[priority_item - 1]
        self.timeon = 0 #Units are in timesteps (i.e if a P3 cluster was powered for 1 cycle, this would be 1)
    def reset(self):
        self.timeon = 0
    def increment_timeon(self):
        self.timeon = self.timeon + 1
    def update_demand_horizon(self, demand_set):
        self.demand_horizon = demand_set

def init(clusters):
    for item in clusters: 
        row_cluster = cluster(name_item=item.get('Cluster'),priority_item=item.get('Priority'))
        TOTAL_CLUSTERS.append(row_cluster)
        if row_cluster.priority == 1: P1.append(row_cluster)
        elif row_cluster.priority == 2: P2.append(row_cluster)
        elif row_cluster.priority == 3: P3.append(row_cluster)
    PRIORITIES = [P1, P2, P3]

    # initialize the csv file that stores the history of building power      
    headerset = ["timestamp"] + [item.name for item in TOTAL_CLUSTERS] + ["Power Stored"]
    data = [ [0]*len(headerset) for i in range(CSV_ENTRY_LENGTH)]
    pd.DataFrame(data, columns=headerset).to_csv(settings.powerreqscsv, index=False)

def reset():
    run_count = 0
    for item in TOTAL_CLUSTERS:
        item.reset()

def update_record(list_of_list_of_clusters, cluster_name, demand_horizon):
    for priority_list in list_of_list_of_clusters:
        for old_item in priority_list:
            if cluster_name == old_item.name:
                old_item.update_demand_horizon(demand_horizon)
                break

def verify_horizon_compatibility(supply_horizon, demand_horizon, battery_power = 0):
    list3 = pointwise_subtraction(supply_horizon, demand_horizon)
    original_battery = battery_power
    for i in list3:
        if i < 0 and i + battery_power > 0:
            battery_power += i
        elif i + battery_power < 0:
            return False, original_battery
    
    return True, battery_power

def clear_lower_priorities(level, list_of_clusters):
    return [item for item in list_of_clusters if item.priority < level]

def cluster_writer(list_of_clusters, storage_value, timestamp):
    powered_cluster_set = pd.read_csv(settings.powerreqscsv, index_col=False)
    new_entry = [timestamp] + [0 for i in range(len(TOTAL_CLUSTERS))] + [storage_value]

    for item in list_of_clusters:
        spot = TOTAL_CLUSTERS.index(item)
        new_entry[spot+1] = 1

    powered_cluster_set.loc[len(powered_cluster_set)] = new_entry
    powered_cluster_set.drop(powered_cluster_set.index[0], inplace=True)

    powered_cluster_set.to_csv(settings.powerreqscsv, index=False)

def cluster_reader(filepath):
    reading_item = pd.read_csv(filepath)
    remaining_supply_horizon = reading_item.iloc[0][7:].tolist()

    for i in range(1, len(reading_item)):
        cluster_name = reading_item.iloc[i][0]
        demand_horizon=reading_item.iloc[i][2:7].tolist()
        update_record(PRIORITIES, cluster_name, demand_horizon)
        i += 1

    return remaining_supply_horizon

def set_timeon_standard(priority_list):
    standard_value = priority_list[0].timeon
    matched_timeons = True
    for item in priority_list:
        if item.timeon != standard_value: 
            if item.timeon > standard_value: standard_value = item.timeon
            matched_timeons = False

    return -1 if matched_timeons else standard_value

def pointwise_subtraction(list1, list2):
    return [list1[i] - list2[i] for i in range(len(list1))]

def pointwise_addition(list1, list2):
    return [list1[i] + list2[i] for i in range(len(list1))]

# this function should be called every time step. However, depending on how many time steps constitute each time horizon
# some details on implementation may be slightly different.

# general procedure:
# increase the timeon statistic for currently powered clusters
# read the supply horizons, and update any important records i.e. priority lists and lists of powered clusters
# allocate power to Tier 1 clusters as much as possible
def fsm():
    global run_count
    global total_battery
    global powered_clusters
    for item in powered_clusters: item.increment_timeon()

    remaining_supply_horizon = cluster_reader(settings.outputfp)
    '''
    P1 SETUP ---------------------
    '''
    # this stores the total P1 demand and a flag for whether or not the program should add more clusters.
    P1_demand = [0] * DEMAND_HORIZON_LENGTH

    # calculate the total P1 demand and remove P1 clusters from the list of clusters that should be powered, temporarily
    for item in P1: P1_demand = pointwise_addition(P1_demand, item.demand_horizon)

    powered_clusters = [item for item in powered_clusters if item not in P1]

    # if you can power all the P1 clusters, do so and continue
    # if not, exit the function and return the list of clusters that should be powered
    if verify_horizon_compatibility(remaining_supply_horizon, P1_demand, total_battery)[0]:
        total_battery = verify_horizon_compatibility(remaining_supply_horizon, P1_demand, total_battery)[1]
        remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P1_demand)
        if remaining_supply_horizon[0] < 0: remaining_supply_horizon[0] = 0
        for item in P1:
            powered_clusters.append(item)
    else:
        for item in P1:
            if verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[0]:
                total_battery = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[1]
                remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                if remaining_supply_horizon[0] < 0: remaining_supply_horizon[0] = 0
                powered_clusters.append(item)
        cluster_writer(powered_clusters, remaining_supply_horizon[0], run_count)
        run_count += 1
        return 1
    
    '''
    P2 and P3 SETUP ---------------------
    '''
    # if time step equals tier 2 + 3 time horizons
    # increment timeon for the previously powered clusters
    # depower all clusters first
    # find a tier 2 cluster to power along the time horizon
    # then find a tier 3 cluster to power along the shorter time horizon
    if run_count % TIME[1] == 0:
        use_battery_2 = total_battery > BATTERY_USE_RATIO_2 * BATTERY_CAPACITY
        use_battery_3 = total_battery > BATTERY_USE_RATIO_3 * BATTERY_CAPACITY
        # find the maximum timeon in each priority list. remove each non-P1 item from the list of powered_clusters as well
        tier2standard = set_timeon_standard(P2)
        tier3standard = set_timeon_standard(P3)

        powered_clusters = clear_lower_priorities(2, powered_clusters)

        ran_out = False
        # while we have not "run out" of power for clusters,
        while not ran_out:
            # iterate through the items in P2. if it can be powered, and has been on for 
            # less than the most-powered cluster that was powered in the last time horizon, power it
            ran_out = True
            for item in P2:
                # check if the cluster can be powered
                if use_battery_2:
                    functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[0]
                else:
                    functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon)[0]

                if functional_item and (tier2standard == -1 or item.timeon < tier2standard) and item not in powered_clusters:
                    powered_clusters.append(item)
                    if use_battery_2: total_battery = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[1]
                    remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                    ran_out = False
                    break # if all items in the list were checked and none could be picked, then ran_out continues to be true

            # ditto for P3. if you couldnt power a P2 cluster, don't power a P3 clusters
            if not ran_out:
                for item in P3:
                    if use_battery_3:
                        functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[0]
                    else:
                        functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon)[0]
                    if functional_item and (tier3standard == -1 or item.timeon < tier3standard) and item not in powered_clusters:
                        powered_clusters.append(item)
                        if use_battery_3: total_battery = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[1]
                        remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                        ran_out = False
                        break
                
    # if time step equals tier 3 time horizons
    # increment timeon for the previously powered clusters
    # depower all tier 3 clusters
    # find new tier 3 clusters to power along the time horizon
    elif run_count % TIME[2] == 0:
        use_battery_3 = total_battery > BATTERY_USE_RATIO_3 * BATTERY_CAPACITY
        power_P3_clusters = True
        # this stores the total demand horizon of the P2 clusters
        P2_used = [0] * DEMAND_HORIZON_LENGTH
        tier3standard = set_timeon_standard(P3)
        # two cases:
            # the P2 clusters are sustainable if no additional P3 clusters are powered
            # the P2 clusters are not sustainable
                # if so, cut power to all P2 clusters in preparation for severe issues
        P2_count = 0
        for item in powered_clusters:
            if item in P2:
                P2_used = pointwise_addition(P2_used, item.demand_horizon)
                P2_count += 1

        powered_clusters = clear_lower_priorities(3, powered_clusters)
                
        # if the P2 clusters require more power than the supply can provide, shut everything down
        if not verify_horizon_compatibility(remaining_supply_horizon, P2_used, total_battery)[0]:
            power_P3_clusters = False
            powered_clusters = clear_lower_priorities(2, powered_clusters)
        else:
            total_battery = verify_horizon_compatibility(remaining_supply_horizon, P2_used, total_battery)[1]
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P2_used)
            if remaining_supply_horizon[0] < 0: remaining_supply_horizon[0] = 0
        
        i = 0
        # if you have enough power for all the P2 clusters, then you can start powering P3 clusters
        if power_P3_clusters:
            ran_out = False
            # while we have not "run out" of clusters,
            while not ran_out and i < P2_count:
                ran_out = True
                # if an item is compatible with the remaining supply horizon, and the cluster wasn't powered before, add the 
                # cluster to the list of powered clusters
                # if there is no such cluster, then break. that's the maximum amount of P3 clusters.
                for item in P3:
                    if use_battery_3:
                        functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[0]
                    else:
                        functional_item = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon)[0]

                    if functional_item and (tier3standard == -1 or item.timeon < tier3standard) and item not in powered_clusters:
                        powered_clusters.append(item)
                        if use_battery_3: total_battery = verify_horizon_compatibility(remaining_supply_horizon, item.demand_horizon, total_battery)[1]
                        remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, item.demand_horizon)
                        ran_out = False
                        i += 1
                        break

    # ensure that current predictions are matching previous predictions
    else:
        # these store the total demand horizon of the P2 and P3 clusters
        P2_used = [0] * DEMAND_HORIZON_LENGTH
        P3_used = [0] * DEMAND_HORIZON_LENGTH

        # accumulate the total power used by P2 and ensure that those clusters can still be powered
        for item in powered_clusters:
            if item in P2: P2_used = pointwise_addition(P2_used, item.demand_horizon)
            if item in P3: P3_used = pointwise_addition(P3_used, item.demand_horizon)

        # if remaining_supply_horizon can't support P2_used, clear all P2 and P3 clusters out of the powered list
        if not verify_horizon_compatibility(remaining_supply_horizon, P2_used, total_battery)[0]:
            powered_clusters = clear_lower_priorities(2, powered_clusters)
            powered_clusters = clear_lower_priorities(3, powered_clusters)
        elif not verify_horizon_compatibility(pointwise_subtraction(remaining_supply_horizon, P2_used), P3_used, total_battery)[0]:
            total_battery = verify_horizon_compatibility(remaining_supply_horizon, P2_used, total_battery)[1]
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P2_used)
            powered_clusters = clear_lower_priorities(3, powered_clusters)
            if remaining_supply_horizon[0] < 0: remaining_supply_horizon[0] = 0
        # if everything can be powered,
        else:
            total_battery = verify_horizon_compatibility(remaining_supply_horizon, pointwise_addition(P2_used, P3_used), total_battery)[1]
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P2_used)
            remaining_supply_horizon = pointwise_subtraction(remaining_supply_horizon, P3_used)
            if remaining_supply_horizon[0] < 0: remaining_supply_horizon[0] = 0

    blackout_status = 0
    for item in TOTAL_CLUSTERS:
        if item not in powered_clusters:
            blackout_status = 1
            break

    total_battery += remaining_supply_horizon[0]
    if total_battery > BATTERY_CAPACITY: total_battery = BATTERY_CAPACITY
    cluster_writer(powered_clusters, total_battery, run_count)
    run_count += 1
    return blackout_status