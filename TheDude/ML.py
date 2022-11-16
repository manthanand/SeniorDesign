# This library is in charge of collecting data, training the ML algorithm, and making predictions
import settings
import pandas as pd
import os
import demand
import supply_ml
import glob
from pandas import read_csv

# This is a dictionary where the key is the cluster name and the value is the csv that
# contains the data associated with that key
clusters = pd.read_csv(settings.clusterfp)

def init(start):
    for i, r in clusters.iterrows():
        data = read_csv((glob.glob(settings.demandfp + r["Cluster"] + "*")[0]), header=0, index_col=0)
        data = data.head(start)
        demand.generate_model(data, os.path.join(settings.dmodelfp, r["Cluster"]))
    supply_ml.generate_model(start, settings.smodelfp, os.path.join(settings.smodelfp, settings.supplyfp))

# This function collects the current supply and demand for all clusters and stores them in "OutputData.csv" every 15 minutes
# It uses input data from the folder InputData
def train(start):
    # This reads all data in the csv and clears all existing data in the dataframe (for rewriting)
    output = pd.DataFrame(columns=["Clusters", "Priority", "Current Demand"] + 
                                  [f"Horizon {i + 1} demand" for i in range(settings.DEMAND_TIME_HORIZONS)] +
                                  ["Current Supply"] + 
                                  ["Horizon 1 supply"])
    output.loc[len(output.index)] = (
                ["Supply", "", "",] + 
                ["" for i in range(settings.DEMAND_TIME_HORIZONS)] +
                supply_ml.compute_prediction(settings.smodelfp, read_csv(os.path.join(settings.smodelfp, settings.supplyfp), index_col=0, header=0).head(start))
    )
    # Write Demand data to dataframe
    for i, r in clusters.iterrows():
        data = read_csv((glob.glob(settings.demandfp + r["Cluster"] + "*")[0]), header=0, index_col=0)
        data = data.head(start)
        predictions = demand.compute_prediction(os.path.join(settings.dmodelfp, r["Cluster"]), data)
        output.loc[len(output.index)] = ([r["Cluster"]] + [r["Priority"]] + predictions + [""] + [""])
    output.to_csv(settings.outputfp, encoding='utf-8', index=False)  # Write Dataframe to csv

# train()