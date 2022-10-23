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

def replace_rows():
    for i, r in clusters.iterrows():
        fp = glob.glob(settings.demandfp + r["Cluster"] + "*")[0]
        cluster_data = pd.read_csv(fp, header=0, index_col=0)
        demand = cluster_data["value"].tolist()
        for i, e in reversed(list(enumerate(demand))):
            if ((i >= 1) and ((demand[i] - demand[i - 1]) >= 0)): demand[i] -= demand[i - 1]
        cluster_data["value"] = demand
        cluster_data.to_csv(fp)


if __name__ == "__main__":
    replace_rows()
