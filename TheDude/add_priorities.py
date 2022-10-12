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

clusters = pd.read_csv(settings.clusterfp, header=0, index_col=0, squeeze=True)
priorities = []
for i, r in clusters.iterrows():
    priorities.append(CLASSIFICATIONS[r["use_type"]])
clusters["Priority"] = priorities
clusters.to_csv(settings.clusterfp, index=False)