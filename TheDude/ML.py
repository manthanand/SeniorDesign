# This library is in charge of collecting data, training the ML algorithm, and making predictions
import settings
from bs4 import BeautifulSoup
import requests
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
                                  [f"Horizon {i + 1} supply" for i in range(settings.SUPPLY_TIME_HORIZONS)])
    # WAIT UNTIL NEW DATA IS AVAILABLE IN SUPPLY PORTAL

    # THIS WILL COLLECT WEATHER DATA AND STORE IT IN "WeatherCSV.csv"
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Austin%2C%20TX?unitGroup=us&include=hours&key=5BSYNGCVWT67XMUAJSFWRV4LT&contentType=csv"
    weathercsv = "./TheDude/Supply Data/SolarTrainingData.csv"
    text = "WeatherText.txt"

    try:
        html = requests.get(url).content
        soup = BeautifulSoup(html, 'html.parser')
        with open(text, 'w') as myFile:
            myFile.write(soup.text)
        myFile.close()
        weather_df = pd.read_csv(text)
        os.remove(text)
        print("Generated new weather dataframe")
    except:
        weather_df = pd.read_csv(weathercsv)  # use old one if cant read from url

    # Write Supply data to dataframe
    output.loc[len(output.index)] = (
                ["Supply", "", "",] + 
                ["" for i in range(settings.SUPPLY_TIME_HORIZONS)] +
                supply_ml.compute_prediction(settings.smodelfp + "Supply", read_csv((glob.glob(settings.supplyfp)[0])).head(start))
    )
    # Write Demand data to dataframe
    for i, r in clusters.iterrows():
        data = read_csv((glob.glob(settings.demandfp + r["Cluster"] + "*")[0]), header=0, index_col=0)
        data = data.head(start)
        predictions = demand.compute_prediction(os.path.join(settings.dmodelfp, r["Cluster"]), data)
        output.loc[len(output.index)] = ([r["Cluster"]] + [r["Priority"]] + predictions + [""] + ["" for i in range(settings.SUPPLY_TIME_HORIZONS)])
    output.to_csv(settings.outputfp, encoding='utf-8', index=False)  # Write Dataframe to csv

# train()