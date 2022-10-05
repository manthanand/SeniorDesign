# This library is in charge of collecting data, training the ML algorithm, and making predictions

import settings
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import demand
import supply_ml
import glob
import multiprocessing

# This is a dictionary where the key is the cluster name and the value is the csv that
# contains the data associated with that key
clusters = pd.read_csv(settings.clusterfp)


# This function collects the current supply and demand for all clusters and stores them in "OutputData.csv" every 15 minutes
# It uses input data from the folder InputData
def train():
    # This reads all data in the csv and clears all existing data in the dataframe (for rewriting)
    output = pd.DataFrame(columns=["Clusters", "Priority", "Current Demand",
                                   "15 Min Demand", "30 Min Demand", "45 Min Demand", "60 Min Demand",
                                   "Current Supply", "60 Min Supply"])
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
                ["Supply", "", "", "", "", "", ""] + supply_ml.generate_supply_predictions(weather_df))
    # Write Demand data to dataframe
    for i, r in clusters.iterrows():
        print(glob.glob(settings.demandfp + r["Cluster"] + "*"))
        file = (glob.glob(settings.demandfp + r["Cluster"] + "*")[0])
        predictions = demand.generate_demand_predictions(file)
        output.loc[len(output.index)] = ([r["Cluster"]] + [r["Priority"]] + predictions + ["", ""])
    output.to_csv(settings.outputfp, encoding='utf-8', index=False)  # Write Dataframe to csv
    