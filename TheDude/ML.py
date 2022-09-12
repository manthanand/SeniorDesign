# This library is in charge of collecting data, training the ML algorithm, and making predictions

import tensorflow
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import demand_ml
import supply_ml
import time

# This is a dictionary where the key is the cluster name and the value is the csv that
# contains the data associated with that key
cluster_csv = {}
output = pd.read_csv('ClusterList.csv')

# This function creates a dictionary that will be used when training the data
def init(clusters):
    for i in clusters: cluster_csv[i['Cluster']] = './CSV Data/' + i['CSV']

# This function collects the current supply and demand for all clusters and stores them in "OutputData.csv" every 15 minutes
# It uses input data from the folder InputData
def train():
    # WAIT UNTIL NEW DATA IS AVAILABLE IN SUPPLY PORTAL
    
    # THIS WILL COLLECT WEATHER DATA AND STORE IT IN "WeatherCSV.csv"
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Austin%2C%20TX?unitGroup=us&include=hours&key=5BSYNGCVWT67XMUAJSFWRV4LT&contentType=csv"
    weathercsv = "WeatherCSV.csv"
    text = "WeatherText.txt"

    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    with open(text, 'w') as myFile:
        myFile.write(soup.text)
    myFile.close()

    dateframe = pd.read_csv(text)
    dateframe.to_csv(weathercsv)
    myFile.close()
    os.remove(text)
    print("Generated new weather csv")

    # convert weather data to solar panel supply
    # convert solar panel supply to power delivered to microgrid

    # TRAIN ON COLLECTED DATA FROM ABOVE FOR SUPPLY AND DEMAND
    # Then write data to OutputData.CSV
    for i in cluster_csv:
        output[0] = [i['Cluster']] + demand_ml.generate_demand_predictions(i['CSV']) + supply_ml.generate_supply_predictions(i['CSV'])
        output.to_csv('OutputData.csv', encoding='utf-8', index=False) 