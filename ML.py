# This library is in charge of collecting data, training the ML algorithm, and making predictions

import tensorflow
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import time


# This function will collect weather data and store it in "WeatherCSV.csv". 
# This data will then be used to predict supply and demand
# Runs every 15 minutes
def collect_data_weather():
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Austin%2C%20TX?unitGroup=us&include=hours&key=5BSYNGCVWT67XMUAJSFWRV4LT&contentType=csv"
    weathercsv = "WeatherCSV.csv"
    text = "WeatherText.txt"

    while(True):
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
        time.sleep(900) #NOTE: This is because we are getting data for free every 15 minutes.

# This function collects the current supply and demand for all clusters and stores them in "CurrentData.csv"
# New data is generated every 14 minutes according to data provided to us.
def collect_data_supplydemand():
    #
    return

# This function trains based on the collected data and writes predictions to "PredictedData.csv"
# Runs every time new data is received
def train_supply():
    while True:
        time.sleep(900) 

# This function trains based on the collected data and writes predictions to "PredictedData.csv"
# Runs every time new data is received
def train_demand():
    while True:
        time.sleep(900)
