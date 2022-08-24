# This library is in charge of collecting data, training the ML algorithm, and making predictions

import tensorflow
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import time

WEATHER_DATA = None
SUPPLY_DATA = None
DEMAND_DATA = None

# This function will collect weather data and store it in "WeatherCSV.csv". 
# This data will then be used to predict supply and demand
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
def collect_data_supplydemand():
    return

# This function trains based on the collected data and writes predictions to "PredictedData.csv"
def train():
    return 
