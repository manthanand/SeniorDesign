# This library is in charge of collecting data, training the ML algorithm, and making predictions

import tensorflow
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import time

# This function collects the current supply and demand for all clusters and stores them in "CurrentData.csv" every 15 minutes
# Writes predictions to "PredictedData.csv"
def train():
    time1 = time.time()
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
    
    while ((time.time() - time1) > (60 * 15)): pass