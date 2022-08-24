import csv
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import time

# runs every 15 minutes to generate weather CSV

while(True):
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Austin%2C%20TX?unitGroup=us&include=hours&key=5BSYNGCVWT67XMUAJSFWRV4LT&contentType=csv"
    csv = "WeatherCSV.csv"
    text = "WeatherText.txt"

    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    with open(text, 'w') as myFile:
        myFile.write(soup.text)
    myFile.close()

    dateframe = pd.read_csv(text)
    dateframe.to_csv(csv)
    myFile.close()
    os.remove(text)
    print("Generated new weather csv")
    time.sleep(900) #NOTE: This is because we are getting data for free every 15 minutes.
