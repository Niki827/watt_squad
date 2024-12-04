# Importing libraries
import sys
import urllib.parse
import requests
import os, csv, json, requests
import glob
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Defining the fetch_data function to fetch data from the EU API. Standard values for lat and lon are Le Wagon Berlin.
def fetch_data(lat, lon):
    BASE_URI = 'https://re.jrc.ec.europa.eu/api/seriescalc' # for the hourly radiation
    startyear = 2021
    endyear = 2023

    params = {
        "lat": lat,
        "lon": lon,
        "startyear": startyear,
        "endyear": endyear,
        "pvcalculation": 1, # If "0" outputs only solar radiation calculations,
            # if "1" outputs the estimation of hourly PV production as well.
        "peakpower": 1, # need if inputting pvcalc as 1
            # our microgrid 'rated output power' is 86.4kWp
        "loss": 14,
        "components": 1, # If "1" outputs beam, diffuse and reflected radiation components.
            #Otherwise, it outputs only global values.
        'optimalangles': 1,
        "outputformat": "json",
        # "browser": # unsure if needed
    }
    result = requests.get(BASE_URI, params=params).json()
    return result

def convert_data(result):
    # Extract the hourly data from the API response
    hourly_data = result['outputs']['hourly']

    # Convert the hourly data into a DataFrame
    df = pd.DataFrame(hourly_data)

    return df

def convert_data_csv(result, name='Le Wagon'):
    # Extract the hourly data from the API response
    hourly_data = result['outputs']['hourly']

   # Convert the hourly data into a DataFrame
    df = pd.DataFrame(hourly_data)

    # Define the path to save the CSV
    csv_path = f'data_predictions/{name}.csv'

    # Save the DataFrame as a CSV
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved at: {csv_path}")

def aggregate_data(df):
    # Convert 'time' to datetime and create 'date' column
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')  # Adjust format if needed
    df['date'] = df['time'].dt.date  # Extract the date for aggregation

    # Convert 'P' to 'pv_output' in kWh and sum daily
    df['pv_output'] = (df['P'] / 1000) * 1  # Convert W to kWh assuming 1-hour intervals
    daily_pv_output = df.groupby('date')['pv_output'].sum().reset_index()

    # Convert irradiance values (Gb(i), Gd(i), Gr(i)) to energy in kWh/m^2 and sum daily
    for col, new_col in {'Gb(i)': 'direct_irradiance', 'Gd(i)': 'diffuse_irradiance', 'Gr(i)': 'reflected_irradiance'}.items():
        df[new_col] = (df[col] * 1) / 1000  # Multiply by time interval (1 hour) and convert to kWh/m^2
        df[new_col] = df[new_col].fillna(0)  # Handle NaN values

    daily_irradiance = df.groupby('date')[['direct_irradiance', 'diffuse_irradiance', 'reflected_irradiance']].sum().reset_index()

    # Average daily values for sun height, temperature, and wind speed
    daily_averages = df.groupby('date')[['H_sun', 'T2m', 'WS10m']].mean().reset_index()
    daily_averages.rename(columns={
        'H_sun': 'sun_height',
        'T2m': 'temp',
        'WS10m': 'wind_speed'
    }, inplace=True)

    # Merge all daily data
    final_daily_data = pd.merge(daily_pv_output, daily_irradiance, on='date')
    final_daily_data = pd.merge(final_daily_data, daily_averages, on='date')

    # Select the required columns
    final_daily_data = final_daily_data[['date', 'pv_output', 'direct_irradiance', 'diffuse_irradiance',
                                         'reflected_irradiance', 'sun_height', 'temp', 'wind_speed']]

    return final_daily_data

# Define the periodicities function
def add_periodicities(df):
    # Transforming column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the day of the year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Total days in the year (account for leap years)
    df['days_in_year'] = df['date'].dt.is_leap_year.apply(lambda x: 366 if x else 365)

    # Compute the cyclic features
    df['year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / df['days_in_year'])
    df['year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / df['days_in_year'])

    # Drop irrelevant columns
    df.drop(columns=['date', 'day_of_year', 'days_in_year'], inplace=True)

    return df

# Define the preprocessing function
def preprocess_data(df, lat, lon):
    # Define the columns to scale
    scale_columns = [
        'direct_irradiance',
        'diffuse_irradiance',
        'reflected_irradiance',
        'sun_height',
        'temp',
        'wind_speed'
    ]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply the scaler only to the specified columns
    df[scale_columns] = scaler.fit_transform(df[scale_columns])

    # Add lat and lon as new columns
    df['lat'] = lat
    df['lon'] = lon

    # Add periodicity features
    df = add_periodicities(df)

    return df
