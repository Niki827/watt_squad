import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the path to the data folder
data_path = 'data/03_aggregated'

# List of cities (corresponding to CSV filenames)
cities = ['Athens', 'Berlin', 'London', 'Madrid', 'Paris']

# Dictionary to store DataFrames
dfs = {}

# Load all CSVs into the dictionary
for city in cities:
    file_path = os.path.join(data_path, f'{city}_aggregated.csv')
    dfs[city] = pd.read_csv(file_path)


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
def preprocess_data(df):
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

    # Add periodicity features
    df = add_periodicities(df)

    return df

# Process each DataFrame and store the results back into the dictionary
preprocessed_dfs = {}
for city, df in dfs.items():
    preprocessed_dfs[city] = preprocess_data(df)

# Save each preprocessed DataFrame to a new CSV file
output_path = 'data/04_preprocessed/'
os.makedirs(output_path, exist_ok=True)

for city, preprocessed_df in preprocessed_dfs.items():
    preprocessed_df.to_csv(os.path.join(output_path, f'{city}_preprocessed.csv'), index=False)

print("All data preprocessed and saved successfully!")
