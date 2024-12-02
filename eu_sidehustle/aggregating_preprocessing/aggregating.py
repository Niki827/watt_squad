import pandas as pd
import os

# Define the path to the data folder
data_path = 'data/02_fit_for_pandas'

# List of cities (corresponding to CSV filenames)
cities = ['Athens', 'Berlin', 'London', 'Madrid', 'Paris']

# Dictionary to store DataFrames
dfs = {}

# Load all CSVs into the dictionary
for city in cities:
    file_path = os.path.join(data_path, f'{city}.csv')
    dfs[city] = pd.read_csv(file_path)

# Define the aggregation function
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

# Process each DataFrame and store the results back into the dictionary
aggregated_dfs = {}
for city, df in dfs.items():
    aggregated_dfs[city] = aggregate_data(df)

# Save each aggregated DataFrame to a new CSV file
output_path = 'data/03_aggregated/'
os.makedirs(output_path, exist_ok=True)

for city, aggregated_df in aggregated_dfs.items():
    aggregated_df.to_csv(os.path.join(output_path, f'{city}_aggregated.csv'), index=False)

print("All data aggregated and saved successfully!")
