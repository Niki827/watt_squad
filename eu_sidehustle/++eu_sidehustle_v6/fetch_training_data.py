import os
import pandas as pd
import fetch_training_data_methods as methods
import numpy as np

# Define the paths
city_coordinates_path = "city_coordinates.csv"
output_path = "preprocessed_training_data/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Load city coordinates
city_coordinates = pd.read_csv(city_coordinates_path)

# Loop through each city and process its data
for index, row in city_coordinates.iterrows():
    city_name = row['city']
    lat = row['lat']
    lon = row['lon']

    print(f"Processing data for {city_name} (Lat: {lat}, Lon: {lon})")

    try:
        # Fetch data
        raw_data = methods.fetch_data(lat=lat, lon=lon)

        # Convert data to DataFrame
        df = methods.convert_data(raw_data)

        # Aggregate data
        aggregated_df = methods.aggregate_data(df)

        # Preprocess data
        preprocessed_df = methods.preprocess_data(aggregated_df)

        # Save the preprocessed data as CSV
        output_file = os.path.join(output_path, f"{city_name}_preprocessed.csv")
        preprocessed_df.to_csv(output_file, index=False)

        print(f"Preprocessed data saved for {city_name} at {output_file}")

    except Exception as e:
        print(f"Error processing data for {city_name}: {e}")

print("All cities processed successfully!")
