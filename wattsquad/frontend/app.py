import streamlit as st
import requests

st.title("My Cost Savings App")

# st.write("Give your flexibility degree:")

flexibility = st.number_input("Give your flexibility degree: ", value=0)


# response = requests.get(f'http://127.0.0.1:8000/predict?flexibility_degree={flexibility_degree}'

if st.button("Predict Your Cost Savings"):

    url = 'http://localhost:8000/predict'
    # url = response = requests.get(f'http://127.0.0.1:8000/predict?

    params = {
        "flexibility": flexibility
        }

    # response = requests.get(url, params=params)
    response = requests.get(f'http://127.0.0.1:8000/predict?flexibility_degree={flexibility}')


    cost_pred = response.json()
    cost = cost_pred["df"][0]

    cost_wos = cost["Total Cost Without Shifting"]
    cost_ws = cost['Total Cost With Shifting']
    cost_s = cost['Cost Savings']
    #'Total Cost With Shifting': 7402.1394766528065, 'Cost Savings': 1218.2230921669852}


    # cost = fare_pred["fare"]

    # st.write(f"Your predicted cost savings are NOK{round(fare, 2)}")

    st.write(f"Total Cost Without Shifting: {cost_wos}")
    st.write(f"Total Cost With Shifting: {cost_ws}")
    st.write(f"Cost Savings: {cost_s}")

    # 'Total Cost Without Shifting'


#### em's code:

import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt


"""
SHOWING OUR MICROGRID ON THE MAP
"""

# Coordinates of Langorgen
latitude = 63.418204
longitude = 10.118774

# Create a folium map centered around Langorgen
m = folium.Map(location=[latitude, longitude], zoom_start=5)

# Add a marker for Langorgen
folium.Marker([latitude, longitude], popup="Our microgrid!").add_to(m)

# Display the map in Streamlit
st.title("Welcome to Langørgen!")
st.text("Welcome to Langørgen, Norway. Since the beginning of 2020, our small community (of three houses and one farm) has been using a newly installed microgrid to produce renewable energy.")
st_folium(m, width=700, height=500)


# """
# SHOWING THE 2020 CONSUMPTION AND PRODUCTION DATA IN A GRAPH
# """

st.title("Exploring 2020")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data for demonstration (replace with your actual 'train_data')
train_data = pd.read_csv("raw_data/train.csv")
test_data = pd.read_csv("raw_data/test.csv")

# Renaming columns
train_data.rename(columns={'time': 'timestamp'}, inplace=True)
train_data.rename(columns={'consumption': 'actual_consumption'}, inplace=True)
train_data.rename(columns={'spot_market_price': 'electricity_price'}, inplace=True)

test_data.rename(columns={'time': 'timestamp'}, inplace=True)
test_data.rename(columns={'consumption': 'actual_consumption'}, inplace=True)
test_data.rename(columns={'spot_market_price': 'electricity_price'}, inplace=True)

# Calculating total actual_production
train_data['actual_production'] = train_data['pv_production'] + train_data['wind_production']
test_data['actual_production'] = test_data['pv_production'] + test_data['wind_production']

# Dropping irrelevant columns
train_data = train_data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]
test_data = test_data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]

# Convert 'timestamp' to datetime
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

# Filter data for 2020
train_data_2020 = train_data[train_data['timestamp'].dt.year == 2020]

# Extract month and year for grouping
train_data_2020['month_year'] = train_data_2020['timestamp'].dt.to_period('M')

# Aggregate actual consumption and production by month
monthly_data_2020 = train_data_2020.groupby('month_year')[['actual_consumption', 'actual_production']].sum().reset_index()

# Convert month_year to a datetime format for plotting and format as 'January 2020'
monthly_data_2020['month_year'] = monthly_data_2020['month_year'].dt.to_timestamp()
monthly_data_2020['month_year_str'] = monthly_data_2020['month_year'].dt.strftime('%B %Y')



if st.button("Check our energy consumption in 2020"):
    # Streamlit UI
    st.title('Energy Consumption and Production')

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(monthly_data_2020))  # Numeric x-axis positions for the months
    width = 0.4  # Width of each bar

    # Plot consumption bars
    ax.bar([pos - width / 2 for pos in x], monthly_data_2020['actual_consumption'],
           color='lightblue', width=width, label='Consumption')

    # Plot production bars
    ax.bar([pos + width / 2 for pos in x], monthly_data_2020['actual_production'],
           color='lightgreen', width=width, label='Production')

    # Formatting the plot
    ax.set_title('Monthly Energy Consumption and Production in 2020', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Energy (kWh)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_data_2020['month_year_str'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.text("Our wind turbine and three solar panels have been working hard!")


# """
# ALLOWING THE USER TO PICK A DATE AND SEE PRODUCTION AND CONSUMPTION
# """

# Sample data for demonstration (replace with your actual 'train_data')
train_data = pd.read_csv("raw_data/train.csv")
test_data = pd.read_csv("raw_data/test.csv")

# Renaming columns
train_data.rename(columns={'time': 'timestamp'}, inplace=True)
train_data.rename(columns={'consumption': 'actual_consumption'}, inplace=True)
train_data.rename(columns={'spot_market_price': 'electricity_price'}, inplace=True)

test_data.rename(columns={'time': 'timestamp'}, inplace=True)
test_data.rename(columns={'consumption': 'actual_consumption'}, inplace=True)
test_data.rename(columns={'spot_market_price': 'electricity_price'}, inplace=True)

# Calculating total actual_production
train_data['actual_production'] = train_data['pv_production'] + train_data['wind_production']
test_data['actual_production'] = test_data['pv_production'] + test_data['wind_production']

# # Dropping irrelevant columns
# train_data = train_data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]
# test_data = test_data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]

# Convert 'timestamp' to datetime
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

# Filter data for 2020
train_data_2020 = train_data[train_data['timestamp'].dt.year == 2020]

# Set the min and max date for the year 2020
start_date_2020 = pd.to_datetime('2020-01-01')
end_date_2020 = pd.to_datetime('2020-12-31')

# Date input for selecting the day (only dates within 2020)
selected_date = st.date_input(
    "Select a date",
    min_value=start_date_2020.date(),
    max_value=end_date_2020.date(),
    value=None
)
# # Date input for selecting the day
# selected_date = st.date_input("Select a date", min_value=train_data['timestamp'].min().date(), max_value=train_data['timestamp'].max().date(), value=None)

if selected_date:
    # Filter data for the selected date (for the 24 hours of the day)
    start_of_day = pd.to_datetime(selected_date)
    end_of_day = start_of_day + pd.Timedelta(days=1)

    # Filter the data for the selected date range
    filtered_data = train_data[(train_data['timestamp'] >= start_of_day) & (train_data['timestamp'] < end_of_day)]

    # Aggregate energy consumption and production over the 24 hours of the selected day
    daily_consumption = filtered_data['actual_consumption'].sum()
    daily_production = filtered_data['actual_production'].sum()

    # Display the results
    st.write(f"Total Energy Consumption: {round(daily_consumption, 2)} kWh")
    st.write(f"Total Energy Production: {round(daily_production, 2)} kWh")

    # Plot the hourly data for the selected day
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot hourly consumption
    ax.plot(filtered_data['timestamp'], filtered_data['actual_consumption'], label='Consumption', color='lightblue', marker='o')

    # Plot solar production (pv_production)
    ax.plot(filtered_data['timestamp'], filtered_data['pv_production'], label='Solar Production (PV)', color='orange', marker='o')

    # Plot wind production
    ax.plot(filtered_data['timestamp'], filtered_data['wind_production'], label='Wind Production', color='green', marker='o')

    # Formatting the plot
    ax.set_title(f"Hourly Energy Consumption, Solar, and Wind Production on {selected_date}", fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=14)
    ax.set_ylabel('Energy (kWh)', fontsize=14)

    # # Plot hourly consumption
    # ax.plot(filtered_data['timestamp'], filtered_data['actual_consumption'], label='Consumption', color='lightblue', marker='o')

    # # Plot hourly production
    # ax.plot(filtered_data['timestamp'], filtered_data['actual_production'], label='Production', color='lightgreen', marker='o')

    # Formatting the plot
    ax.set_title(f"Hourly Energy Consumption and Production on {selected_date}", fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=14)
    ax.set_ylabel('Energy (kWh)', fontsize=14)

    # Set x-axis format to HH:MM (e.g., 00:00, 04:00)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.legend(fontsize=12)
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

else:
    st.write("Select a date to see your energy data.")



# """
# "2021 - PREDICTIONS FOR THE NEXT 24H ETC"
# """

st.title("What's next?")
st.text("Our resolutions for 2021 are to optimise our energy consumption and become even more sustainable.")
st.text("The question is: how?")
