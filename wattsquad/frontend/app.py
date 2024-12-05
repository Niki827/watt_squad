import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from io import StringIO
import plotly.express as px

st.title("Welcome to Langørgen!")
st.image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARIAAAC4CAMAAAAYGZMtAAAAllBMVEXuLS8hL2P////zLjCcHh8PIVsYKWGqqq5fYnP+7+r7083vNiztEhYWIVYiK1br6+vJycv+7OTycGXuJCjtCBCnp6cWH0EdI0FmExSnm5aEhIWfSUKcGBr/8++tra1vcHKlioebCg0NFTgVGzibm5ruGwvuLyWcIhw+P0tVWGslL2ApMl8JFTyOjpP83dfsAADxZlsdHzcBQ3VkAAACTElEQVR4nO3b107DMABGYbeEUfZoSgcEWsre7/9yICQoykklItsSiPNdW/nVc1nLYVhEm47KT9V4NmhyMV8cmZzET2Y0DMVKtH6v+2V/p9Nkd29x5PAofjKj4j1JiNU/aJkkejEjk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk4BJwCRgEjAJmARMAiYBk8B7kuutaJctk1zFT2Z0HUanvWjdVkkSDGZ0OgplN6kfJPnlSpPUmQRMAiYBk4BJwCRgEihDlfaDfz9JFW42krq9a0xyn3Ylp5swW0ursUink3glo1kYLPkN/9bAJHUmAZOAScAkYBIwCZgETAImAZOAScAkYBIwCZgETAImAZPAIDysJrXs7+i0Kzk9hLP1pB6XXFrM085kdObVVl3lBWidd8JgEjAJmARMAiYBk4BJoAznm/G+ffAHSZ4SLGZ0Hqbb0Z7bPT556cdPZjT1iVKNr7bAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEnAJGASMAmYBEwCJgGTgEngI0m0fq9lkvjJjIowLKJNR+Wnanxx3OR1vjgyOYmfzGj4BmpfD7IHYkXJAAAAAElFTkSuQmCC", width=75)
st.write("""

            Hei! We live in **Langørgen, Norway**.

            Since the beginning of 2020, our small community of three houses :house_buildings: and one farm :farmer: has been using a newly installed microgrid to produce renewable energy.



            """)

st.divider()


tabs = st.tabs(["Home", "Our Microgrid in 2020", "Our Microgrid in 2021", "Solar Energy Production across the Globe"])

# Tab: Home




# """
# SHOWING OUR MICROGRID ON THE MAP
# """
with tabs[0]:

    # Coordinates of Langorgen
    latitude = 63.418204
    longitude = 10.118774

    # Create a folium map centered around Langorgen
    m = folium.Map(location=[latitude, longitude], color="red", zoom_start=18)

    # Add a marker for Langorgen
    folium.Marker([latitude, longitude], popup="Our microgrid!").add_to(m)

    # Display the map in Streamlit


    st.subheader("A microgrid?")

    st.write("""
            A microgrid is a small energy system with a local source of supply that is able to function independently of the centralized electricty grid. :battery:

            Our microgrid is comprised of **three solar panels** :sun_with_face: and a **wind turbine**. :wind_blowing_face:


            """)

    st_folium(m, width=700, height=500)

    st.image("https://www.remote-euproject.eu/remote18/rem18-cont/uploads/2021/11/rye-byneset-750x350.jpg")



    st.write("""
            Over the past year of using the microgrid, we've collected data on our **energy consumption** and our **renewable energy production**.

            Take a look at the next pages to see what happened in the last year!

            """)

# st.divider()

with tabs[1]:
    # """
    # SHOWING THE 2020 CONSUMPTION AND PRODUCTION DATA IN A GRAPH
    # """


    st.image("https://cdn-icons-png.flaticon.com/512/3967/3967606.png", width=100)

    st.subheader("Our energy consumption and production in 2020")

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


    st.write("""
            Click the button below to explore our **energy consumption compared to our production** through our sustainable microgrid in the last year.  :bar_chart:
            """)
    if st.button("Check energy statistics for 2020"):
        # Streamlit UI
        # st.subheader('Energy Consumption and Production')

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

        st.write("Our wind turbine and three solar panels have been working hard! :muscle:")
    st.divider()

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

    st.write("""


            """)

    # st.subheader("Interested in the details?")
    st.write("""
            You can even pick a **specific date** below to see the energy consumption and production for those 24 hours. :clock2:

            """)

    # Date input for selecting the day (only dates within 2020)
    selected_date = st.date_input(
        "Select a date",
        min_value=start_date_2020.date(),
        max_value=end_date_2020.date(),
        value=start_date_2020.date()
    )
    # # Date input for selecting the day
    # selected_date = st.date_input("Select a date", min_value=train_data['timestamp'].min().date(), max_value=train_data['timestamp'].max().date(), value=None)

    if selected_date:
        if st.button("Check energy statistics for this day!"):
            # Filter data for the selected date (for the 24 hours of the day)
            start_of_day = pd.to_datetime(selected_date)
            end_of_day = start_of_day + pd.Timedelta(days=1)

            # Filter the data for the selected date range
            filtered_data = train_data[(train_data['timestamp'] >= start_of_day) & (train_data['timestamp'] < end_of_day)]

            # Aggregate energy consumption and production over the 24 hours of the selected day
            daily_consumption = filtered_data['actual_consumption'].sum()
            daily_production = filtered_data['actual_production'].sum()

            # Display the results
            st.write(f"**Total Energy Consumption:** {round(daily_consumption, 2)} kWh")
            st.write(f"**Total Energy Production:** {round(daily_production, 2)} kWh")

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
            ax.set_title(f"Energy Consumption and Production on {selected_date}", fontsize=16)
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

    st.divider()

    st.subheader("Our battery usage")
    st.write("""
            Oh, we nearly forgot to mention! In addition to our solar panels and wind turbine, our microgrid includes our very own **battery**. :battery:

            Click below to see how our battery has performed in the last year.

            INPUT BATTERY DESCRIPTIVES
            """)

    # Input fields for user to provide input
    battery_capacity = st.slider(
        "Select battery capacity (kWh):",
        min_value=100,
        max_value=1000,
        value=500
    )
    electricity_price_share = st.slider(
        "Enter electricity price share (%):",
        min_value=0,
        max_value=100,
        value=50
    )

    # Button to trigger API call
    if st.button("Check battery"):
        # Making a GET request to the FastAPI endpoint
        url = "https://mvp3-1071061957527.europe-west1.run.app/battery_product"  # Replace with your actual endpoint URL
        params = {
            "battery_capacity": battery_capacity,
            "electricity_price_share": (electricity_price_share)/100,
        }
        response = requests.get(url, params=params)

        # Check the response
        if response.status_code == 200:
            data = response.json()

            # Display the message
            st.success(data["message"])

            # Display the DataFrame
            df = pd.DataFrame(data["df"])
            st.dataframe(df)

            # Display other results
            st.metric("Electricity Sold (kWh)", data["electricity_sold_kwH"])
            st.metric("Electricity Sold (NOK)", data["electricity_sold_NOK"])
            st.metric("Electricity Bought (NOK)", data["electricity_bought_NOK"])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

with tabs[2]:
    # """
    # "2021 - PREDICTIONS FOR THE NEXT 24H ETC"
    # """
    # st.divider()

    st.title("Well, what's next?")

    st.write("""
            Our resolutions for 2021 are to **optimise our energy consumption** and become even more sustainable.
            """)

    st.subheader("How can we make this happen?")


    st.write("Well, we've learned some lessons from 2020:")



    st.markdown("- As you can see from our battery usage, in summer, the battery is often charged to 100\%. Any excess energy we product through our microgrid while our battery is full goes to waste.")
    st.markdown("- Meanwhile, in winter our battery charge frequently sits at 0%, meaning we need to purchase energy from the centralized grid if the weather doesn't play along.")

    st.subheader("What can we do about it?")
    st.write("""
            We've thought of two solutions:
            """)

    st.markdown("- **We consume energy when we produce it.** If we time our energy consumption with our production via the microgrid, we reduce our need to purchase energy, thereby minimizing our costs and maximising our sustainability.")
    st.markdown("- **We sell our excess energy from production in the summer to finance our energy purchases in the winter.** Here we would only sell what we can't store in our own battery. Potentially, we can achieve a profit and deliver more green energy to the grid.")

    st.divider()

    # """
    # cost savings model
    # """

    # st.title("Optimising sustainability for 2021")

    st.subheader("Shifting our consumption throughout the day")

    st.write("""
            We've realised that if we can **shift our consumption** throughout the day to **match when we have the highest energy production** through our microgrid, we can minimize the amount of enery we need to purchase from the centralized grid and save money.

            :money_mouth_face:

            Thanks to all the data we collected in 2020, we can accurately **predict** our energy consumption and our energy production (based on the weather forecast!) for the **next 24 hours**.

            Let's try it. **How much of your consumption can you shift around today?**
            """)
    # Slider for flexibility degree
    flexibility_degree = st.slider(
        "Select Flexibility Degree (%)",
        min_value=0,
        max_value=100,
        value=0,  # default value
    )

    # Button to fetch and display the graph
    if st.button("Get cost savings and show optimum energy usage"):
        #st.status("label", *, expanded=False, state="running") ####


        # Call the API
        try:
            response = requests.get(
                "https://mvp3-1071061957527.europe-west1.run.app/predict",  # Replace with your API endpoint
                params={"flexibility_degree": (flexibility_degree/100)}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])

                st.write(f"Costs without shifting consumption: NOK {round(result['costs_no_shift'], 2)}")
                st.write(f"Costs with shifted consumption: NOK {round(result['costs_with_shift'], 2)}")

                cost_saved = round(result['costs_no_shift'], 2) - round(result['costs_with_shift'], 2)
                st.write(f"Money saved today: NOK {round(cost_saved, 2)}")

                st.divider()

                # Load the dataframe
                if "df" in result:

                    data = pd.DataFrame(result["df"])
                    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Ensure timestamp is in datetime format

                    # Convert timestamp to number of hours since the first timestamp
                    data['hour'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600

                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot forecasted consumption and production
                    ax.plot(
                        data['hour'],
                        data['adjusted_forecasted_consumption'],
                        label='Forecasted Consumption',
                        color='#CC6666',
                        linestyle='--'
                    )
                    ax.plot(
                        data['hour'],
                        data['adjusted_forecasted_production'],
                        label='Forecasted Production',
                        color='#6699CC',
                        linestyle='-',
                        linewidth=3
                    )

                    # Plot shifted consumption
                    ax.plot(
                        data['hour'],
                        data['shifted_consumption'],
                        label='Shifted Consumption',
                        color='#66CC99',
                        linestyle='-',
                        linewidth=3
                    )

                    # Highlight areas where consumption shifted
                    shifted_consumption_less = data['shifted_consumption'] <= data['adjusted_forecasted_consumption']
                    ax.fill_between(
                        data['hour'],
                        data['adjusted_forecasted_consumption'],
                        data['shifted_consumption'],
                        where=shifted_consumption_less,
                        color='#CC6666',
                        alpha=0.1
                        # label="Mom: TURN YOUR LIGHTS OFF!!!"
                    )

                    shifted_consumption_more = data['shifted_consumption'] >= data['adjusted_forecasted_consumption']
                    ax.fill_between(
                        data['hour'],
                        data['shifted_consumption'],
                        data['adjusted_forecasted_consumption'],
                        where=shifted_consumption_more,
                        color='#66CC99',
                        alpha=0.1
                        # label="You: I LEAVE MY LIGHTS ON!!!"
                    )

                    # Adding speaking bubbles using text
                    ax.text(x=12, y=60, s="Fu** it, I LEAVE MY LIGHTS ON!!!", fontsize=10, color='black', ha='center',
                            bbox=dict(facecolor='#66CC99', edgecolor='black', boxstyle="round,pad=0.5", alpha=0.7))
                    # Line from the green shaded area to the text box
                    ax.annotate('', xy=(10, 50), xytext=(12, 58),
                                arrowprops=dict(arrowstyle="->", color='black', lw=1))

                    ax.text(x=24, y=80, s="TURN YOUR LIGHTS OFF!!!", fontsize=10, color='black', ha='center',
                            bbox=dict(facecolor='#CC6666', edgecolor='black', boxstyle="round,pad=0.5", alpha=0.7))
                    # Line from the green shaded area to the text box
                    ax.annotate('', xy=(19, 70), xytext=(23, 78),
                                arrowprops=dict(arrowstyle="->", color='black', lw=1))

                    # Customize axes appearance
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                    # Add labels and title
                    ax.set_title("Forecasted Consumption and Production with Consumption Shift", fontsize=14)
                    ax.set_xlabel("Hour", fontsize=12)
                    ax.set_ylabel("Energy (kWh)", fontsize=12)
                    ax.legend()

                    # Fine-tune plot layout
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(False)
                    fig.tight_layout()

                    # Render plot in Streamlit
                    st.pyplot(fig)
                    # # Plotting
                    # fig, ax = plt.subplots(figsize=(10, 6))

                    # # Plot forecasted consumption and production
                    # ax.plot(
                    #     data['hour'],
                    #     data['adjusted_forecasted_consumption'],
                    #     label='Forecasted Consumption',
                    #     color='#CC6666',
                    #     linestyle='--'
                    # )
                    # ax.plot(
                    #     data['hour'],
                    #     data['adjusted_forecasted_production'],
                    #     label='Forecasted Production',
                    #     color='#6699CC',
                    #     linestyle='-',
                    #     linewidth=3
                    # )

                    # # Plot shifted consumption
                    # ax.plot(
                    #     data['hour'],
                    #     data['shifted_consumption'],
                    #     label='Shifted Consumption',
                    #     color='#66CC99',
                    #     linestyle='-',
                    #     linewidth=3
                    # )

                    # # Highlight areas where consumption shifted
                    # shifted_consumption_less = data['shifted_consumption'] <= data['adjusted_forecasted_consumption']
                    # ax.fill_between(
                    #     data['hour'],
                    #     data['adjusted_forecasted_consumption'],
                    #     data['shifted_consumption'],
                    #     where=shifted_consumption_less,
                    #     color='#CC6666',
                    #     alpha=0.1,
                    #     label="Mom: TURN YOUR LIGHTS OFF!!!"
                    # )

                    # shifted_consumption_more = data['shifted_consumption'] >= data['adjusted_forecasted_consumption']
                    # ax.fill_between(
                    #     data['hour'],
                    #     data['shifted_consumption'],
                    #     data['adjusted_forecasted_consumption'],
                    #     where=shifted_consumption_more,
                    #     color='#66CC99',
                    #     alpha=0.1,
                    #     label="You: I LEAVE MY LIGHTS ON!!!"
                    # )

                    # # Customize axes appearance
                    # ax.spines['top'].set_visible(False)
                    # ax.spines['right'].set_visible(False)
                    # ax.spines['left'].set_visible(False)
                    # ax.spines['bottom'].set_visible(False)

                    # # Add labels and title
                    # ax.set_title("Forecasted Consumption and Production with Consumption Shift", fontsize=14)
                    # ax.set_xlabel("Hour", fontsize=12)
                    # ax.set_ylabel("Energy (kWh)", fontsize=12)
                    # ax.legend()

                    # # Fine-tune plot layout
                    # ax.tick_params(axis='x', rotation=45)
                    # ax.grid(False)
                    # fig.tight_layout()

                    # # Render plot in Streamlit
                    # st.pyplot(fig)
                #     # Convert data to DataFrame
                #     data = pd.DataFrame(result["df"])
                #     data['timestamp'] = pd.to_datetime(data['timestamp'])  # Ensure timestamp is in datetime format

                #     # Convert timestamp to number of hours since the first timestamp
                #     data['hour'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600

                #     # Plotting
                #     fig, ax = plt.subplots(figsize=(10, 6))

                #     # Plot forecasted consumption and production
                #     ax.plot(data['hour'], data['adjusted_forecasted_consumption'], label='Forecasted Consumption', color='red', linestyle='--', marker='o')
                #     ax.plot(data['hour'], data['adjusted_forecasted_production'], label='Forecasted Production', color='lightgreen', linestyle='-')

                #     # Plot shifted consumption
                #     ax.plot(data['hour'], data['shifted_consumption'], label='Shifted Consumption', color='blue', linestyle='-', marker='o')

                #     # Highlight the areas where consumption was shifted
                #     shifted_consumption = data['shifted_consumption'] < data['adjusted_forecasted_consumption']
                #     ax.fill_between(data['hour'], data['adjusted_forecasted_consumption'], data['shifted_consumption'], where=shifted_consumption, color='blue', alpha=0.3, label="Shifted Area")

                #     # Add labels, legend, and title
                #     ax.set_title("Forecasted Consumption and Production with Consumption Shift")
                #     ax.set_xlabel("Hour")
                #     ax.set_ylabel("Energy (kWh)")
                #     ax.legend()
                #     ax.grid(True)

                #     # Render plot in Streamlit
                #     st.pyplot(fig)
                # else:
                #     st.error("No data available to plot.")
            else:
                st.error(f"Error from API: {response.status_code}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    st.divider()

    st.image("https://www.shutterstock.com/image-photo/day-night-time-change-concept-600nw-2326233419.jpg", width=705)

# st.divider()

with tabs[3]:
    # """
    # using Niki's EU rnn model
    # """

    # Title and description
    st.title("Solar PV production across the globe :earth_africa:")
    st.write("Enter any city to get the forecasted PV production in this location over a typical year. :sunny:")

    city_name = st.text_input("Enter a city to see the predicted PV production in that location.")

    api_key = "bc31ed29030a92462069b2bd82a34d5d"

    def fetch_lat_lon(city_name, api_key=api_key):
        BASE_URI = 'https://api.openweathermap.org/geo/1.0/direct'  # for lat and lon
        params = {
            "q": city_name,
            "limit": 1,
            "appid": api_key
        }
        response = requests.get(BASE_URI, params=params)

        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            if result:
                return result[0]  # return the first result (lat, lon)
            else:
                st.error("No results found for this city.")
                return None
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return None

    if city_name:  # Check if city_name is not empty
        location = fetch_lat_lon(city_name.capitalize(), api_key=api_key)

        if location:  # Proceed only if location data is found
            lat = location['lat']
            lon = location['lon']

    # Input fields for latitude and longitude
    # lat = st.number_input("Latitude", value=0.0, format="%.6f")
    # lon = st.number_input("Longitude", value=0.0, format="%.6f")

    # Button to trigger the API call
    if st.button("Get annual forecast"):
        try:
            # Make API call to the local endpoint
            api_url = "https://mvp3-1071061957527.europe-west1.run.app/eu_predict"  # Replace with the actual API URL if different
            params = {"lat": lat, "lon": lon}
            response = requests.get(api_url, params=params)

            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                st.success(data["message"])  # Display the message

                # Display a map with the input coordinates
                st.write("Location of Input Coordinates:")
                # st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
                st.map(data=pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=3)

                # Convert the data to a DataFrame and plot the graph
                df = pd.DataFrame(data["df"])
                if not df.empty and 'date' in df.columns and 'predicted_output' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
                    # st.write("Forecast of Predicted Output Over Time:")
                    fig = px.line(df, x="date", y="predicted_output", title="Predicted Output Over Time",
                                labels={"date": "Date", "predicted_output": "Predicted Output"})
                    st.plotly_chart(fig)
                else:
                    st.warning("The data does not contain the required columns ('date' and 'predicted_output').")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
