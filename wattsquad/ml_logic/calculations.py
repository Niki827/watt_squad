'''
This file contains all the important calculations necessary to produce the API and website output.
'''
# Importing packages
import pandas as pd
from wattsquad.ml_logic import models
from wattsquad.ml_logic import preproc


def load_entire_data():
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
    data = data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]

    data = pd.read_csv("raw_data/train.csv")

def load_training_data():
    data = pd.read_csv("raw_data/train.csv")

    # Renaming columns
    data.rename(columns={'time': 'timestamp'}, inplace=True)
    data.rename(columns={'consumption': 'actual_consumption'}, inplace=True)
    data.rename(columns={'spot_market_price': 'electricity_price'}, inplace=True)

    # Calculating total actual_production
    data['actual_production'] = data['pv_production'] + data['wind_production']

    # Dropping irrelevant columns
    data = data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]

    return data

# Loading actual and predicted relevant data
def load_data():
    '''
    This function loads actual and forecasted production and consumption data for functions down below.
	•	The expected columns in data are:
	•	'timestamp': Date and time of each record.
	•	'actual_consumption': Measured consumption.
	•	'forecasted_consumption': Predicted consumption used for planning shifts.
	•	'actual_production': Measured production from sources like solar and wind.
	•	'forecasted_production': Predicted production.
	•	'electricity_price': Electricity price at each time period.
    '''

    # Importing data for testing period
    data = pd.read_csv("raw_data/test.csv")[:24]

    # Renaming columns
    data.rename(columns={'time': 'timestamp'}, inplace=True)
    data.rename(columns={'consumption': 'actual_consumption'}, inplace=True)
    data.rename(columns={'spot_market_price': 'electricity_price'}, inplace=True)

    # Calculating total actual_production
    data['actual_production'] = data['pv_production'] + data['wind_production']

    # Dropping irrelevant columns
    data = data[['timestamp', 'actual_consumption', 'actual_production', 'electricity_price']]


    # merging the predictions on the timestamp
    # solar_predictions_df = models.XGBRegressor_solar()
    # data = data.merge(solar_predictions_df, on='timestamp')


    # the code below uses actual values for consumption and wind_production as placeholders until corresponding forecasts are ready

    # receive data from model (dataframe with 24 values for consumption)
    forecasted_solar_prod = models.predict_rnn_solar()
    forecasted_consumption = models.predict_rnn_consumption()


    # merge with main dataframe
    data['forecasted_solar_prod'] = forecasted_solar_prod
    data['forecasted_consumption'] = forecasted_consumption



    # placeholder_data = pd.read_csv('raw_data/test.csv')
    # placeholder_data.rename(columns={'time': 'timestamp'}, inplace=True)
    # placeholder_data = placeholder_data[['timestamp', 'wind_production', 'consumption']]
    # placeholder_data.rename(columns={'consumption': 'forecasted_consumption'}, inplace=True)
    # merging with the data df
    #data = data.merge(placeholder_data, on='timestamp')

    # creating forecasted_production column
    data['forecasted_production'] = data['wind_production'] + data['pv_forecast']
    data = data[['timestamp', 'actual_consumption', 'forecasted_consumption', 'actual_production', 'forecasted_production', 'electricity_price']]

    return data


# Actual electricity cost
def electricity_cost_without_renewables():
    '''
    This function calculates the total electricity cost. For the Norwegian microgrid and the time period of the test dataset.
    It assumes that no electricity is generated by solar and wind and therefore all consumption will have to be fuelled by the grid.
    '''
    # Importing data
    data = load_data()

    # Calculating electricity_cost
    data['electricity_cost'] = data['actual_consumption'] * data['electricity_price']
    return data


def actual_electricity_cost_with_renewables():
    '''
    This function calculates the net electricity cost. For the Norwegian microgrid and the time period of the test dataset.
    It takes into account the electricity generated by wind and solar. Therefore, only the excess consumption (consumption that is not met by production)
    will have to be bought from the grid. It assumes that no battery storage system exists.
    '''
    # Importing data
    data = load_data()

    # Calculating excess consumption
    data['excess_consumption_kwH'] = data['actual_consumption'] - data['actual_production']

    # Calculating electricity_cost
    data['electricity_cost'] = data['excess_consumption_kwH'] * data['electricity_price']
    return data



# Predicted electricity cost
def pred_electricity_cost_with_renewables():
    '''
    This function calculates the net electricity cost. For the Norwegian microgrid and the time period of the test dataset.
    It takes into account the electricity generated by wind and solar. Therefore, only the excess consumption (consumption that is not met by production)
    will have to be bought from the grid.
    Importantly, this function will use our predictions for pv_production, wind_production, consumption to perform this calculation.
    It assumes that no battery storage system exists.
    '''
    # Importing data
    data = load_data()

    # Calculating excess consumption
    data['excess_consumption_kwH'] = data['forecasted_consumption'] - data['forecasted_production']

    # Calculating electricity_cost
    data['electricity_cost'] = data['excess_consumption_kwH'] * data['electricity_price']
    return data


# Predicting cost savings if consumption can be shifted
def cost_savings(flexibility_degree):
    '''
    Calculates cost savings by optimizing energy consumption patterns to align better with renewable energy production, based on a user-defined flexibility degree.

    **Function Overview:**
    This function simulates two scenarios over a given time period:
    1. **Without Consumption Shifting:** Calculates the total electricity cost assuming no adjustments are made to consumption patterns.
    2. **With Consumption Shifting:** Adjusts consumption patterns within a defined flexibility range to minimize electricity costs by aligning consumption with renewable production.

    The function ensures that the adjusted consumption totals match the actual consumption levels to maintain consistency, while avoiding any bias due to discrepancies between forecasted and actual data.

    **Key Steps:**
    1. **Load Data:** Fetches relevant data including actual and forecasted consumption, production, and electricity prices.
    2. **Flexible Consumption:** Calculates the daily total of consumption that can be shifted based on the provided flexibility degree.
    3. **Consumption Shifting:** Redistributes consumption from deficit (high-cost) hours to surplus (low-cost) hours using forecasted data.
    4. **Scaling Adjusted Consumption:** Ensures that the total adjusted consumption matches the actual daily consumption levels.
    5. **Cost Calculations:** Computes electricity costs for both scenarios and derives cost savings.

    **Parameters:**
    - `start_date` (*str* or *datetime*): Start date for the analysis period.
    - `end_date` (*str* or *datetime*): End date for the analysis period.
    - `flexibility_degree` (*float*): The percentage of total daily consumption that can be shifted (0 to 100).

    **Returns:**
    - A pandas DataFrame containing:
      - `'Total Cost Without Shifting'`: Total electricity cost without any consumption adjustments.
      - `'Total Cost With Shifting'`: Total electricity cost after optimizing consumption.
      - `'Cost Savings'`: The difference between the two scenarios, representing the cost savings achieved.

    **Function Logic:**
    1. **Preprocessing:**
       - Ensures timestamps are properly formatted and creates a daily grouping for calculations.
       - Initializes `adjusted_consumption` with forecasted values.

    2. **Calculate Flexible Consumption:**
       - Determines daily flexible consumption based on forecasted data and flexibility degree.
       - Accounts for discrepancies by subtracting over-forecasted values.

    3. **Shift Consumption:**
       - Identifies surplus (overproduction) and deficit (overconsumption) hours.
       - Prioritizes surplus hours with the highest production and deficit hours with the highest electricity prices.
       - Adjusts consumption iteratively, respecting the flexibility limits.

    4. **Scaling Adjusted Consumption:**
       - Ensures that the total adjusted consumption matches the actual daily consumption, maintaining consistency.

    5. **Cost Calculations:**
       - Computes costs for both scenarios based on excess consumption and electricity prices.
       - Excess consumption is calculated as the difference between consumption and production, clipped at zero.

    6. **Results Compilation:**
       - Summarizes total costs and cost savings into a DataFrame.

    **Example Usage:**
    ```python
    start_date = '2023-01-01'
    end_date = '2023-01-07'
    flexibility_degree = 80.0  # 80% of daily consumption is flexible

    results = cost_savings(start_date, end_date, flexibility_degree)
    print(results)
    ```

    **Assumptions:**
    - Consumption shifting is limited to a single day (no cross-day adjustments).
    - Unforecasted excess consumption is considered non-flexible.
    - Actual production values are used for both scenarios to ensure realism.

    **Dependencies:**
    - The `load_data` function must provide the required columns:
      - `'timestamp'`, `'actual_consumption'`, `'forecasted_consumption'`, `'actual_production'`, `'forecasted_production'`, `'electricity_price'`.

    **Additional Notes:**
    - The function balances hypothetical scenarios with real-world constraints by scaling adjusted consumption.
    - Results depend on the accuracy of forecasted data and the flexibility degree specified.
    '''
    # Load data
    data = load_data()

    # Ensure 'timestamp' is datetime and create 'date' column
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['date'] = data['timestamp'].dt.date

    # Initialize adjusted consumption with forecasted consumption
    data['adjusted_consumption'] = data['forecasted_consumption']

    # Calculate total forecasted consumption per day
    daily_forecast = data.groupby('date')['forecasted_consumption'].sum().reset_index()
    daily_forecast.rename(columns={'forecasted_consumption': 'total_forecasted_consumption'}, inplace=True)

    # Merge daily totals back into the main data
    data = data.merge(daily_forecast, on='date')

    # Calculate total flexible consumption per day
    data['total_flexible_consumption'] = data['total_forecasted_consumption'] * (flexibility_degree / 100)

    # Calculate unforecasted excess consumption (actual - forecasted)
    data['unforecasted_excess_consumption'] = data['actual_consumption'] - data['forecasted_consumption']
    data['unforecasted_excess_consumption'] = data['unforecasted_excess_consumption'].clip(lower=0)

    # Calculate over-forecasted consumption if actual < forecasted
    data['over_forecasted_consumption'] = data['forecasted_consumption'] - data['actual_consumption']
    data['over_forecasted_consumption'] = data['over_forecasted_consumption'].clip(lower=0)

    # Adjust total flexible consumption per day
    daily_over_forecasted = data.groupby('date')['over_forecasted_consumption'].sum().reset_index()
    data = data.merge(daily_over_forecasted, on='date', suffixes=('', '_daily'))
    data['total_flexible_consumption'] -= data['over_forecasted_consumption_daily']
    data['total_flexible_consumption'] = data['total_flexible_consumption'].clip(lower=0)

    # Proceed with shifting using flexible consumption
    for date in data['date'].unique():
        day_data = data[data['date'] == date]
        indices = day_data.index
        remaining_flexible_consumption = day_data['total_flexible_consumption'].iloc[0]

        # Identify surplus and deficit hours
        surplus_hours = day_data[day_data['forecasted_production'] > day_data['forecasted_consumption']].copy()
        deficit_hours = day_data[day_data['forecasted_production'] <= day_data['forecasted_consumption']].copy()

        # Calculate surplus and deficit
        surplus_hours['surplus'] = surplus_hours['forecasted_production'] - surplus_hours['forecasted_consumption']
        deficit_hours['deficit'] = deficit_hours['forecasted_consumption'] - deficit_hours['forecasted_production']

        # Sort surplus hours by surplus amount (descending)
        surplus_hours = surplus_hours.sort_values(by='surplus', ascending=False)

        # Sort deficit hours by electricity price (descending)
        deficit_hours = deficit_hours.sort_values(by='electricity_price', ascending=False)

        # Shift consumption from deficit hours to surplus hours
        for idx_s, row_s in surplus_hours.iterrows():
            for idx_d, row_d in deficit_hours.iterrows():
                if remaining_flexible_consumption <= 0:
                    break

                # Maximum shiftable consumption from deficit hour
                max_shift_from_deficit = row_d['forecasted_consumption'] * (flexibility_degree / 100)

                # Determine the amount that can be shifted
                shift_amount = min(
                    row_s['surplus'],
                    row_d['deficit'],
                    max_shift_from_deficit,
                    remaining_flexible_consumption
                )

                if shift_amount <= 0:
                    continue

                # Update adjusted consumption
                data.at[idx_s, 'adjusted_consumption'] += shift_amount
                data.at[idx_d, 'adjusted_consumption'] -= shift_amount

                # Update remaining flexible consumption
                remaining_flexible_consumption -= shift_amount

                # Update surplus and deficit
                surplus_hours.at[idx_s, 'surplus'] -= shift_amount
                deficit_hours.at[idx_d, 'deficit'] -= shift_amount

                if surplus_hours.at[idx_s, 'surplus'] <= 0:
                    break

    # Calculate total adjusted consumption per day
    daily_adjusted = data.groupby('date')['adjusted_consumption'].sum().reset_index()
    daily_adjusted.rename(columns={'adjusted_consumption': 'total_adjusted_consumption'}, inplace=True)

    # Calculate total actual consumption per day
    daily_actual = data.groupby('date')['actual_consumption'].sum().reset_index()
    daily_actual.rename(columns={'actual_consumption': 'total_actual_consumption'}, inplace=True)

    # Merge daily totals back into the main data
    data = data.merge(daily_adjusted, on='date')
    data = data.merge(daily_actual, on='date')

    # Calculate scaling factor per day to match total actual consumption
    data['scaling_factor'] = data['total_actual_consumption'] / data['total_adjusted_consumption']

    # Apply scaling factor to adjusted_consumption
    data['adjusted_consumption'] = data['adjusted_consumption'] * data['scaling_factor']

    # Clean up any potential negative values (though none should exist)
    data['adjusted_consumption'] = data['adjusted_consumption'].clip(lower=0)

    # Cost without shifting
    data['excess_consumption_no_shift'] = data['actual_consumption'] - data['actual_production']
    data['excess_consumption_no_shift'] = data['excess_consumption_no_shift'].clip(lower=0)
    data['cost_without_shifting'] = data['excess_consumption_no_shift'] * data['electricity_price']
    total_cost_without_shifting = data['cost_without_shifting'].sum()

    # Cost with shifting
    data['excess_consumption_with_shift'] = data['adjusted_consumption'] - data['actual_production']
    data['excess_consumption_with_shift'] = data['excess_consumption_with_shift'].clip(lower=0)
    data['cost_with_shifting'] = data['excess_consumption_with_shift'] * data['electricity_price']
    total_cost_with_shifting = data['cost_with_shifting'].sum()

    # Calculate cost savings
    cost_savings_value = total_cost_without_shifting - total_cost_with_shifting

    # Compile results
    result = pd.DataFrame({
        'Total Cost Without Shifting': [total_cost_without_shifting],
        'Total Cost With Shifting': [total_cost_with_shifting],
        'Cost Savings': [cost_savings_value]
    })

    return result

    # note that we can also return the columns data['cost_without_shifting'] and data['cost_with_shifting'] to return dataframes that provide cost savings on a daily/hourly basis
    # might be nice to visualize instead of stating just a single result
    # will leave the single result for now however, better to validate the API.

#my_cost_savings = cost_savings(0.9)
#print(my_cost_savings)

def actual_battery_percentage(capacity=500, initial_battery_percentage=0):
    """
    Calculate the actual battery percentage based on production and consumption.

    Args:
        capacity (float): Battery capacity in kWh (default is 500 kWh).
        initial_battery_percentage (float): Initial battery percentage (default is 0).

    Returns:
        DataFrame: Updated DataFrame with columns for battery status and grid purchases.
    """
    # Load data
    data = load_training_data()

    # Initialize new columns
    data['excess_production_kwH'] = data['actual_production'] - data['actual_consumption']
    data['battery_percentage'] = initial_battery_percentage
    data['electricity_bought_kwH'] = 0
    data['electricity_bought_NOK'] = 0

    # Iterate over each time step
    for i in range(len(data)):
        excess_production = data.loc[i, 'excess_production_kwH']
        battery_percentage = data.loc[i, 'battery_percentage']

        if excess_production > 0: # Excess production, charge the battery
            additional_charge = excess_production / capacity
            new_battery_percentage = battery_percentage + additional_charge
            data.loc[i, 'battery_percentage'] = min(new_battery_percentage, 1) # Cap at 100%

        elif excess_production < 0: # Deficit, discharge battery or buy from the grid
            required_discharge = abs(excess_production) /  capacity
            if battery_percentage >= required_discharge: # Sufficient battery
                data.loc[i, 'battery_percentage'] = battery_percentage - required_discharge
            else: # Insufficient battery, buy from grid
                data.loc[i, 'battery_percentage'] = 0
                shortfall_kwH = abs(excess_production) - (battery_percentage * capacity)
                data.loc[i, 'electricity_bought_kwH'] = shortfall_kwH
                data.loc[i, 'electricity_bought_NOK'] = shortfall_kwH * data.loc[i, 'electricity_price']

        # Carry forward the battery percentage to the next row
        if i < len(data) -1:
            data.loc[i+1, 'battery_percentage'] = data.loc[i, 'battery_percentage']

    return data

def pred_battery_percentage(capacity=500, initial_battery_percentage=0):
    """
    Calculate the predicted battery percentage based on production and consumption.

    Args:
        capacity (float): Battery capacity in kWh (default is 500 kWh).
        initial_battery_percentage (float): Initial battery percentage (default is 0).

    Returns:
        DataFrame: Updated DataFrame with columns for battery status and grid purchases.
    """
    # Load data
    data = load_data()

    # Initialize new columns
    data['excess_production_kwH'] = data['forecasted_production'] - data['forecasted_consumption']
    data['battery_percentage'] = initial_battery_percentage
    data['electricity_bought_kwH'] = 0
    data['electricity_bought_NOK'] = 0

    # Iterate over each time step
    for i in range(len(data)):
        excess_production = data.loc[i, 'excess_production_kwH']
        battery_percentage = data.loc[i, 'battery_percentage']

        if excess_production > 0: # Excess production, charge the battery
            additional_charge = excess_production / capacity
            new_battery_percentage = battery_percentage + additional_charge
            data.loc[i, 'battery_percentage'] = min(new_battery_percentage, 1) # Cap at 100%

        elif excess_production < 0: # Deficit, discharge battery or buy from the grid
            required_discharge = abs(excess_production) /  capacity
            if battery_percentage >= required_discharge: # Sufficient battery
                data.loc[i, 'battery_percentage'] = battery_percentage - required_discharge
            else: # Insufficient battery, buy from grid
                data.loc[i, 'battery_percentage'] = 0
                shortfall_kwH = abs(excess_production) - (battery_percentage * capacity)
                data.loc[i, 'electricity_bought_kwH'] = shortfall_kwH
                data.loc[i, 'electricity_bought_NOK'] = shortfall_kwH * data.loc[i, 'electricity_price']

        # Carry forward the battery percentage to the next row
        if i < len(data) -1:
            data.loc[i+1, 'battery_percentage'] = data.loc[i, 'battery_percentage']

    return data


my_battery = actual_battery_percentage()
print(my_battery[['battery_percentage', 'electricity_bought_kwH', 'electricity_bought_NOK']].describe())
