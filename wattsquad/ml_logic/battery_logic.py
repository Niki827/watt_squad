from wattsquad.ml_logic.calculations import load_training_data
import pandas as pd


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


def selling_electricity(battery_capacity: 500,
                       electricity_price_share: float):
    '''
    This function returns the costs a prosumer could have saved by selling electricity
    when production exceeds consumption and the battery is fully charged.
    '''

    # 1. Load data
    data = actual_battery_percentage(battery_capacity)

    # 2. Whenever production exceeds consumption and the battery is fully charged calculate kwH the user can sell
    data['electricity_sold_kwH'] = data.apply(
    lambda row: row['excess_production_kwH'] if row['actual_production'] > row['actual_consumption'] and row['battery_percentage'] == 1 else 0,
    axis=1
    )

    # 3. Electricity price of sold electricity
    data['electricity_sold_NOK'] = data['electricity_sold_kwH'] * data['electricity_price'] * electricity_price_share

    electricity_sold_kwH = data['electricity_sold_kwH'].sum()
    electricity_sold_NOK = data['electricity_sold_NOK'].sum()


    electricity_bought_NOK = data['electricity_bought_NOK'].sum()

    # Convert 'timestamp' to datetime if not already
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Resample to hourly averages (if needed)
    hourly_data = data.set_index('timestamp')

    # Use .loc for date slicing
    june_data = hourly_data.loc['2020-06']
    december_data = hourly_data.loc['2020-12']


    return june_data, december_data, electricity_sold_kwH, electricity_sold_NOK, electricity_bought_NOK
    # return data, electricity_sold_kwH, electricity_sold_NOK, electricity_bought_NOK
