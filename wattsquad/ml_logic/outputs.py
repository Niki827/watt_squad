'''
This file contains the code to answer API requests and build the website. It relies heavily on the calculations.py file.
'''

#import calculations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wattsquad.ml_logic.models import predict_rnn_solar
from wattsquad.ml_logic.models import predict_rnn_consumption


def prediction_accuracy():
    '''
    Output an accuracy metric for our predictions of pv_production, wind_production and consumption.
    Moreover, provide graphs on the website that overlay the actual and predicted pv_production, wind_production and consumption. Bar graphs might be suitable here?
    Make use of
    - actual_wind_production()
    - actual_solar_production()
    - actual_consumption()
    - pred_wind_production()
    - pred_solar_production()
    - pred_consumption()
    '''
    test_data = pd.read_csv('raw_data/test.csv')
    y_true = test_data[['pv_production', 'wind_production', 'consumption']][:24]

    #PV production
    #True
    y_true_pv = y_true['pv_production']
    y_true_pv = np.array(y_true_pv).reshape(24,1)
    #Forecast
    y_pred_pv = predict_rnn_solar()
    mae_pv = np.mean(abs(y_true_pv - y_pred_pv))

    # #Wind production
    # #True
    # y_true_wind = y_true['wind_production']
    # y_true_wind = np.array(y_true_wind).reshape(24,1)
    # #Forecast
    # y_pred_wind = models.predict_rnn_wind()
    # mae_wind = np.mean(abs(y_true_wind - y_pred_wind))

    #Consumpion
    #True
    y_true_consumption = y_true['consumption']
    y_true_consumption = np.array(y_true_consumption).reshape(24,1)
    #Forecast
    y_pred_consumption = predict_rnn_consumption()
    mae_consumption = np.mean(abs(y_true_consumption - y_pred_consumption))


    #MAE test
    # #PV
    # mae_test_pv = np.mean(abs(y_true_pv - y_pred_pv))

    # #Wind
    # mae_test_wind = np.mean(abs(y_true_wind - y_pred_wind))

    # #Consumption
    # mae_test_consumption = np.mean(abs(y_true_consumption - y_pred_consumption))


    #Plot Overlay of forecasted and actual data
    # Create a figure with 3 subplots, sharing the x-axis
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Plot PV Production (1st subplot)
    # axes[0].plot(y_pred_pv, label='PV Forecast', color='blue', linestyle='-')
    # axes[0].plot(y_true_pv, label='PV Production', color='orange', linestyle='--')
    # axes[0].set_ylabel('PV Production (kWh/h)')
    # axes[0].set_title('PV Forecast vs Actual')
    # axes[0].legend()
    # axes[0].grid(True)

    # # Plot Wind Production (2nd subplot)
    # axes[1].plot(y_pred_wind, label='Wind Forecast', color='green', linestyle='-')
    # axes[1].plot(y_true_wind, label='Wind Production', color='red', linestyle='--')
    # axes[1].set_ylabel('Wind Production (kWh/h)')
    # axes[1].set_title('Wind Forecast vs Actual')
    # axes[1].legend()
    # axes[1].grid(True)

    # Plot Consumption (3rd subplot)
    axes[2].plot(y_pred_consumption, label='Consumption Forecast', color='purple', linestyle='-')
    axes[2].plot(y_true_consumption, label='Consumption', color='brown', linestyle='--')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Consumption (kWh/h)')
    axes[2].set_title('Consumption Forecast vs Actual')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Display the plots
    plt.show()






def total_electricity_cost():
    '''
    Output how much electricity cost the Norwegian grid would face if they would not have any pv_production nor any solar_production.
    Provide graphs on the website on the electricity cost across the time period.

    Make use of
    - total_electricity_cost()
    '''

    df = pd.read_csv('raw_data/test.csv')
    df = df[['consumption', 'spot_market_price']][:24]
    df['total_electricity_price'] = df['consumption'] * df['spot_market_price']

    return df['total_electricity_price']


def net_electricity_cost():
    '''
    Output how much electricity cost the Norwegian grid faces after taking into account their pv_production and their solar_production.
    Output both the actual and predicted values.
    Provide graphs on the website on the electricity cost across the time period. Maybe integrate both actual and predicted values into a single graph.

    Make use of
    - net_electricity_cost()
    - pred_net_electricity_cost()
    '''
    df = df = pd.read_csv('raw_data/test.csv')[:24]
    total_electricity_cost = total_electricity_cost()

    total_electricity_prod = df['pv_production'] + df['wind_production']
    electricity_excess = total_electricity_prod - total_electricity_cost
    return electricity_excess
