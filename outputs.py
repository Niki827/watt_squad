'''
This file contains the code to answer API requests and build the website. It relies heavily on the calculations.py file.
'''

import calculations

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
    pass


def total_electricity_cost():
    '''
    Output how much electricity cost the Norwegian grid would face if they would not have any pv_production nor any solar_production.
    Provide graphs on the website on the electricity cost across the time period.

    Make use of
    - total_electricity_cost()
    '''
    pass


def net_electricity_cost():
    '''
    Output how much electricity cost the Norwegian grid faces after taking into account their pv_production and their solar_production.
    Output both the actual and predicted values.
    Provide graphs on the website on the electricity cost across the time period. Maybe integrate both actual and predicted values into a single graph.

    Make use of
    - net_electricity_cost()
    - pred_net_electricity_cost()
    '''
    pass
