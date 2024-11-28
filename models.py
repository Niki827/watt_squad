'''
This file contains all of the relevant prediction models.
'''

# importing relevant packages
import pandas as pd
import numpy as np
#import

# Importing pre processed data


# creating an XGBRegressor model for solar production
def XGBRegressor_solar():
    '''
    This function will build an XGBRegressor to predict the solar production of the Norwegian Rye microgrid during the testing period.
    '''

    # creating y_train dataframe
    y_train = y.copy()['pv_production']

    # Importing y_test
    y_test = pd.read_csv("raw_data/test.csv")
    y_test = y_test['pv_production']
