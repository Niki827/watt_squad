'''
This file contains all of the relevant prediction models.
'''

# importing relevant packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import preproc


# creating an XGBRegressor model for solar production
def XGBRegressor_solar():
    '''
    This function will build an XGBRegressor to predict the solar production of the Norwegian Rye microgrid during the testing period.
    '''
    # importing relevant data
    train_data = pd.read_csv('../../raw_data/train.csv')
    test_data = pd.read_csv('../../raw_data/test.csv')

    # creating y_train and y_test
    y_train = train_data['pv_production'].copy()
    y_test = test_data['pv_production'].copy()

    # creating X_train and X_test
    X_train = train_data
    X_train = X_train.drop(columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price'])
    X_test = test_data
    X_test = X_test.drop(columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price'])

    # Preprocessing features
    X_train_transformed = preproc.transform_data(X_train)
    X_test_transformed = preproc.transform_data(X_test)

    #Creating X_val and y_val
    X_train_transformed, X_val, y_train, y_val = train_test_split(
    X_train_transformed, y_train, test_size = 0.1, random_state = 42  # val = 10%
    )

    # Initialize the model with the best parameters from grid search
    xgb_reg = XGBRegressor(
        max_depth=7,                # Optimal value found
        n_estimators=300,           # Optimal value found
        learning_rate=0.05,         # Optimal value found
        reg_alpha=0.05,             # Optimal value found
        reg_lambda=20,              # Optimal value found
        subsample=0.8,              # Optimal value found
        colsample_bytree=0.8,       # Optimal value found
        objective='reg:squarederror',
        eval_metric="mae",
        random_state=42             # Ensuring reproducibility
    )

    # Fit the model on the training data
    xgb_reg.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_train_transformed, y_train), (X_val, y_val)],
        verbose=False,
        early_stopping_rounds=5     # Retain early stopping
    )

    print("➡️  model fitting done")

    # Make predictions
    y_pred = xgb_reg.predict(X_test_transformed)

    print("➡️  performed predictions")

    #format predictions in a suitable dataframe
    predictions_df = pd.read_csv("../../raw_data/test.csv")
    predictions_df = predictions_df[['time']]
    predictions_df['pv_forecast'] = y_pred

    # renaming the column 'time' to 'timestamp' in order to integrate it into calculations.load_data()
    predictions_df.rename(columns={'time': 'timestamp'}, inplace=True)
    return predictions_df
