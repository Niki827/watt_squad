'''
This file contains all of the relevant prediction models.
'''

# importing relevant packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import preproc
from keras import models
from keras import layers
from keras import optimizers, metrics
from keras import regularizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from typing import Tuple
from keras.models import load_model





# creating an XGBRegressor model for solar production
def XGBRegressor_solar():
    '''
    This function will build an XGBRegressor to predict the solar production of the Norwegian Rye microgrid during the testing period.
    '''
    # importing relevant data
    train_data = pd.read_csv('raw_data/train.csv')
    test_data = pd.read_csv('raw_data/test.csv')

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
    predictions_df = pd.read_csv("raw_data/test.csv")
    predictions_df = predictions_df[['time']]
    predictions_df['pv_forecast'] = y_pred

    # renaming the column 'time' to 'timestamp' in order to integrate it into calculations.load_data()
    predictions_df.rename(columns={'time': 'timestamp'}, inplace=True)
    return predictions_df



# train validation split for sequences
def train_val_split(df:pd.DataFrame,
                    train_val_ratio: float,
                    input_length: int) -> Tuple[pd.DataFrame]:
    '''
    Returns a train dataframe and a test dataframe (fold_train, fold_test)
    from which one can sample (X,y) sequences.
    df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    '''

    # TRAIN SET
    # ======================
    last_train_idx = round(train_val_ratio * len(df))  # split_ratio * number of rows in the fold (split_ratio of the fold for train)
    fold_train = df.iloc[0:last_train_idx, :]   # 1st until last row of train set, all columns

    # TEST SET
    # ======================
    first_val_idx = last_train_idx - input_length  # last row of train set - 2 weeks --> test set starts 2 weeks
                                                                    # before train set ends --> overlap (not a problem with X)
    fold_val = df.iloc[first_val_idx:, :]   # 1st until last row of val set, all columns

    return (fold_train, fold_val)


# Get one random sequence
def get_Xi_yi(
    df:pd.DataFrame,
    input_length:int,  # 120
    output_length:int,
    target: str):  # 120
    '''
    - given a fold, it returns one sequence (X_i, y_i)
    - with the starting point of the sequence being chosen at random
    '''
    # YOUR CODE
    first_possible_start = 0                                    # the +1 accounts for the index, that is exclusive.
    last_possible_start = len(df) - (input_length + output_length) + 1    # It can start as long as there are still
                                                                             # 120 + 12 hours after the 1st hour.
    random_start = np.random.randint(first_possible_start, last_possible_start)  # np.random to pick a day inside
                                                                                    # the possible interval.
    X_i = df.iloc[random_start:random_start+input_length]

    y_i = df.iloc[random_start+input_length:
                  random_start+input_length+output_length][target]  # creates a pd.DataFrame for the target y

    return (X_i, y_i)



# Creates many random sequences
def get_X_y(
    df:pd.DataFrame,
    number_of_sequences:int,
    input_length:int,
    output_length:int,
    target: str
):
    # YOUR CODE HERE
    X, y = [], []  # lists for the sequences for X and y

    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(df, input_length, output_length, target)   # calls the previous function to generate sequences X + y
        X.append(Xi)
        y.append(yi)

    return np.array(X), np.array(y)


# Model for one output (input has to be X_train (8563, 120, 14), y_train ((8563, 12, 1)))
def init_model(X_train, y_train):

    # 1 - RNN architecture
    # ======================
    model = models.Sequential()

    ## 1.1 - Recurrent Layers
    model.add(layers.LSTM(32,
                          activation='tanh',
                          return_sequences=True,
                          input_shape=X_train.shape[1:],
                          kernel_regularizer=regularizers.l2(0.01)))

    model.add(layers.Dropout(0.3))

    model.add(layers.LSTM(16, activation='tanh', return_sequences=True))

    ## 1.2 - Slice the output to focus only on the last 12 time steps
    model.add(layers.Lambda(lambda x: x[:, -24:, :]))  # Keep only the last 12 time steps

    ## 1.3 - Hidden Dense Layer
    model.add(layers.TimeDistributed(layers.Dense(64, activation="relu")))

    ## 1.4 - Predictive Dense Layer
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))

    # 2 - Compiler
    # ======================
    adam = optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mae'
                  ])


    return model


def fit_model(model: tf.keras.Model,
              X_train,
              y_train,
              X_val,
              y_val,
              verbose=1) -> Tuple[tf.keras.Model, dict]:

    es = EarlyStopping(monitor = "val_loss",
                      patience = 15,
                      mode = "min",
                      restore_best_weights = True)


    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        shuffle = False,
                        batch_size = 32,
                        epochs = 1000,
                        callbacks = [es],
                        verbose = verbose)

    return model, history


# creating an RNN/LSTM model for solar production
def RNN_solar():
    '''
    This function will build an LSTM to predict the solar production of the Norwegian Rye microgrid during the testing period.
    '''

    # importing relevant data
    train_data = pd.read_csv('raw_data/train.csv')
    test_data = pd.read_csv('raw_data/test.csv')

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

    # What is our target
    TARGET = 'pv_production'

    # creating a df with features and target(s)
    df = X_train_transformed.copy()
    df[TARGET] = y_train

    # Drop columns which are not necessary
    columns_drop = ['minmaxscaler__dew_point_100m:C', 'minmaxscaler__month_sine', 'minmaxscaler__total_cloud_cover:p',
                    'minmaxscaler__sin_sun_azimuth:d', 'minmaxscaler__t_100m:C', 'minmaxscaler__t_50m:C',
                    'minmaxscaler__high_cloud_cover:p', 'minmaxscaler__t_10m:C', 'minmaxscaler__temp', 'minmaxscaler__wind_speed_50m:ms',
                    'minmaxscaler__relative_humidity_100m:p', 'minmaxscaler__relative_humidity_10m:p', 'minmaxscaler__wind_speed_10m:ms',
                    'onehotencoder__precip_type:idx_1.0', 'minmaxscaler__effective_cloud_cover:p', 'minmaxscaler__relative_humidity_50m:p',
                    'minmaxscaler__sin_wind_dir_2m:d', 'minmaxscaler__low_cloud_cover:p', 'minmaxscaler__cos_wind_dir_50m:d',
                    'minmaxscaler__wind_speed_100m:ms', 'minmaxscaler__precip_1h:mm', 'minmaxscaler__direct_rad_1h:Wh',
                    'minmaxscaler__sin_wind_dir_10m:d', 'onehotencoder__precip_type:idx_0.0', 'minmaxscaler__cos_wind_dir_2m:d',
                    'minmaxscaler__season_sine', 'minmaxscaler__clear_sky_rad:W', 'minmaxscaler__dew_point_10m:C', 'minmaxscaler__prob_precip_1h:p',
                    'minmaxscaler__wind_speed_2m:ms','minmaxscaler__cos_wind_dir_10m:d', 'minmaxscaler__dew_point_2m:C',
                    'minmaxscaler__cos_wind_dir_100m:d', 'minmaxscaler__sunshine_duration_1h:min','minmaxscaler__relative_humidity_2m:p',
                    'minmaxscaler__season_cosine', 'minmaxscaler__diffuse_rad_1h:Wh', 'minmaxscaler__clear_sky_energy_1h:J',
                    'onehotencoder__precip_type:idx_3.0', 'onehotencoder__precip_type:idx_2.0']


    # Dropping the unimportant columns
    df = df.drop(columns = columns_drop).copy()

    # Five days as input length
    INPUT_LENGTH = 24 * 5 # records every hour x 24 hours
                        # for 5 days

    # 12 hours as output length
    OUTPUT_LENGTH = 24

    #How many mini sequences do i want in my train and val set? Is according to the split_ratio
    NUMBER_OF_SEQUENCES_TRAIN = int(len(df) * 0.9)
    NUMBER_OF_SEQUENCES_VAL = int(len(df) * 0.1)

    #Set train_val ratio
    TRAIN_VAL_RATIO = 0.9

    #Split the dataset into train & val
    df_train, df_val = train_val_split(df, TRAIN_VAL_RATIO, INPUT_LENGTH)

    #get mini sequences for both train & val set
    X_train, y_train = get_X_y(df_train, NUMBER_OF_SEQUENCES_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH, TARGET)
    X_val, y_val = get_X_y(df_val, NUMBER_OF_SEQUENCES_VAL, INPUT_LENGTH, OUTPUT_LENGTH, TARGET)

    #Dropping the targets from the X (we dont want to train the model on the targets)
    X_train = X_train[:, :, :-1]
    X_val = X_val[:, :, :-1]

    #Expanding the dimension of the y for the model
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)


    model = init_model(X_train, y_train)
    #model.summary()

    # 2 - Training
    # ====================================
    model, history = fit_model(model, X_train, y_train, X_val, y_val)

    model.saved_model('rnn_solar')

    return model, history



def predict_rnn_solar():
    model = load_model('rnn_solar')
    train_data = pd.read_csv('raw_data/test.csv')
    X_train = train_data.drop(columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price'])


    X_train_transformed = preproc.transform_data(X_train)
    columns_drop = ['minmaxscaler__dew_point_100m:C', 'minmaxscaler__month_sine', 'minmaxscaler__total_cloud_cover:p',
                    'minmaxscaler__sin_sun_azimuth:d', 'minmaxscaler__t_100m:C', 'minmaxscaler__t_50m:C',
                    'minmaxscaler__high_cloud_cover:p', 'minmaxscaler__t_10m:C', 'minmaxscaler__temp', 'minmaxscaler__wind_speed_50m:ms',
                    'minmaxscaler__relative_humidity_100m:p', 'minmaxscaler__relative_humidity_10m:p', 'minmaxscaler__wind_speed_10m:ms',
                    'onehotencoder__precip_type:idx_1.0', 'minmaxscaler__effective_cloud_cover:p', 'minmaxscaler__relative_humidity_50m:p',
                    'minmaxscaler__sin_wind_dir_2m:d', 'minmaxscaler__low_cloud_cover:p', 'minmaxscaler__cos_wind_dir_50m:d',
                    'minmaxscaler__wind_speed_100m:ms', 'minmaxscaler__precip_1h:mm', 'minmaxscaler__direct_rad_1h:Wh',
                    'minmaxscaler__sin_wind_dir_10m:d', 'onehotencoder__precip_type:idx_0.0', 'minmaxscaler__cos_wind_dir_2m:d',
                    'minmaxscaler__season_sine', 'minmaxscaler__clear_sky_rad:W', 'minmaxscaler__dew_point_10m:C', 'minmaxscaler__prob_precip_1h:p',
                    'minmaxscaler__wind_speed_2m:ms','minmaxscaler__cos_wind_dir_10m:d', 'minmaxscaler__dew_point_2m:C',
                    'minmaxscaler__cos_wind_dir_100m:d', 'minmaxscaler__sunshine_duration_1h:min','minmaxscaler__relative_humidity_2m:p',
                    'minmaxscaler__season_cosine', 'minmaxscaler__diffuse_rad_1h:Wh', 'minmaxscaler__clear_sky_energy_1h:J',
                    'onehotencoder__precip_type:idx_3.0', 'onehotencoder__precip_type:idx_2.0']

    X_train_transformed = X_train_transformed.drop(columns = columns_drop).copy()
    X_train_transformed = X_train_transformed.iloc[-120:]
    X_train_transformed = np.expand_dims(X_train_transformed, axis=0)
    y_pred= model.predict(X_train_transformed)
    return y_pred


def RNN_consumption():
    '''
    This function will build an LSTM to predict the consumption of the Norwegian Rye microgrid during the testing period.
    '''

    # importing relevant data
    train_data = pd.read_csv('raw_data/train.csv')
    test_data = pd.read_csv('raw_data/test.csv')

    # creating y_train and y_test
    y_train = train_data['consumption'].copy()
    y_test = test_data['consumption'].copy()

    # creating X_train and X_test
    X_train = train_data
    X_train = X_train.drop(columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price'])
    X_test = test_data
    X_test = X_test.drop(columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price'])

    # Preprocessing features
    X_train_transformed = preproc.transform_data(X_train)
    X_test_transformed = preproc.transform_data(X_test)

    # What is our target
    TARGET = 'consumption'

    # creating a df with features and target(s)
    df = X_train_transformed.copy()
    df[TARGET] = y_train

    # Drop columns which are not necessary
    columns_drop = ['minmaxscaler__dew_point_100m:C', 'minmaxscaler__month_sine', 'minmaxscaler__total_cloud_cover:p',
                    'minmaxscaler__sin_sun_azimuth:d', 'minmaxscaler__t_100m:C', 'minmaxscaler__t_50m:C',
                    'minmaxscaler__high_cloud_cover:p', 'minmaxscaler__t_10m:C', 'minmaxscaler__temp', 'minmaxscaler__wind_speed_50m:ms',
                    'minmaxscaler__relative_humidity_100m:p', 'minmaxscaler__relative_humidity_10m:p', 'minmaxscaler__wind_speed_10m:ms',
                    'onehotencoder__precip_type:idx_1.0', 'minmaxscaler__effective_cloud_cover:p', 'minmaxscaler__relative_humidity_50m:p',
                    'minmaxscaler__sin_wind_dir_2m:d', 'minmaxscaler__low_cloud_cover:p', 'minmaxscaler__cos_wind_dir_50m:d',
                    'minmaxscaler__wind_speed_100m:ms', 'minmaxscaler__precip_1h:mm', 'minmaxscaler__direct_rad_1h:Wh',
                    'minmaxscaler__sin_wind_dir_10m:d', 'onehotencoder__precip_type:idx_0.0', 'minmaxscaler__cos_wind_dir_2m:d',
                    'minmaxscaler__season_sine', 'minmaxscaler__clear_sky_rad:W', 'minmaxscaler__dew_point_10m:C', 'minmaxscaler__prob_precip_1h:p',
                    'minmaxscaler__wind_speed_2m:ms','minmaxscaler__cos_wind_dir_10m:d', 'minmaxscaler__dew_point_2m:C',
                    'minmaxscaler__cos_wind_dir_100m:d', 'minmaxscaler__sunshine_duration_1h:min','minmaxscaler__relative_humidity_2m:p',
                    'minmaxscaler__season_cosine', 'minmaxscaler__diffuse_rad_1h:Wh', 'minmaxscaler__clear_sky_energy_1h:J',
                    'onehotencoder__precip_type:idx_3.0', 'onehotencoder__precip_type:idx_2.0']


    # Dropping the unimportant columns
    df = df.drop(columns = columns_drop).copy()

    # Five days as input length
    INPUT_LENGTH = 24 * 5 # records every hour x 24 hours
                        # for 5 days

    # 12 hours as output length
    OUTPUT_LENGTH = 24

    #How many mini sequences do i want in my train and val set? Is according to the split_ratio
    NUMBER_OF_SEQUENCES_TRAIN = int(len(df) * 0.9)
    NUMBER_OF_SEQUENCES_VAL = int(len(df) * 0.1)

    #Set train_val ratio
    TRAIN_VAL_RATIO = 0.9

    #Split the dataset into train & val
    df_train, df_val = train_val_split(df, TRAIN_VAL_RATIO, INPUT_LENGTH)

    #get mini sequences for both train & val set
    X_train, y_train = get_X_y(df_train, NUMBER_OF_SEQUENCES_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH, TARGET)
    X_val, y_val = get_X_y(df_val, NUMBER_OF_SEQUENCES_VAL, INPUT_LENGTH, OUTPUT_LENGTH, TARGET)

    #Dropping the targets from the X (we dont want to train the model on the targets)
    X_train = X_train[:, :, :-1]
    X_val = X_val[:, :, :-1]

    #Expanding the dimension of the y for the model
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)


    model = init_model(X_train, y_train)
    #model.summary()

    # 2 - Training
    # ====================================
    model, history = fit_model(model, X_train, y_train, X_val, y_val)

    model.saved_model('rnn_consumption')

    return model, history


def predict_rnn_solar():
    model = load_model('rnn_consumption')
    train_data = pd.read_csv('raw_data/test.csv')
    X_train = train_data.drop(columns=['pv_production', 'wind_production', 'consumption', 'spot_market_price'])


    X_train_transformed = preproc.transform_data(X_train)
    columns_drop = ['minmaxscaler__dew_point_100m:C', 'minmaxscaler__month_sine', 'minmaxscaler__total_cloud_cover:p',
                    'minmaxscaler__sin_sun_azimuth:d', 'minmaxscaler__t_100m:C', 'minmaxscaler__t_50m:C',
                    'minmaxscaler__high_cloud_cover:p', 'minmaxscaler__t_10m:C', 'minmaxscaler__temp', 'minmaxscaler__wind_speed_50m:ms',
                    'minmaxscaler__relative_humidity_100m:p', 'minmaxscaler__relative_humidity_10m:p', 'minmaxscaler__wind_speed_10m:ms',
                    'onehotencoder__precip_type:idx_1.0', 'minmaxscaler__effective_cloud_cover:p', 'minmaxscaler__relative_humidity_50m:p',
                    'minmaxscaler__sin_wind_dir_2m:d', 'minmaxscaler__low_cloud_cover:p', 'minmaxscaler__cos_wind_dir_50m:d',
                    'minmaxscaler__wind_speed_100m:ms', 'minmaxscaler__precip_1h:mm', 'minmaxscaler__direct_rad_1h:Wh',
                    'minmaxscaler__sin_wind_dir_10m:d', 'onehotencoder__precip_type:idx_0.0', 'minmaxscaler__cos_wind_dir_2m:d',
                    'minmaxscaler__season_sine', 'minmaxscaler__clear_sky_rad:W', 'minmaxscaler__dew_point_10m:C', 'minmaxscaler__prob_precip_1h:p',
                    'minmaxscaler__wind_speed_2m:ms','minmaxscaler__cos_wind_dir_10m:d', 'minmaxscaler__dew_point_2m:C',
                    'minmaxscaler__cos_wind_dir_100m:d', 'minmaxscaler__sunshine_duration_1h:min','minmaxscaler__relative_humidity_2m:p',
                    'minmaxscaler__season_cosine', 'minmaxscaler__diffuse_rad_1h:Wh', 'minmaxscaler__clear_sky_energy_1h:J',
                    'onehotencoder__precip_type:idx_3.0', 'onehotencoder__precip_type:idx_2.0']

    X_train_transformed = X_train_transformed.drop(columns = columns_drop).copy()
    X_train_transformed = X_train_transformed.iloc[-120:]
    X_train_transformed = np.expand_dims(X_train_transformed, axis=0)
    y_pred= model.predict(X_train_transformed)
    return y_pred
