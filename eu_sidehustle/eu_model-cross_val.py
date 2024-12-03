# Importing libraries
# Data manipulation
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# Data Visualiation
import matplotlib.pyplot as plt
import seaborn as sns

# System
import os

# Deep Learning
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers  # Ensure this line is included
from tensorflow.keras import metrics
from tensorflow.keras.regularizers import L1L2
from typing import Dict, List, Tuple, Sequence
from tensorflow.keras.layers import Lambda  # Wraps arbitrary expressions as a Layer object
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

# Importing data
path='aggregating_preprocessing/data/04_preprocessed'
athens = pd.read_csv(f'{path}/Athens_preprocessed.csv')
berlin = pd.read_csv(f'{path}/Berlin_preprocessed.csv')
london = pd.read_csv(f'{path}/London_preprocessed.csv')
madrid = pd.read_csv(f'{path}/Madrid_preprocessed.csv')
paris = pd.read_csv(f'{path}/Paris_preprocessed.csv')

# Define train test split function
def train_test_split(fold:pd.DataFrame) -> Tuple[pd.DataFrame]:
    # Calculate the index for two-thirds of the rows
    train_size = int(fold.shape[0] * (2 / 3))

    # Create a new DataFrame with the first two-thirds of rows
    fold_train = fold.iloc[:train_size]

    # Create a new DataFrame with the last third of rows
    fold_test = fold.iloc[train_size:]

    return (fold_train, fold_test)

# Define sequences function
def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int, sequence_stride: int):
    '''
    - slides through a `fold` Time Series (2D array) to create sequences of equal
        * `input_length` for X,
        * `output_length` for y,
    using a temporal gap `sequence_stride` between each sequence
    - returns a list of sequences, each as a 2D-array time series
    '''

    X, y = [], []

    for i in range(0, len(fold), sequence_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (i + input_length + output_length) >= len(fold):
            break
        X_i = fold.iloc[i:i + input_length, :]
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][['pv_output']] # index + length of sequence until index + length of seq. + length of target
        X.append(X_i)
        y.append(y_i)

    return np.array(X), np.array(y)

# Define init_model function (LSTM)
def init_model(X_train, y_train):
    # 1 - RNN architecture
    model = models.Sequential()
    ## 1.1 - Recurrent Layer
    model.add(layers.LSTM(64,
                          activation='tanh',
                          return_sequences=False,
                          kernel_regularizer=L1L2(l1=0.05, l2=0.05),
                          input_shape=(X_train.shape[1], X_train.shape[2])  # Specify input shape
                          ))
    ## 1.2 - Predictive Dense Layers
    output_length = y_train.shape[1]
    model.add(layers.Dense(output_length, activation='linear'))

    # 2 - Compiler
    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model

def init_baseline():

    # YOUR CODE HERE
    model = models.Sequential()
    # a layer to take the last value of the sequence and output it
    model.add(layers.Lambda(lambda x: x[:,-1,1,None]))  # all sequences, last day, 1 feature (temperature)


    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model

# Plot the loss history
def plot_history(history):

    fig, ax = plt.subplots(1,2, figsize=(20,7))
    # --- LOSS: MSE ---
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('MSE')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- METRICS:MAE ---

    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    return ax

# Define fit_model function

def fit_model(model: tf.keras.Model, verbose=1) -> Tuple[tf.keras.Model, dict]:

    es = EarlyStopping(monitor = "val_loss",
                      patience = 3,
                      mode = "min",
                      restore_best_weights = True)


    history = model.fit(X_train_dict['paris_train'], y_train_dict['paris_train'],
                        validation_split = 0.3,
                        shuffle = False,     # the order matters!!!
                        batch_size = 32,
                        epochs = 50,
                        callbacks = [es],
                        verbose = verbose)

    return model, history   # returns a tuple: (model, dict(history))


# Define cross_validate_baseline_and_lstm function
def cross_validate_baseline_and_lstm():
    '''
    This function cross-validates
    - the "last seen value" baseline model
    - the RNN model
    '''

    list_of_mae_baseline_model = []
    list_of_mae_recurrent_model = []

    # 0 - Creating folds
    # =========================================
    # Assume you have a dictionary with DataFrames for each city
    city_dfs = {
        'paris': paris,
        'athens': athens,
        'berlin': berlin,
        'london': london,
        'madrid': madrid
    }
    # Create a list of DataFrames (folds) for each city
    folds = [city_dfs[city] for city in city_dfs.keys()]

    for fold_id, fold in enumerate(folds):
        # 1 - Train/Test split the current fold
        # =========================================
        (fold_train, fold_test) = train_test_split(fold) # function we coded to split train/test

        X_train, y_train = get_X_y_strides(fold_train, input_length=730, output_length=365, sequence_stride=1)  # function we coded to get multiple
        X_test, y_test = get_X_y_strides(fold_test, input_length=730, output_length=365, sequence_stride=1)       # sequences from a fold

        # 2 - Modelling
        # =========================================

        ##### Baseline Model
        baseline_model = init_baseline()
        mae_baseline = baseline_model.evaluate(X_test, y_test, verbose=0)[1]   # evaluating baseline model (metric)
        list_of_mae_baseline_model.append(mae_baseline)
        print("-"*50)
        print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

        ##### LSTM Model
        model = init_model(X_train, y_train)
        es = EarlyStopping(monitor = 'val_mae',
                          mode = "min",
                           patience = 2,
                           restore_best_weights = True)
        history = model.fit(X_train, y_train,
                           validation_split = 0.3,
                           shuffle = False,
                           batch_size = 32,
                           epochs = 50,
                           callbacks = [es],
                           verbose = 0)
        res = model.evaluate(X_test, y_test, verbose = 0) # evaluating LSTM (metric)
        mae_lstm = res[1]
        list_of_mae_recurrent_model.append(mae_lstm)
        print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")

        ##### Comparison LSTM vs Baseline for the current fold
        print(f"🏋🏽‍♂️ improvement over baseline: {round((1 - (mae_lstm/mae_baseline))*100,2)} % \n")
        # Save the model

        model.save('model')

    return list_of_mae_baseline_model, list_of_mae_recurrent_model


# Run the baseline and LSTM model
mae_baselines, mae_lstms = cross_validate_baseline_and_lstm()
print(mae_baselines, mae_lstms)
