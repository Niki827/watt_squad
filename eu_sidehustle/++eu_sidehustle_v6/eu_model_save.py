# Importing libraries
import os
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from typing import Tuple, List

# Set the path for the preprocessed training data
path = 'preprocessed_training_data'

# Dynamically load all CSV files in the directory
def load_all_preprocessed_data(path: str) -> List[pd.DataFrame]:
    cities_data = []
    for file in os.listdir(path):
        if file.endswith('_preprocessed.csv'):
            city_name = file.replace('_preprocessed.csv', '')
            print(f"Loading data for {city_name}...")
            city_df = pd.read_csv(os.path.join(path, file))
            cities_data.append(city_df)
    return cities_data

# Load all city data
cities = load_all_preprocessed_data(path)

# Define train-test split function
def train_test_split(fold: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(fold.shape[0] * (2 / 3))
    fold_train = fold.iloc[:train_size]
    fold_test = fold.iloc[train_size:]
    return fold_train, fold_test

# Define sequences function
def get_X_y_strides(fold: pd.DataFrame, input_length: int, output_length: int, sequence_stride: int):
    X, y = [], []
    for i in range(0, len(fold), sequence_stride):
        if (i + input_length + output_length) >= len(fold):
            break
        X_i = fold.iloc[i:i + input_length, :]
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][['pv_output']]
        X.append(X_i)
        y.append(y_i)
    return np.array(X), np.array(y)

# Prepare RNN data
def get_RNN_data(cities: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_sequences, y_train_sequences = [], []
    X_test_sequences, y_test_sequences = [], []

    for city in cities:
        fold_train, fold_test = train_test_split(city)
        X_train, y_train = get_X_y_strides(fold_train, input_length=730, output_length=365, sequence_stride=1)
        X_test, y_test = get_X_y_strides(fold_test, input_length=730, output_length=365, sequence_stride=1)
        X_train_sequences.append(X_train)
        y_train_sequences.append(y_train)
        X_test_sequences.append(X_test)
        y_test_sequences.append(y_test)

    X_train_sequences = np.concatenate(X_train_sequences, axis=0)
    y_train_sequences = np.concatenate(y_train_sequences, axis=0)
    X_test_sequences = np.concatenate(X_test_sequences, axis=0)
    y_test_sequences = np.concatenate(y_test_sequences, axis=0)

    return X_train_sequences, y_train_sequences, X_test_sequences, y_test_sequences

# Define the model
def init_model(X_train: np.ndarray, y_train: np.ndarray) -> models.Sequential:
    normalizer = Normalization(axis=-1)
    normalizer.adapt(X_train)
    model = models.Sequential()
    model.add(normalizer)
    model.add(layers.LSTM(64, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(layers.Dense(y_train.shape[1], activation='linear'))
    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])
    return model

# Train and save the model
def fit_model(model, X_train, y_train):
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.3,
        shuffle=False,
        batch_size=32,
        epochs=10,
        callbacks=[es],
        verbose=1
    )
    return model, history

def build_and_save_model():
    X_train, y_train, X_test, y_test = get_RNN_data(cities)
    model = init_model(X_train, y_train)
    model, history = fit_model(model, X_train, y_train)
    model.save('eu_RNN_model')
    return model, history

# Build and save the model
build_model = build_and_save_model()
