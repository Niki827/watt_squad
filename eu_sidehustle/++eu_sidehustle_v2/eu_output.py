import preprocessing_predictions as preprocessing
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Preprocess all the data for the location requested
# Fetch the data from the EU API. Default lat and lon are for Le Wagon Berlin - we can make them dynamic

def preprocessing_data(lat, lon):
    result = preprocessing.fetch_data(lat=lat, lon=lon)
    df = preprocessing.convert_data(result)
    aggregated_df = preprocessing.aggregate_data(df)
    preprocessed_df = preprocessing.preprocess_data(aggregated_df)

    # Create lewagon_X with the first 730 rows
    lewagon_X = preprocessed_df.iloc[:730].copy()

    # Create lewagon_y with the last 365 rows, only the 'pv_output' column
    lewagon_y = preprocessed_df.iloc[-365:][['pv_output']].copy()

    # Reshape lewagon_X and lewagon_y to the shape (samples, timesteps, features)
    lewagon_X_reshaped = lewagon_X.values.reshape((1, 730, lewagon_X.shape[1]))
    lewagon_y_reshaped = lewagon_y.values.reshape((1, 365, lewagon_y.shape[1]))

    return lewagon_X_reshaped, lewagon_y_reshaped

#lewagon_X_reshaped, lewagon_y_reshaped = preprocessing_data()
#print(lewagon_X_reshaped.shape, lewagon_y_reshaped.shape)

# Load the saved RNN model
def load_our_model():
    model = load_model('eu_RNN_model')
    return model

# Generate predictions for PV production
def predict(lewagon_X_reshaped, model):

    y_lewagon_pred = model.predict(lewagon_X_reshaped, verbose = 0)
    return y_lewagon_pred

# Visualize the predictions
def visualize(y_lewagon_pred):
    # Flatten the predictions to 1D (if necessary)
    y_lewagon_pred_flat = y_lewagon_pred.flatten()

    # Create an index for the x-axis
    x = range(len(y_lewagon_pred_flat))

    # Plot the predictions
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=x, y=y_lewagon_pred_flat)
    plt.xlabel("Time (days)")
    plt.ylabel("Predicted PV Output")
    plt.title("Predicted Photovoltaic Output Over Time")
    plt.show()


def format_predictions(y_lewagon_pred):
    """
    Formats the predictions into a DataFrame with the following columns:
    - 'date': A datetime object representing dates from 01/01/2023 to 31/12/2023
    - 'predicted_output': The values from y_lewagon_pred flattened from columns

    Args:
    - y_lewagon_pred (numpy array): Array of predicted values with shape (1, 365)

    Returns:
    - pd.DataFrame: Formatted DataFrame
    """
    # Ensure the input is a 2D array with 1 row and 365 columns
    if len(y_lewagon_pred.shape) == 2 and y_lewagon_pred.shape[0] == 1:
        y_lewagon_pred_flat = y_lewagon_pred.flatten()
    else:
        raise ValueError("y_lewagon_pred must have a shape of (1, 365)")

    # Generate date range for the year 2023
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")

    # Create DataFrame with two columns: date and predicted_output
    formatted_df = pd.DataFrame({
        "date": dates,
        "predicted_output": y_lewagon_pred_flat
    })

    return formatted_df

# Run the whole process
def predict_on_website(lat, lon):
    lewagon_X_reshaped, lewagon_y_reshaped = preprocessing_data(lat=lat, lon=lon)
    model = load_our_model()
    y_lewagon_pred = predict(lewagon_X_reshaped, model)
    y_lewagon_pred_df = format_predictions(y_lewagon_pred)
    #sum_y_lewagon_pred = y_lewagon_pred.sum()
    line_graph = visualize(y_lewagon_pred)
    return line_graph

our_line_graph = predict_on_website(33.4489, 70.6693)
print(our_line_graph)
