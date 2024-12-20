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
    # Flatten the predictions to 1D
    y_lewagon_pred_flat = y_lewagon_pred.flatten()

    # Ensure we have 12 data points (one per month)
    if len(y_lewagon_pred_flat) != 12:
        raise ValueError("Expected 12 monthly values for y_lewagon_pred.")

    # Month labels (modify as needed)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Numeric x-axis positions for the months
    x = range(12)

    # Plot a single set of bars for PV production
    # Use a green color similar to the 'Production' bars in your provided code
    ax.bar(x, y_lewagon_pred_flat, color='lightgreen', width=0.4)

    # Set titles and labels
    ax.set_title('Monthly Predicted PV Output', fontsize=16)
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Energy (kWh)', fontsize=14)

    # Set the x-ticks and x-labels
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=10)

    fig.tight_layout()

    # Display the plot
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

    # Aggregate daily predictions into monthly totals
    monthly_preds = (y_lewagon_pred_df
                     .set_index('date')
                     .resample('M')['predicted_output']
                     .sum()
                     .values)

    # Now monthly_preds has 12 values, one for each month
    line_graph = visualize(monthly_preds)
    return line_graph

our_line_graph = predict_on_website(-33.834, 151.209)
print(our_line_graph)
