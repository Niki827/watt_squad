from wattsquad.ml_logic.models import predict_rnn_solar
from wattsquad.ml_logic.outputs import prediction_accuracy
from wattsquad.ml_logic.models import predict_rnn_consumption
from wattsquad.ml_logic.models import XGBRegressor_solar
from wattsquad.ml_logic.calculations import cost_saving



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data, costs_no_shift, costs_with_shift = cost_saving(0.5)

# Convert timestamp to number of hours since the first timestamp
data['hour'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600

# Plotting
plt.figure(figsize=(10, 6))

# Plotting forecasted consumption and production
plt.plot(data['hour'], data['adjusted_forecasted_consumption'], label='Forecasted Consumption', color='red', linestyle='--', marker='o')
plt.plot(data['hour'], data['adjusted_forecasted_production'], label='Forecasted Production', color='green', linestyle='-', marker='x')

# Plotting shifted consumption
plt.plot(data['hour'], data['shifted_consumption'], label='Shifted Consumption', color='blue', linestyle='-', marker='^')

# Highlight the areas where consumption was shifted (optional)
shifted_consumption = data['shifted_consumption'] < data['adjusted_forecasted_consumption']
plt.fill_between(data['hour'], data['adjusted_forecasted_consumption'], data['shifted_consumption'], where=shifted_consumption, color='blue', alpha=0.3, label="Shifted Area")

# Adding labels and title
plt.title("Forecasted Consumption and Production with Consumption Shift")
plt.xlabel('Hour')
plt.ylabel('Energy (kWh)')
plt.legend()

# Display plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# pred = predict_rnn_solar()

# print(pred)

# pred = predict_rnn_consumption()

# print(pred)

#prediction_accuracy()
#predictions_df['errorer'] = predictions_df['pv_production']predictions_df['pv_forecast']


# xgb_predictions = XGBRegressor_solar()
# print(xgb_predictions)
# x = xgb_predictions['pv_forecast']
# print(x[:24])

# test_data = pd.read_csv('raw_data/test.csv')
# y_true = test_data[['pv_production', 'wind_production', 'consumption']][:24]

# #PV production
# #True
# y_true_pv = y_true['pv_production']
# y_true_pv = np.array(y_true_pv).reshape(24,1)


# plt.figure(figsize=(10, 6))
# plt.plot(pred, label='PV Forecast', color='blue', linestyle='-')
# plt.plot(y_true_pv, label='PV Production', color='orange', linestyle='--')
# plt.xlabel('Time (hours)')
# plt.ylabel('PV Production (kWh/h)')
# plt.title('PV Forecast vs PV Production')
# plt.legend()
# plt.show()
