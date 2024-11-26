import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#Reading Raw X_train
X_train = pd.read_csv('raw_data/train.csv')
X_test = pd.read_csv('raw_data/test.csv')

#Log columns
f_logs = [
    'precip_1h:mm',
    'prob_precip_1h:p',
    'clear_sky_rad:W',
    'clear_sky_energy_1h:J',
    'diffuse_rad:W',
    'diffuse_rad_1h:Wh',
    'direct_rad:W',
    'direct_rad_1h:Wh',
    'global_rad:W',
    'global_rad_1h:Wh',
    'wind_speed_2m:ms',
    'wind_speed_10m:ms',
    'wind_speed_50m:ms',
    'wind_speed_100m:ms'
]

epsilon = 1e-5

for f in f_logs:
    X_train[f] = np.log(X_train[f] + epsilon)
    X_test[f] = np.log(X_test[f] + epsilon)

# Converting time to datetime
# We might have done that before already
X_train['time']= pd.to_datetime(X_train['time'])
X_test['time']= pd.to_datetime(X_test['time'])

#the following two steps creates new columns to get the input for the sine & cosine columns
#creating columns indicating the hour and the month
X_train['hour'] = X_train['time'].dt.hour
X_train['month'] = X_train['time'].dt.month

X_test['hour'] = X_test['time'].dt.hour
X_test['month'] = X_test['time'].dt.month

#creating column indicating the season
def assign_season(month):
    if month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    elif month in [9, 10, 11]:
        return 3  # Fall
    else:  # December, January, February
        return 4  # Winter

# X_train
X_train['season'] = X_train['month'].apply(assign_season)

X_train['hour_sine'] = np.sin(2 * np.pi * X_train['hour'] / 24)
X_train['hour_cosine'] = np.cos(2 * np.pi * X_train['hour'] / 24)

X_train['month_sine'] = np.sin(2 * np.pi * X_train['month'] / 12)
X_train['month_cosine'] = np.cos(2 * np.pi * X_train['month'] / 12)

X_train['season_sine'] = np.sin(2 * np.pi * X_train['season'] / 4)
X_train['season_cosine'] = np.cos(2 * np.pi * X_train['season'] / 4)

X_train = X_train.drop(columns = ['hour', 'month', 'season'])

# X_test
X_test['season'] = X_test['month'].apply(assign_season)

X_test['hour_sine'] = np.sin(2 * np.pi * X_test['hour'] / 24)
X_test['hour_cosine'] = np.cos(2 * np.pi * X_test['hour'] / 24)

X_test['month_sine'] = np.sin(2 * np.pi * X_test['month'] / 12)
X_test['month_cosine'] = np.cos(2 * np.pi * X_test['month'] / 12)

X_test['season_sine'] = np.sin(2 * np.pi * X_test['season'] / 4)
X_test['season_cosine'] = np.cos(2 * np.pi * X_test['season'] / 4)

X_test = X_test.drop(columns = ['hour', 'month', 'season'])

# Cyclic features
cyclical_features = ['sun_azimuth:d', 'wind_dir_2m:d', 'wind_dir_10m:d', 'wind_dir_50m:d', 'wind_dir_100m:d']
degrees = 360

for cyclical_feature in cyclical_features:
    sin_column_name = f'sin_{cyclical_feature}'
    cos_column_name = f'cos_{cyclical_feature}'
    X_train[sin_column_name] = np.sin(2 * np.pi * X_train[cyclical_feature]/2)
    X_train[cos_column_name] = np.cos(2 * np.pi * X_train[cyclical_feature]/degrees)
    X_test[sin_column_name] = np.sin(2 * np.pi * X_test[cyclical_feature]/2)
    X_test[cos_column_name] = np.cos(2 * np.pi * X_test[cyclical_feature]/degrees)

X_train = X_train.drop(columns=cyclical_features)
X_test = X_test.drop(columns=cyclical_features)

# targets = ['pv_production', 'wind_production', 'consumption']
f_minmax = [
    'hour_sine',
    'hour_cosine',
    'month_sine',
    'month_cosine',
    'season_sine',
    'season_cosine',
    'precip_1h:mm',
    'prob_precip_1h:p',
    'clear_sky_rad:W',
    'clear_sky_energy_1h:J',
    'diffuse_rad:W',
    'diffuse_rad_1h:Wh',
    'direct_rad:W',
    'direct_rad_1h:Wh',
    'global_rad:W',
    'global_rad_1h:Wh',
    'sunshine_duration_1h:min',
    'low_cloud_cover:p',
    'medium_cloud_cover:p',
    'high_cloud_cover:p',
    'total_cloud_cover:p',
    'effective_cloud_cover:p',
    'sin_sun_azimuth:d',
    'cos_sun_azimuth:d',
    'sin_wind_dir_2m:d',
    'cos_wind_dir_2m:d',
    'sin_wind_dir_10m:d',
    'cos_wind_dir_10m:d',
    'sin_wind_dir_50m:d',
    'cos_wind_dir_50m:d',
    'sin_wind_dir_100m:d',
    'cos_wind_dir_100m:d',
    'relative_humidity_2m:p',
    'relative_humidity_10m:p',
    'relative_humidity_50m:p',
    'relative_humidity_100m:p',
    'dew_point_2m:C',
    'dew_point_10m:C',
    'dew_point_50m:C',
    'dew_point_100m:C',
    'temp'
]
f_standard = ['sun_elevation:d']
f_robust = [
    't_10m:C',
    't_50m:C',
    't_100m:C',
    'wind_speed_2m:ms',
    'wind_speed_10m:ms',
    'wind_speed_50m:ms',
    'wind_speed_100m:ms'
]

f_ohe = ['precip_type:idx']

# other = ['spot_market_price']

# target
y = X_train[['pv_production', 'wind_production', 'consumption']]
# features
X_train = X_train.drop(columns=['time', 'pv_production', 'wind_production', 'consumption', 'spot_market_price'])
X_test = X_test.drop(columns=['time', 'pv_production', 'wind_production', 'consumption', 'spot_market_price'])

# Preprocessing Pipeline
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
cat_transformer = OneHotEncoder()
robust_scaler = RobustScaler()

preproc_basic = make_column_transformer(
    (minmax_scaler, f_minmax ),
    (standard_scaler, f_standard),
    (robust_scaler, f_robust),
    (cat_transformer, f_ohe),
    remainder='passthrough'
)
# Train X
X_train_transformed = preproc_basic.fit_transform(X_train)

# Adding Column names
X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=preproc_basic.get_feature_names_out()
)
# Test x
X_test_transformed = preproc_basic.transform(X_test)
# Adding Column names
X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=preproc_basic.get_feature_names_out()
)
