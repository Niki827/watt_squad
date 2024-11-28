import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

## lists of features:


# f_time = ['time']


## our custom functions:

def log_transformed(data):
    """ replaces values in columns in a dataframe with the log values """
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
    for col in f_logs:
        data[col] = np.log(data[col] + 1e-5)
    return data

def time_transformed(data):
    """takes a df and splits the 'time' feature into three features: hour, month, season;
    drops the original time column"""

    feature = pd.to_datetime(data.time)

    hour = feature.dt.hour
    month  = feature.dt.month

    def assign_season(month):
        if month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        elif month in [9, 10, 11]:
            return 3  # Fall
        else:  # December, January, February
            return 4  # Winter

    season = month.apply(assign_season)
    hour_sine = np.sin(2 * np.pi * hour / 24)
    hour_cosine = np.cos(2 * np.pi * hour / 24)
    month_sine = np.sin(2 * np.pi * month / 12)
    month_cosine = np.cos(2 * np.pi * month / 12)
    season_sine = np.sin(2 * np.pi * season / 4)
    season_cosine = np.cos(2 * np.pi * season / 4)

    data["hour_sine"] = hour_sine
    data["hour_cosine"] = hour_cosine
    data["month_sine"] = month_sine
    data["month_cosine"] = month_cosine
    data["season_sine"] = season_sine
    data["season_cosine"] = season_cosine

    data = data.drop(columns=["time"])

    return data

def degree_transformed(data):
    """ takes a df 'data' and takes the features with degree units (in the specific list f_degree);
    creates a sin and cos column for each to make them cyclical. drops the original columns"""

    f_degree = ['sun_azimuth:d', 'wind_dir_2m:d', 'wind_dir_10m:d', 'wind_dir_50m:d', 'wind_dir_100m:d']

    for col in f_degree:
        sin_column = np.sin(2 * np.pi * data[col]/360)
        cos_column = np.cos(2 * np.pi * data[col]/360)

        data[f"sin_{col}"] = sin_column
        data[f"cos_{col}"] = cos_column
        data = data.drop(columns=[col])

    return data

def transform_data(data):
    """ applies the above three functions to the input dataframe """
    data = degree_transformed(time_transformed(log_transformed(data)))

    all_col = list(data.columns)

    # defining the columns we don't want in our X_train
    drop_col = ['pv_production',
            'wind_production',
            'consumption',
            'spot_market_price',
            'precip_type:idx']

    f_ohe = ['precip_type:idx']

    scale_col = [col for col in all_col if col not in drop_col and f_ohe]

    # defining our scalers
    minmax = MinMaxScaler()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output = False)


    # our preproc pipline
    preproc = make_column_transformer(
        (ohe, f_ohe),
        (minmax, scale_col),
        remainder = "drop"
    )

    data_transformed = preproc.fit_transform(data)
    data_transformed = pd.DataFrame(data_transformed, columns=preproc.get_feature_names_out())
    data_transformed['onehotencoder__precip_type:idx_2.0'] = 0

    print('➡️ preprocessing done')
    return data_transformed



# ## building the pipeline

# data = pd.read_csv("raw_data/train.csv")

# # calling our custom functions on our dataframe
# data_ft = degree_transformed(time_transformed(log_transformed(data)))

# all_col = list(data_ft.columns)

# # defining the columns we don't want in our X_train
# drop_col = ['pv_production',
#             'wind_production',
#             'consumption',
#             'spot_market_price',
#             'precip_type:idx']

# # defining the columns we want to scale
# scale_col = [col for col in all_col if col not in drop_col and f_ohe]

# # defining our scalers
# minmax = MinMaxScaler()
# ohe = OneHotEncoder(handle_unknown='ignore', sparse_output = False)

# # our preproc pipline
# preproc = make_column_transformer(
#     (ohe, f_ohe),
#     (minmax, scale_col),
#     remainder = "drop"
# )

# data_transformed = preproc.fit_transform(data_ft)
# data_transformed = pd.DataFrame(data_transformed, columns=preproc.get_feature_names_out())
