import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


## PRECIPITATION TYPE

# change dtype from float to int (optional)
df = df.astype({'precip_type:idx': 'int32'})

# Instantiate the OneHotEncoder
ohe = OneHotEncoder(sparse_output = False)

# Fit encoder
ohe.fit(df[['precip_type:idx']])

# Transform the current "precip_type" column
df[ohe.get_feature_names_out()] = ohe.transform(df[['precip_type:idx']])

# Drop the original column. can assign to the OG dataframe!
df_precip = df.drop(columns = ['precip_type:idx'])
