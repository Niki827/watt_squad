"""

 Your API is available locally on port 8000, unless otherwise specified 👉  http://127.0.0.1:8000

"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from wattsquad.mr_worldwide.eu_output import predict_on_website
from wattsquad.ml_logic.calculations import cost_saving
from wattsquad.ml_logic.battery_logic import selling_electricity


class DataFrameRequest(BaseModel):
    data: list[dict]

app = FastAPI()
# data = pd.read_csv("raw_data/train.csv")
# print('preproc')
# print(preproc.transform_data(data))

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

## ROOT ENDPOINT

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="api works, you're a genius, go grab a drink")
    # $CHA_END


@app.get("/predict")
def predict(flexibility_degree):

    my_cost_savings = cost_saving(flexibility_degree=float(flexibility_degree))

    return {"message": f"Cost Savings for {int(float(flexibility_degree)*100)}% flexibility degree",
            "df": my_cost_savings[0].to_dict(orient='records'),
            "costs_no_shift" : my_cost_savings[1],
            "costs_with_shift" : my_cost_savings[2]}


@app.get("/eu_predict")
def eu_predict(lat, lon):

    df_eu = predict_on_website(lat=lat, lon=lon)

    return {"message": f"Solar Photovoltaic Production Forecast for {lat} latitude and {lon} longitude",
            "df": df_eu.to_dict(orient='records')}

@app.get("/battery_product")
def battery_product(battery_capacity, electricity_price_share):

    battery_tuple = selling_electricity(int(battery_capacity), float(electricity_price_share))


    return {"message": f"Costs saved for {battery_capacity} battery_capacity and {electricity_price_share} electricity price share",
            "df_june": battery_tuple[0].to_dict(orient='records'),
            "df_december": battery_tuple[1].to_dict(orient='records'),
            "electricity_sold_kwH":   battery_tuple[2],
            "electricity_sold_NOK":   battery_tuple[3],
            "electricity_bought_NOK": battery_tuple[4]
            }
