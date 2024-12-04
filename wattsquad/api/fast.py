"""
just some notes:

maybe add to requirements.txt:
# API
fastapi         # API framework
pytz            # time zone management
uvicorn         # web server
# tests
httpx           # HTTP client
pytest-asyncio  # asynchronous I/O support for pytest


maybe add to Makefile: (when youre running local, run "uvicorn wattsquad.api.fast:app --reload" in CLI!)
run_api:
	uvicorn taxifare.api.fast:app --reload

 Your API is available locally on port 8000, unless otherwise specified 👉 http://localhost:8000 or http://127.0.0.1:8000


"""

# $WIPE_BEGIN

# from taxifare.ml_logic.registry import load_model
# from taxifare.ml_logic.preprocessor import preprocess_features
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from wattsquad.eu_logic.eu_output import predict_on_website
from wattsquad.ml_logic.calculations import cost_saving


class DataFrameRequest(BaseModel):
    data: list[dict]
# from wattsquad import preproc

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


## TESTING API WITH DUMMY CALCULATE ENDPOINT

# @app.get("/calculate")
# def mock_calc(first, second):
#     # $CHA_BEGIN
#     calculation = int(first) * int(second)
#     return dict(result=calculation)
#     # $CHA_END


## PREDICT ENDPOINT

#@app.post("/predict")
# def predict(request: DataFrameRequest):
#     print(request)
#     data = pd.DataFrame(request.data)
#     print(data)
#     return {"message": "DataFrame received successfully",
#             "df": data.to_dict(orient='records')}

# app.state.model = load_model()

@app.get("/predict")
def predict(flexibility_degree):

    my_cost_savings = cost_saving(flexibility_degree=float(flexibility_degree))
    print(my_cost_savings)


    return {"message": f"Cost Savings for {int(float(flexibility_degree)*100)}% flexibility degree",
            "df": my_cost_savings[0].to_dict(orient='records'),
            "costs_no_shift" : my_cost_savings[1],
            "costs_with_shift" : my_cost_savings[2]}



@app.get("/eu_predict")
def eu_predict(lat, lon):

    df_eu = predict_on_website(lat=lat, lon=lon)

    return {"message": f"Solar Photovoltaic Production Forecast for {lat} latitude and {lon} longitude",
            "df": df_eu.to_dict(orient='records')}
