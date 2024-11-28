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


maybe add to Makefile: (when youre running local, run "uvicorn fast:app --reload" in CLI!)
run_api:
	uvicorn taxifare.api.fast:app --reload

 Your API is available locally on port 8000, unless otherwise specified 👉 http://localhost:8000 or http://127.0.0.1:8000


"""
import pandas as pd
# $WIPE_BEGIN

# from taxifare.ml_logic.registry import load_model
# from taxifare.ml_logic.preprocessor import preprocess_features
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

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


## PREDICT ENDPOINT

# app.state.model = load_model()

# import api_preproc

# @app.get("/predict")
# def predict(
#     # eg filepath??
# )

@app.get("/calculate")
def mock_calc(first, second):
    # $CHA_BEGIN
    calculation = int(first) * int(second)
    return dict(result=calculation)
    # $CHA_END
