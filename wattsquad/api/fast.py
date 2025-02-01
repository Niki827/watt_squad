"""
WattSquad API - Energy Management and Prediction Service

This API provides endpoints for energy cost savings calculations, EU solar predictions,
and battery product analysis.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, confloat, conint
from typing import List, Dict, Tuple, Any
import logging
from wattsquad.mr_worldwide.eu_output import predict_on_website
from wattsquad.ml_logic.calculations import cost_saving
from wattsquad.ml_logic.battery_logic import selling_electricity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WattSquad Energy API",
    description="API for energy management, predictions, and cost optimization",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class CostSavingsResponse(BaseModel):
    message: str
    df: List[Dict[str, Any]]
    costs_no_shift: float
    costs_with_shift: float

class EUPredictionResponse(BaseModel):
    message: str
    df: List[Dict[str, Any]]

class BatteryProductResponse(BaseModel):
    message: str
    df_june: List[Dict[str, Any]]
    df_december: List[Dict[str, Any]]
    electricity_sold_kwH: float
    electricity_sold_NOK: float
    electricity_bought_NOK: float

@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint to verify API is running.

    Returns:
        Dict[str, str]: A greeting message
    """
    return {"greeting": "api works, you're a genius, go grab a drink"}

@app.get("/predict", response_model=CostSavingsResponse)
async def predict(
    flexibility_degree: float = Query(..., ge=0, le=1, description="Flexibility degree between 0 and 1")
) -> CostSavingsResponse:
    """Calculate cost savings based on flexibility degree.

    Args:
        flexibility_degree (float): Degree of flexibility (0-1)

    Returns:
        CostSavingsResponse: Cost savings analysis results

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(f"Calculating cost savings for flexibility degree: {flexibility_degree}")
        my_cost_savings = cost_saving(flexibility_degree=flexibility_degree)
        
        return CostSavingsResponse(
            message=f"Cost Savings for {int(flexibility_degree*100)}% flexibility degree",
            df=my_cost_savings[0].to_dict(orient='records'),
            costs_no_shift=my_cost_savings[1],
            costs_with_shift=my_cost_savings[2]
        )
    except Exception as e:
        logger.error(f"Error calculating cost savings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/eu_predict", response_model=EUPredictionResponse)
async def eu_predict(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude")
) -> EUPredictionResponse:
    """Predict solar photovoltaic production for given coordinates.

    Args:
        lat (float): Latitude (-90 to 90)
        lon (float): Longitude (-180 to 180)

    Returns:
        EUPredictionResponse: Solar production forecast

    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Predicting solar production for coordinates: {lat}, {lon}")
        df_eu = predict_on_website(lat=lat, lon=lon)
        
        return EUPredictionResponse(
            message=f"Solar Photovoltaic Production Forecast for {lat} latitude and {lon} longitude",
            df=df_eu.to_dict(orient='records')
        )
    except Exception as e:
        logger.error(f"Error predicting EU solar production: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/battery_product", response_model=BatteryProductResponse)
async def battery_product(
    battery_capacity: int = Query(..., gt=0, description="Battery capacity in kWh"),
    electricity_price_share: float = Query(..., gt=0, lt=1, description="Electricity price share (0-1)")
) -> BatteryProductResponse:
    """Calculate battery-related costs and savings.

    Args:
        battery_capacity (int): Battery capacity in kWh
        electricity_price_share (float): Electricity price share between 0 and 1

    Returns:
        BatteryProductResponse: Battery analysis results

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(f"Calculating battery product for capacity: {battery_capacity}, price share: {electricity_price_share}")
        battery_tuple = selling_electricity(battery_capacity, electricity_price_share)
        
        return BatteryProductResponse(
            message=f"Costs saved for {battery_capacity} battery_capacity and {electricity_price_share} electricity price share",
            df_june=battery_tuple[0].to_dict(orient='records'),
            df_december=battery_tuple[1].to_dict(orient='records'),
            electricity_sold_kwH=battery_tuple[2],
            electricity_sold_NOK=battery_tuple[3],
            electricity_bought_NOK=battery_tuple[4]
        )
    except Exception as e:
        logger.error(f"Error calculating battery product: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler for generic exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )
