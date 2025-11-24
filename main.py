"""
Fraud Detection API - Main Application

Entry point for the fraud detection service.
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Fraud Detection API",
    description="""
    A real-time fraud detection API for financial transactions.

    ## Features
    * Real-time fraud prediction
    * Rule-based detection engine
    * Risk scoring and categorization
    * Detailed fraud indicators

    ## How to use
    1. Send a POST request to `/api/v1/predict` with transaction data
    2. Receive fraud prediction with score and risk level
    3. Take action based on the risk level
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# WARNING: allow_origins=["*"] is for development only
# In production, specify exact domains to prevent CSRF attacks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint - provides API information."""
    return {
        "message": "Welcome to the Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "predict_endpoint": "/api/v1/predict",
    }
