"""
Fraud Detection API - Main Application

This is the entry point for the FastAPI application.
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Create FastAPI application instance
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
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
)

# Add CORS middleware (for web frontends)
# In production, configure this properly with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint - provides API information."""
    return {
        "message": "Welcome to the Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
