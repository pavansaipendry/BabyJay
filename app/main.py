"""
BabyJay API - Main Application
==============================
FastAPI server with chat, history, and authentication.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# from app.api.chat import router as chat_router
from app.api.feedback import router as feedback_router
from app.api.rate_limit import rate_limit_middleware, router as rate_limit_router


# Load environment variables
load_dotenv()

# Import routes
from app.routers.api_routes import router as api_router

# Create FastAPI app
app = FastAPI(
    title="BabyJay API",
    description="KU Campus Assistant - Chat API with conversation history",
    version="2.0.0"
)

app.middleware("http")(rate_limit_middleware)

# CORS configuration - Allow frontend to call API
# Update these origins when you deploy
ALLOWED_ORIGINS = [
    "http://localhost:3000",          # Local React dev
    "http://localhost:5173",          # Local Vite dev
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://babyjay.netlify.app",    # Production frontend
    "https://*.netlify.app",
    "https://baby-jay-frontend.vercel.app",          # Any Netlify preview
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)
app.include_router(feedback_router)
app.include_router(rate_limit_router)


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "BabyJay API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )