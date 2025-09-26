#!/usr/bin/env python3
"""
AI Prediction Service with Gemini Integration
Updated for Pydantic V2 compatibility
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator, Annotated
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

import requests
import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator, ConfigDict
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, select, desc, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
import uvicorn

# --- Configuration and Logging Setup ---
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai-prediction-service")

# Configuration with validation
class Config:
    """Application configuration with environment variable validation"""
    
    def __init__(self):
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.schedule_hours = float(os.getenv("SCHEDULE_HOURS", "3"))
        self.database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./predictions.db")
        self.coingecko_api_base = os.getenv("COINGECKO_API_BASE", "https://api.coingecko.com/api/v3")
        self.prompt_version = int(os.getenv("PROMPT_VERSION", "2"))
        
        # Gemini-specific config
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        
        # Service config
        self.predict_symbols = os.getenv("PREDICT_SYMBOLS", "bitcoin,ethereum").split(",")
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # Validate required configuration
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider")
        
        # Initialize Gemini
        if self.llm_provider == "gemini":
            genai.configure(api_key=self.google_api_key)

config = Config()

# --- Database Models ---
class Base(DeclarativeBase):
    pass

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    probability_up = Column(Float, nullable=False)
    probability_down = Column(Float, nullable=False)
    commentary = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.0)
    source_data = Column(JSON, nullable=False)
    model_meta = Column(JSON, nullable=False)
    processing_time_ms = Column(Integer, default=0)

class PredictionAudit(Base):
    __tablename__ = "prediction_audit"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), nullable=False)  # success, error
    error_message = Column(Text)
    processing_time_ms = Column(Integer)

# --- Database Setup ---
engine = create_async_engine(
    config.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in config.database_url else {},
    pool_pre_ping=True,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true"
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# --- Pydantic Schemas (Updated for V2) ---
class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    symbol: str = Field(..., min_length=1, max_length=50, description="Cryptocurrency symbol")
    
    @field_validator('symbol')
    @classmethod
    def symbol_lowercase(cls, v: str) -> str:
        return v.lower()

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    symbol: str
    timestamp: datetime
    probability_up: float = Field(..., ge=0, le=100)
    probability_down: float = Field(..., ge=0, le=100)
    commentary: str
    confidence_score: Optional[float] = None
    processing_time_ms: Optional[int] = None
    source_data: Dict[str, Any]
    model_meta: Dict[str, Any]

class PredictionListResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    page: int
    size: int

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    llm_provider: str
    database_status: str
    coingecko_status: str

class StatsResponse(BaseModel):
    total_predictions: int
    predictions_by_symbol: Dict[str, int]
    average_processing_time_ms: float
    success_rate: float

# --- Service Classes ---
class MarketDataService:
    """Service for fetching cryptocurrency market data using requests"""
    
    def __init__(self):
        self.base_url = config.coingecko_api_base
        self.timeout = config.request_timeout

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data from CoinGecko API using async thread pool"""
        try:
            # Use async thread pool for HTTP requests
            loop = asyncio.get_event_loop()
            market_data = await loop.run_in_executor(
                None, 
                self._sync_get_market_data, 
                symbol
            )
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch market data: {str(e)}"
            )

    def _sync_get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Synchronous method to fetch market data"""
        try:
            # First, try the detailed endpoint
            url = f"{self.base_url}/coins/{symbol}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                market_data = data.get("market_data", {})
                return {
                    "symbol": symbol,
                    "name": data.get("name", symbol),
                    "price_usd": market_data.get("current_price", {}).get("usd"),
                    "price_change_24h": market_data.get("price_change_24h"),
                    "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
                    "market_cap": market_data.get("market_cap", {}).get("usd"),
                    "volume_24h": market_data.get("total_volume", {}).get("usd"),
                    "high_24h": market_data.get("high_24h", {}).get("usd"),
                    "low_24h": market_data.get("low_24h", {}).get("usd"),
                    "last_updated": data.get("last_updated"),
                    "source": "coingecko"
                }
            else:
                # Fallback to simple price endpoint
                return self._fallback_market_data(symbol)
                
        except requests.exceptions.Timeout:
            raise Exception("CoinGecko API timeout")
        except requests.exceptions.RequestException as e:
            raise Exception(f"CoinGecko API error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")

    def _fallback_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback method using simple price endpoint"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": symbol,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true"
            }
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                if symbol in data:
                    coin_data = data[symbol]
                    return {
                        "symbol": symbol,
                        "price_usd": coin_data.get("usd"),
                        "price_change_percentage_24h": coin_data.get("usd_24h_change"),
                        "market_cap": coin_data.get("usd_market_cap"),
                        "volume_24h": coin_data.get("usd_24h_vol"),
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "source": "coingecko_simple"
                    }
            
            raise Exception(f"Symbol {symbol} not found in CoinGecko response")
            
        except Exception as e:
            raise Exception(f"Fallback market data failed: {e}")

class LLMService:
    """Service for interacting with Gemini LLM"""
    
    def __init__(self):
        if config.llm_provider == "gemini":
            self.model = genai.GenerativeModel(config.gemini_model)
            self.generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=config.llm_temperature
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

    def build_prompt(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """Build the prompt for the LLM"""
        prompt = f"""
        As a expert cryptocurrency analyst, analyze this market data for {symbol} and provide a 24-hour price prediction.

        CRITICAL: You MUST return ONLY valid JSON with this exact structure:
        {{
            "probability_up": number between 0-100,
            "probability_down": number between 0-100,
            "commentary": "2-3 sentence analysis",
            "reasoning": "brief reasoning for your prediction"
        }}

        The probabilities must sum to 100.

        MARKET DATA:
        {json.dumps(market_data, indent=2)}

        Analyze price trends, volume, market cap, and recent changes. Be objective and data-driven.
        """

        # Add context based on price movement
        price_change = market_data.get('price_change_percentage_24h', 0)
        if price_change and abs(price_change) > 5:
            prompt += f"\nNote: Significant 24h price movement ({price_change:.2f}%) detected."

        return prompt

    async def generate_prediction(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction using Gemini LLM"""
        start_time = datetime.now(timezone.utc)
        
        for attempt in range(config.max_retries):
            try:
                prompt = self.build_prompt(symbol, market_data)
                response = await self.model.generate_content_async(
                    prompt, 
                    generation_config=self.generation_config
                )
                
                # Parse and validate response
                result = json.loads(response.text)
                
                # Validate required fields
                required_fields = ['probability_up', 'probability_down', 'commentary']
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields in LLM response")
                
                # Normalize probabilities to sum to 100
                prob_up = float(result['probability_up'])
                prob_down = float(result['probability_down'])
                total = prob_up + prob_down
                
                if total > 0:
                    prob_up = (prob_up / total) * 100
                    prob_down = 100 - prob_up
                else:
                    prob_up, prob_down = 50.0, 50.0
                
                # Calculate confidence score
                confidence = abs(prob_up - prob_down) / 100.0
                
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                logger.info(f"LLM prediction generated for {symbol} in {processing_time:.2f}ms")
                
                return {
                    'probability_up': round(prob_up, 2),
                    'probability_down': round(prob_down, 2),
                    'commentary': result['commentary'].strip(),
                    'reasoning': result.get('reasoning', ''),
                    'confidence_score': round(confidence, 3),
                    'processing_time_ms': int(processing_time)
                }
                
            except json.JSONDecodeError as e:
                logger.warning(f"LLM JSON parse error on attempt {attempt + 1}: {e}")
                if attempt == config.max_retries - 1:
                    raise HTTPException(
                        status_code=500,
                        detail="LLM returned invalid JSON format"
                    )
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt == config.max_retries - 1:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate prediction: {str(e)}"
                    )
                await asyncio.sleep(1)  # Brief backoff

class PredictionService:
    """Main service for handling predictions"""
    
    def __init__(self):
        self.market_data_service = MarketDataService()
        self.llm_service = LLMService()
        self.start_time = datetime.now(timezone.utc)

    async def audit_prediction(self, db: AsyncSession, symbol: str, status: str, 
                             error_message: str = None, processing_time: int = None):
        """Audit prediction attempts"""
        audit = PredictionAudit(
            symbol=symbol,
            status=status,
            error_message=error_message,
            processing_time_ms=processing_time
        )
        db.add(audit)
        await db.commit()

    async def generate_prediction(self, symbol: str, db: AsyncSession) -> Prediction:
        """Generate a complete prediction for a symbol"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Fetch market data
            market_data = await self.market_data_service.get_market_data(symbol)
            
            # Generate LLM prediction
            llm_result = await self.llm_service.generate_prediction(symbol, market_data)
            
            # Create prediction record
            prediction = Prediction(
                symbol=symbol,
                probability_up=llm_result['probability_up'],
                probability_down=llm_result['probability_down'],
                commentary=llm_result['commentary'],
                confidence_score=llm_result['confidence_score'],
                source_data=market_data,
                model_meta={
                    "provider": config.llm_provider,
                    "model": config.gemini_model,
                    "prompt_version": config.prompt_version,
                    "temperature": config.llm_temperature
                },
                processing_time_ms=llm_result['processing_time_ms']
            )
            
            db.add(prediction)
            await db.commit()
            await db.refresh(prediction)
            
            # Audit success
            await self.audit_prediction(
                db, symbol, "success", 
                processing_time=llm_result['processing_time_ms']
            )
            
            logger.info(f"Successfully generated prediction for {symbol}: "
                       f"UP {llm_result['probability_up']}%")
            return prediction
            
        except HTTPException:
            raise
        except Exception as e:
            processing_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            error_msg = str(e)
            
            # Audit failure
            await self.audit_prediction(db, symbol, "error", error_msg, processing_time_ms)
            
            logger.error(f"Prediction failed for {symbol}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    async def get_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get service statistics"""
        # Total predictions
        total_result = await db.execute(select(func.count(Prediction.id)))
        total_predictions = total_result.scalar() or 0
        
        # Predictions by symbol
        symbol_result = await db.execute(
            select(Prediction.symbol, func.count(Prediction.id))
            .group_by(Prediction.symbol)
        )
        predictions_by_symbol = dict(symbol_result.all())
        
        # Average processing time
        avg_time_result = await db.execute(select(func.avg(Prediction.processing_time_ms)))
        avg_processing_time = avg_time_result.scalar() or 0
        
        # Success rate (from audit table)
        total_audits = await db.execute(select(func.count(PredictionAudit.id)))
        success_audits = await db.execute(
            select(func.count(PredictionAudit.id))
            .where(PredictionAudit.status == "success")
        )
        total_count = total_audits.scalar() or 1
        success_count = success_audits.scalar() or 0
        success_rate = (success_count / total_count) * 100
        
        return {
            "total_predictions": total_predictions,
            "predictions_by_symbol": predictions_by_symbol,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "success_rate": round(success_rate, 2)
        }

# --- FastAPI Application ---
prediction_service = PredictionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting AI Prediction Service...")
    
    # Initialize database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Start scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scheduled_predictions,
        IntervalTrigger(hours=config.schedule_hours),
        id="scheduled_predictions"
    )
    scheduler.start()
    
    logger.info(f"Scheduler started: predictions every {config.schedule_hours} hours")
    logger.info(f"Monitoring symbols: {', '.join(config.predict_symbols)}")
    
    yield
    
    # Shutdown
    scheduler.shutdown()
    logger.info("AI Prediction Service stopped")

# Enhanced CORS configuration
def setup_cors(app: FastAPI):
    # List of allowed origins - update with your frontend URLs
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://.onrender.com",  # Add your production domain
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Update your app initialization:
app = FastAPI(
    title="AI Prediction Service (Gemini)",
    description="Cryptocurrency price prediction service using Google Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Setup CORS
setup_cors(app)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AI Prediction Service (Gemini) - Healthy"}

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Comprehensive health check endpoint"""
    try:
        # Test database
        await db.execute(select(1))
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Test CoinGecko
    try:
        response = requests.get(f"{config.coingecko_api_base}/ping", timeout=5)
        coingecko_status = "healthy" if response.status_code == 200 else f"error: {response.status_code}"
    except Exception as e:
        coingecko_status = f"error: {str(e)}"
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        llm_provider=config.llm_provider,
        database_status=db_status,
        coingecko_status=coingecko_status
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_now(request: PredictRequest, db: AsyncSession = Depends(get_db)):
    """Generate a new prediction for a symbol"""
    prediction = await prediction_service.generate_prediction(request.symbol, db)
    return prediction

@app.get("/predictions/{symbol}", response_model=PredictionListResponse)
async def get_predictions(
    symbol: str, 
    limit: int = 20, 
    page: int = 1,
    db: AsyncSession = Depends(get_db)
):
    """Get prediction history for a symbol"""
    if page < 1 or limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")
    
    offset = (page - 1) * limit
    
    # Get total count
    total_result = await db.execute(
        select(func.count(Prediction.id))
        .where(Prediction.symbol == symbol.lower())
    )
    total = total_result.scalar() or 0
    
    # Get predictions
    predictions_result = await db.execute(
        select(Prediction)
        .where(Prediction.symbol == symbol.lower())
        .order_by(desc(Prediction.timestamp))
        .offset(offset)
        .limit(limit)
    )
    predictions = predictions_result.scalars().all()
    
    return PredictionListResponse(
        predictions=predictions,
        total=total,
        page=page,
        size=len(predictions)
    )

@app.get("/predictions/{symbol}/latest", response_model=PredictionResponse)
async def get_latest_prediction(symbol: str, db: AsyncSession = Depends(get_db)):
    """Get the latest prediction for a symbol"""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.symbol == symbol.lower())
        .order_by(desc(Prediction.timestamp))
        .limit(1)
    )
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions found for symbol")
    
    return prediction

@app.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get service statistics"""
    stats = await prediction_service.get_stats(db)
    return StatsResponse(**stats)

# --- Scheduler Function ---
async def scheduled_predictions():
    """Run scheduled predictions for all configured symbols"""
    logger.info(f"Running scheduled predictions for: {config.predict_symbols}")
    
    async with AsyncSessionLocal() as db:
        for symbol in config.predict_symbols:
            try:
                prediction = await prediction_service.generate_prediction(symbol, db)
                logger.info(f"Scheduled prediction successful for {symbol}: "
                          f"UP {prediction.probability_up:.1f}%")
            except Exception as e:
                logger.error(f"Scheduled prediction failed for {symbol}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Use import string for uvicorn reload to work properly
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )