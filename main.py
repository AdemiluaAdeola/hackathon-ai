# ai_prediction_service.py
import os
import time
import json
from datetime import datetime, timezone
from typing import List, Optional

import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from apscheduler.schedulers.background import BackgroundScheduler
import google.generativeai as genai
from dotenv import load_dotenv

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
SCHEDULE_HOURS = float(os.getenv("SCHEDULE_HOURS", "3"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")
COINGECKO_API_BASE = os.getenv("COINGECKO_API_BASE", "https://api.coingecko.com/api/v3")
PROMPT_VERSION = 2  # Updated for Gemini 2.0

# Configure Gemini
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured successfully")

# Using Gemini 2.0 Flash - the latest and most capable model
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Alternative models if the above doesn't work
BACKUP_MODELS = [
    "gemini-2.0-flash",  # Primary stable version
    "gemini-1.5-flash-latest",  # Fallback
    "gemini-1.5-flash",  # Secondary fallback
]

# SQLAlchemy setup
Base = declarative_base()

def get_engine():
    # Default SQLite URL
    if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
        db_path = os.path.join(os.path.dirname(__file__), "predictions.db")
        database_url = f"sqlite:///{db_path}"
        print(f"📊 Using SQLite database at: {db_path}")
        return create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        print(f"📊 Using database: {DATABASE_URL}")
        return create_engine(DATABASE_URL)

engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime)
    probability_up = Column(Float)
    probability_down = Column(Float)
    commentary = Column(String)
    source_data = Column(JSON)
    model_meta = Column(JSON)

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")
except Exception as e:
    print(f"⚠️ Database creation warning: {e}")

# FastAPI app
app = FastAPI(
    title="AI Prediction Service with Gemini 2.0 Flash",
    description="Advanced cryptocurrency price prediction using Google's latest Gemini 2.0 Flash model",
    version="3.0.0"
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request/response models
class PredictRequest(BaseModel):
    symbol: str
    lookback_minutes: Optional[int] = 60

class PredictionResponse(BaseModel):
    symbol: str
    timestamp: str
    probability_up: float
    probability_down: float
    commentary: str
    source_data: dict
    model_meta: dict

# Utility: List available Gemini models
def list_available_models():
    """List available Gemini models"""
    if not GEMINI_API_KEY:
        return ["API key not configured"]
    
    try:
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                # Extract just the model name
                model_name = model.name.split('/')[-1] if '/' in model.name else model.name
                available_models.append(model_name)
        return sorted(available_models)
    except Exception as e:
        return [f"Error fetching models: {e}"]

def find_best_model():
    """Find the best available Gemini model"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        available_models = list_available_models()
        print("🔍 Available Gemini models:")
        for model in available_models:
            print(f"   - {model}")
        
        # Try to find Gemini 2.0 models first
        for model in available_models:
            if 'gemini-2.0' in model and 'flash' in model:
                print(f"🎯 Selected model: {model}")
                return model
        
        # Fallback to Gemini 1.5 flash
        for model in available_models:
            if 'gemini-1.5' in model and 'flash' in model:
                print(f"🔙 Fallback model: {model}")
                return model
        
        # Last resort - any available model
        if available_models:
            print(f"⚠️ Using available model: {available_models[0]}")
            return available_models[0]
        
        return None
    except Exception as e:
        print(f"❌ Error finding best model: {e}")
        return None

# Utility: fetch enhanced market data from CoinGecko
def fetch_coingecko_market_data(symbol: str):
    """
    Fetch comprehensive market data for better analysis
    """
    try:
        # Extended symbol mapping
        symbol_mapping = {
            'btc': 'bitcoin', 'eth': 'ethereum', 'bnb': 'binancecoin',
            'sol': 'solana', 'xrp': 'ripple', 'ada': 'cardano',
            'doge': 'dogecoin', 'matic': 'polygon', 'dot': 'polkadot',
            'avax': 'avalanche-2', 'link': 'chainlink', 'ltc': 'litecoin',
            'bch': 'bitcoin-cash', 'xlm': 'stellar', 'atom': 'cosmos',
            'fil': 'filecoin', 'etc': 'ethereum-classic', 'xmr': 'monero'
        }
        
        coin_id = symbol_mapping.get(symbol.lower(), symbol.lower())
        
        print(f"📈 Fetching comprehensive market data for {symbol}")
        
        # Get basic price data
        resp = requests.get(f"{COINGECKO_API_BASE}/simple/price",
                            params={
                                "ids": coin_id, 
                                "vs_currencies": "usd", 
                                "include_24hr_change": "true", 
                                "include_24hr_vol": "true",
                                "include_market_cap": "true",
                                "include_last_updated_at": "true"
                            },
                            timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if coin_id not in data:
            raise ValueError(f"Symbol '{symbol}' not found in CoinGecko")
        
        d = data[coin_id]
        
        # Get additional market data if available
        market_data = {
            "symbol": symbol.upper(),
            "coin_id": coin_id,
            "price_usd": d.get("usd"),
            "24h_change_pct": d.get("usd_24h_change"),
            "volume_24h": d.get("usd_24h_vol"),
            "market_cap": d.get("usd_market_cap"),
            "last_updated": datetime.fromtimestamp(d.get("last_updated_at"), tz=timezone.utc).isoformat() if d.get("last_updated_at") else datetime.now(timezone.utc).isoformat(),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"✅ Market data: ${market_data['price_usd']:,.2f} "
              f"({market_data['24h_change_pct']:+.2f}% 24h) "
              f"Vol: ${market_data['volume_24h']:,.0f}")
        
        return market_data
    except Exception as e:
        print(f"❌ Failed to fetch market data for {symbol}: {e}")
        raise RuntimeError(f"Market data fetch failed: {e}")

# Enhanced prompt optimized for Gemini 2.0 Flash
def build_prompt(symbol: str, source_data: dict) -> str:
    """
    Advanced prompt optimized for Gemini 2.0 Flash's capabilities
    """
    prompt = f"""
# CRYPTOCURRENCY PRICE PREDICTION ANALYSIS

## ASSET: {symbol.upper()} ({source_data.get('coin_id', 'N/A')})

## MARKET DATA:
- **Current Price**: ${source_data.get('price_usd', 'N/A'):,.2f}
- **24h Change**: {source_data.get('24h_change_pct', 'N/A'):+.2f}%
- **24h Volume**: ${source_data.get('volume_24h', 'N/A'):,.0f}
- **Market Cap**: ${source_data.get('market_cap', 'N/A'):,.0f}
- **Last Updated**: {source_data.get('last_updated', 'N/A')}

## ANALYSIS TASK:
As a senior cryptocurrency analyst, provide a data-driven prediction for the next 24 hours:

1. **Upside Probability** (0-100): Likelihood of price increase
2. **Downside Probability** (0-100): Likelihood of price decrease  
3. **Expert Commentary** (2-3 sentences): Key factors influencing your prediction

## ANALYSIS FRAMEWORK:
- **Price Momentum**: Recent trend strength and sustainability
- **Volume Analysis**: Trading activity and market participation
- **Market Context**: Overall crypto market conditions
- **Technical Factors**: Support/resistance levels, volatility
- **Sentiment Indicators**: Market psychology and positioning

## RESPONSE REQUIREMENTS:
- Return ONLY valid JSON format
- JSON structure: {{"probability_up": float, "probability_down": float, "commentary": string}}
- Probabilities must sum to 100 (±2% tolerance)
- Commentary: 2-3 concise, professional sentences
- Base analysis strictly on provided data and market principles

## EXAMPLE OUTPUT:
{{
  "probability_up": 67.8,
  "probability_down": 32.2,
  "commentary": "Strong bullish momentum with 12.5% gains on elevated volume suggests continued upside potential. However, overbought conditions near resistance may trigger short-term consolidation. Overall bias remains positive given sustained institutional interest."
}}

## YOUR ANALYSIS:
"""
    return prompt.strip()

def parse_llm_json(text: str):
    """Robust JSON parsing with enhanced error handling"""
    text = text.strip()
    
    # Clean the text
    text = text.replace('```json', '').replace('```', '')
    
    try:
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
            
        json_text = text[start:end]
        parsed = json.loads(json_text)
        
        # Validate required fields
        required_fields = ['probability_up', 'probability_down', 'commentary']
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate probability ranges
        prob_up = float(parsed['probability_up'])
        prob_down = float(parsed['probability_down'])
        
        if not (0 <= prob_up <= 100) or not (0 <= prob_down <= 100):
            raise ValueError("Probabilities must be between 0 and 100")
            
        total = prob_up + prob_down
        if abs(total - 100) > 5:  # Allow 5% tolerance
            raise ValueError(f"Probabilities sum to {total}, should be ~100")
                
        return parsed
        
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"📄 Raw response: {text}")
        raise ValueError(f"LLM response parsing failed: {e}")

def call_gemini(prompt: str, model: str = None) -> dict:
    """Call Gemini with model fallback system"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    
    # Determine which model to use
    selected_model = model or GEMINI_MODEL
    available_models = []
    
    try:
        available_models = list_available_models()
        print(f"🔍 Available models: {len(available_models)} found")
        
        # Check if selected model is available
        if selected_model not in available_models:
            print(f"⚠️ Model {selected_model} not available, trying fallbacks...")
            for backup in BACKUP_MODELS:
                if backup in available_models:
                    selected_model = backup
                    print(f"🔄 Using fallback model: {selected_model}")
                    break
            else:
                # Use first available model as last resort
                if available_models:
                    selected_model = available_models[0]
                    print(f"🚨 Using available model: {selected_model}")
                else:
                    raise RuntimeError("No Gemini models available")
        
        print(f"🤖 Calling Gemini: {selected_model}")
        
        # Optimized configuration for financial analysis
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more consistent financial analysis
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
        }

        model_client = genai.GenerativeModel(
            model_name=selected_model,
            generation_config=generation_config
        )
        
        response = model_client.generate_content(prompt)
        
        if not response.candidates:
            raise RuntimeError("No response generated")
            
        text = response.candidates[0].content.parts[0].text
        print("✅ Gemini analysis completed successfully")
        
        return parse_llm_json(text)
        
    except Exception as e:
        print(f"❌ Gemini API error with model {selected_model}: {e}")
        
        # Show available models for debugging
        if available_models:
            print("📋 Available models:")
            for model_name in available_models[:8]:
                print(f"   - {model_name}")
                
        raise RuntimeError(f"Analysis failed: {e}")

def call_llm(prompt: str):
    """LLM provider router"""
    if LLM_PROVIDER == "openai":
        raise NotImplementedError("OpenAI provider not yet implemented")
    elif LLM_PROVIDER == "gemini":
        return call_gemini(prompt)
    else:
        raise NotImplementedError(f"Unknown provider {LLM_PROVIDER}")

# Core prediction function with Gemini 2.0
def generate_prediction_for(symbol: str, db: Session):
    print(f"🎯 Generating prediction for {symbol} using Gemini 2.0")
    
    try:
        # 1. Fetch comprehensive market data
        source_data = fetch_coingecko_market_data(symbol)
        
        # 2. Build advanced prompt
        prompt = build_prompt(symbol, source_data)
        
        # 3. Call Gemini 2.0 Flash
        print("💭 Calling Gemini 2.0 for advanced analysis...")
        llm_out = call_llm(prompt)
        
        # 4. Process results
        prob_up = float(llm_out.get("probability_up", 0.0))
        prob_down = float(llm_out.get("probability_down", 0.0))
        commentary = str(llm_out.get("commentary", "")).strip()
        
        # Normalize probabilities
        total = prob_up + prob_down
        if total <= 0:
            prob_up, prob_down = 50.0, 50.0
            print("⚠️ Probabilities normalized to 50/50")
        else:
            # Normalize to sum to 100
            scaling_factor = 100.0 / total
            prob_up *= scaling_factor
            prob_down *= scaling_factor

        timestamp = datetime.now(timezone.utc)
        model_meta = {
            "provider": LLM_PROVIDER,
            "model": GEMINI_MODEL,
            "prompt_version": PROMPT_VERSION,
            "timestamp": timestamp.isoformat(),
            "confidence": "high" if abs(prob_up - prob_down) > 20 else "medium"
        }

        # 5. Save to database
        rec = Prediction(
            symbol=symbol,
            timestamp=timestamp,
            probability_up=round(prob_up, 2),
            probability_down=round(prob_down, 2),
            commentary=commentary,
            source_data=source_data,
            model_meta=model_meta
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        
        print(f"✅ Prediction saved: {symbol} - 🔺 UP: {prob_up:.1f}% | 🔻 DOWN: {prob_down:.1f}%")
        
        return {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "probability_up": round(prob_up, 2),
            "probability_down": round(prob_down, 2),
            "commentary": commentary,
            "source_data": source_data,
            "model_meta": model_meta
        }
        
    except Exception as e:
        print(f"❌ Prediction failed for {symbol}: {e}")
        raise

# FastAPI endpoints
@app.post("/predict", response_model=PredictionResponse)
def predict_now(req: PredictRequest, db: Session = Depends(get_db)):
    """Generate a new prediction"""
    try:
        out = generate_prediction_for(req.symbol, db)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest/{symbol}", response_model=PredictionResponse)
def latest(symbol: str, db: Session = Depends(get_db)):
    """Get latest prediction for a symbol"""
    rec = db.query(Prediction).filter(Prediction.symbol == symbol).order_by(Prediction.timestamp.desc()).first()
    if not rec:
        raise HTTPException(status_code=404, detail="No prediction found")
    return {
        "symbol": rec.symbol,
        "timestamp": rec.timestamp.isoformat(),
        "probability_up": rec.probability_up,
        "probability_down": rec.probability_down,
        "commentary": rec.commentary,
        "source_data": rec.source_data,
        "model_meta": rec.model_meta
    }

@app.get("/history/{symbol}", response_model=List[PredictionResponse])
def history(symbol: str, limit: int = 20, db: Session = Depends(get_db)):
    """Get prediction history for a symbol"""
    recs = db.query(Prediction).filter(Prediction.symbol == symbol).order_by(Prediction.timestamp.desc()).limit(limit).all()
    return [{
        "symbol": r.symbol,
        "timestamp": r.timestamp.isoformat(),
        "probability_up": r.probability_up,
        "probability_down": r.probability_down,
        "commentary": r.commentary,
        "source_data": r.source_data,
        "model_meta": r.model_meta
    } for r in recs]

@app.get("/")
def root():
    return {
        "message": "AI Prediction Service with Gemini 2.0 Flash", 
        "status": "healthy",
        "model": GEMINI_MODEL,
        "version": "3.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "latest": "GET /latest/{symbol}",
            "history": "GET /history/{symbol}",
            "health": "GET /health",
            "models": "GET /models"
        }
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
        
        # Test Gemini
        gemini_status = "configured" if GEMINI_API_KEY else "not configured"
        model_status = "available" if GEMINI_API_KEY and list_available_models() else "unknown"
        
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        gemini_status = "unknown"
        model_status = "unknown"
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "llm_provider": LLM_PROVIDER,
        "gemini_model": GEMINI_MODEL,
        "gemini_status": gemini_status,
        "model_status": model_status
    }

@app.get("/models")
def list_models():
    """List available Gemini models"""
    models = list_available_models()
    return {
        "available_models": models,
        "current_model": GEMINI_MODEL,
        "backup_models": BACKUP_MODELS,
        "total_available": len(models)
    }

# Scheduler setup
def start_scheduler():
    """Start scheduler only if not on Vercel"""
    if os.getenv("VERCEL") != "1":
        symbols = os.getenv("PREDICT_SYMBOLS", "bitcoin,ethereum,solana").split(",")
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            scheduled_job, 
            "interval", 
            hours=SCHEDULE_HOURS, 
            next_run_time=datetime.now(timezone.utc),
            args=[symbols]
        )
        scheduler.start()
        print(f"⏰ Scheduler started: predictions every {SCHEDULE_HOURS} hours for {symbols}")

def scheduled_job(symbols: List[str]):
    """Scheduled prediction job"""
    print(f"⏰ [{datetime.now(timezone.utc).isoformat()}] Running scheduled predictions")
    db = SessionLocal()
    try:
        for symbol in symbols:
            try:
                out = generate_prediction_for(symbol, db)
                trend = "🔺 BULLISH" if out['probability_up'] > 60 else "🔻 BEARISH" if out['probability_up'] < 40 else "➡️ NEUTRAL"
                print(f"✅ {symbol}: {trend} (UP: {out['probability_up']:.1f}%)")
            except Exception as e:
                print(f"❌ {symbol}: {e}")
    finally:
        db.close()

# Application startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting AI Prediction Service with Gemini 2.0 Flash")
    print(f"🔧 Configuration: {GEMINI_MODEL}")
    
    # Model discovery
    if GEMINI_API_KEY:
        try:
            models = list_available_models()
            gemini_2_models = [m for m in models if 'gemini-2' in m]
            if gemini_2_models:
                print(f"🎯 Gemini 2.0 models available: {gemini_2_models}")
            else:
                print("ℹ️ Using Gemini 1.5 models")
        except Exception as e:
            print(f"⚠️ Model discovery: {e}")
    
    start_scheduler()

# For local development
if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting server on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)