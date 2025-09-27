## 🚀 AI Cryptocurrency Prediction Service with Gemini 2.0 Flash

This service provides real-time, LLM-driven cryptocurrency price predictions via a **FastAPI** backend. It is designed to be consumed by a frontend application to display current predictions and historical analysis for popular cryptocurrencies.

### Core Technology Overview

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python) | High-performance API server. |
| **AI Model** | Google **Gemini 2.0 Flash** | Generates the price prediction and commentary based on market data. |
| **Data Source** | CoinGecko API | Fetches comprehensive, real-time market data (price, volume, change). |
| **Database** | SQLAlchemy (ORM) | Persistence layer for storing prediction history and metadata (defaulting to SQLite). |

---

### 🧠 Prediction Process Explained

The service acts as a specialized AI financial analyst. For every prediction, it follows these steps:

1.  **Data Acquisition:** Fetches current market metrics (Price, 24h Volume, 24h Change, Market Cap) for the target crypto asset (e.g., `btc`, `eth`) from CoinGecko.
2.  **Prompt Construction:** The fetched data is inserted into an **advanced, structured prompt** optimized for Gemini 2.0 Flash's analytical capabilities.
3.  **LLM Analysis:** The Gemini model processes the data and the prompt's analysis framework to generate a prediction (probability up/down) and a concise expert commentary.
4.  **Normalization & Storage:** The model's probabilities are validated and **normalized to sum to $100\%$**. The complete result is then saved to the database, ensuring data integrity and historical tracking.

The prediction is a **24-hour directional forecast**.

---

## ⚙️ API Endpoints for Frontend Developers

All endpoints return a standard JSON format.

### 1. Generate a New Prediction (On-Demand)

This endpoint forces the full prediction workflow, including data fetch, LLM call, and database write.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Triggers a new prediction for the specified symbol immediately. |

#### **Request Body (`POST /predict`)**

| Field | Type | Description |
| :--- | :--- | :--- |
| `symbol` | `string` | The cryptocurrency symbol (e.g., **`btc`**, **`eth`**, **`sol`**). |

```json
{
  "symbol": "btc"
}
```

### 2. Retrieve Latest Prediction

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/latest/{symbol}` | Triggers the last prediction made for the specified symbol immediately. |

```
{
  "symbol": "btc",
  "timestamp": "2025-09-27T14:25:38.348272+00:00",
  "probability_up": 58,
  "probability_down": 42,
  "commentary": "Bitcoin shows a slight positive 24h change with substantial volume, indicating continued interest. The high market cap suggests stability, but the upside is tempered by potential profit-taking at this price level. Expect moderate volatility with a slight upward bias.",
  "source_data": {
    "symbol": "BTC",
    "coin_id": "bitcoin",
    "price_usd": 109363,
    "24h_change_pct": 0.21008529749368454,
    "volume_24h": 33864925655.1949,
    "market_cap": 2179208938875.3438,
    "last_updated": "2025-09-27T14:25:12+00:00",
    "analysis_timestamp": "2025-09-27T14:25:32.779565+00:00"
  },
  "model_meta": {
    "provider": "gemini",
    "model": "gemini-2.0-flash-exp",
    "prompt_version": 2,
    "timestamp": "2025-09-27T14:25:38.348272+00:00",
    "confidence": "medium"
  }
}
```