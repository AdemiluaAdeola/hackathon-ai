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
