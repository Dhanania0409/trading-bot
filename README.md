# Trading Bot with Sentiment Analysis and Technical Indicators

This repository contains a **Python-based trading bot** that suggests whether to buy a stock based on a combination of **sentiment analysis** from news articles and several key **technical indicators**. The bot integrates data from the **Alpaca API** and sentiment from the **Finnhub News API**, along with several calculated metrics like moving averages, RSI, and trading volume spikes to determine if a stock is worth buying.

## Features

- **Sentiment Analysis**: Fetches and analyzes the latest news articles related to a stock.
- **Technical Analysis**: Evaluates stock price data using indicators like moving averages, RSI, and trading volume.
- **Weightage-Based Decision Making**: Each indicator is weighted to contribute towards a final buy/sell decision, making the logic flexible and realistic.
- **Alpaca API Integration**: Fetches real-time stock data (OHLC) for analysis.
- **Finnhub News API**: Retrieves recent news articles for sentiment scoring.

---

## Key Indicators and Weightages

1. **Sentiment Score (50%)**
   - **Definition**: Sentiment analysis evaluates the tone of news articles about a stock. A sentiment score is calculated based on positive, neutral, and negative sentiments.
   - **How it works**: 
     - The bot fetches the latest 5 news articles about the stock.
     - Each article is given a score (+1 for positive, 0 for neutral, and -1 for negative).
     - The final sentiment score is the total sum of these articles.
   - **Weightage**: If the sentiment score is 3/5 or higher, it contributes 50% to the overall decision.

2. **Moving Averages (30%)**
   - **Definition**: Moving averages smooth out price data to help identify trends over different time periods.
   - **How it works**: The bot calculates:
     - **Short-Term (20-day) Moving Average**: Focuses on recent price trends.
     - **Mid-Term (50-day) Moving Average**: Provides insight into medium-term market direction.
     - **Long-Term (100-day) Moving Average**: Tracks long-term momentum.
   - **Weightage**:
     - Last close price above the **20-day MA** adds 10%.
     - Last close price above the **50-day MA** adds 20%.
     - Last close price above the **100-day MA** adds another 20%.

3. **RSI (Relative Strength Index) (20%)**
   - **Definition**: RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate whether a stock is overbought or oversold.
   - **How it works**: 
     - An RSI value between **50-75** is considered favorable and contributes to a buy decision.
     - If RSI is less than **50**, the stock is considered oversold, which can also be a strong buy signal.
   - **Weightage**:
     - RSI between **50-75** adds 20%.
     - RSI below **50** adds 10%.

4. **Volume Spike (10%)**
   - **Definition**: A volume spike indicates an unusual increase in the number of shares traded, signaling heightened market interest.
   - **How it works**: 
     - The bot compares the current trading volume with the 30-day average.
     - If the current volume is **15% higher than the 30-day average**, it is considered a volume spike.
   - **Weightage**: A 15% higher volume spike adds 10% to the overall decision.

---

## Buy Signal Criteria

A stock receives a **BUY** signal when the combined weightage of all indicators is **greater than or equal to 60%**. Each indicator contributes a specific percentage to the overall score, and the final score determines the buy recommendation.

### Example of a Buy Signal:

- **Sentiment Score**: 4/5 → 50% (positive sentiment)
- **Price vs. Moving Averages**:
  - Last close price > 50-day moving average → 20%
- **RSI**: RSI = 65 (within favorable range) → 20%
- **Volume Spike**: True → 10%

Final score = 50% (Sentiment) + 20% (Moving Average) + 20% (RSI) + 10% (Volume Spike) = **100%**, resulting in a **BUY** signal.

---

## How to Use the Trading Bot

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API Keys  
Create a config.json file in the root of the project and add your Alpaca and Finnhub API keys:
```json
{
  "api_key": "your_alpaca_api_key",
  "api_secret": "your_alpaca_api_secret",
  "finnhub_api_key": "your_finnhub_api_key",
  "stop_loss_margin": 0.05,
  "take_profit_margin": 0.10,
  "check_interval_minutes": 5,
  "timeout_hours": 2
}
```

### 4. Run the Bot  
```bash
python bot.py
```  
## Future Enhancements

- Adding stop-loss and take-profit mechanisms.
- Expanding support for international stock markets using additional APIs.
- Improving sentiment analysis by incorporating additional sources (e.g., Twitter, Reddit).


