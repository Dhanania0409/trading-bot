import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import logging as lg
from datetime import datetime, timedelta

lg.basicConfig(level=lg.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
custom_lexicon = {
    'breaking all records': 3.0,
    'on the rise': 2.0,
    'record earnings': 2.5,
    'significant losses': -2.5,
    'disappointing': -2.0
}
sid.lexicon.update(custom_lexicon)

# Finnhub API key
API_KEY = 'crep6r9r01qnd5d02ne0crep6r9r01qnd5d02neg'
previous_sentiment_scores = []
def fetch_news_articles(ticker, start_date=None, end_date=None):
    base_url = 'https://finnhub.io/api/v1/company-news'
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    params = {
        'symbol': ticker,
        'from': start_date,
        'to': end_date,
        'token': API_KEY
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        news_articles = response.json()

        if not news_articles:
            lg.info(f"No news found for {ticker} from {start_date} to {end_date}.")
            return []

        lg.info(f"Fetched {len(news_articles)} news articles for {ticker}.")
        return [article['headline'] for article in news_articles]

    except requests.exceptions.RequestException as e:
        lg.error(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    lg.info(f"Sentiment scores for '{text}': {sentiment_scores}")
    previous_sentiment_scores.append(sentiment_scores['compound'])
    if len(previous_sentiment_scores) > 10: 
        previous_sentiment_scores.pop(0)
    avg_previous_sentiment = sum(previous_sentiment_scores) / len(previous_sentiment_scores)
    if sentiment_scores['compound'] >= 0.05:
        sentiment_result = 1  
    elif sentiment_scores['compound'] <= -0.05:
        sentiment_result = -1 
    else:
        sentiment_result = 0 
    lg.info(f"Current sentiment: {sentiment_result}, Average of previous sentiment scores: {avg_previous_sentiment:.2f}")

    return sentiment_result

def analyze_news_sentiment(news_articles):
    overall_score = 0
    for article in news_articles:
        sentiment_score = analyze_sentiment(article)
        overall_score += sentiment_score

    # Return average sentiment score if there are articles; otherwise, return 0
    return overall_score / len(news_articles) if news_articles else 0