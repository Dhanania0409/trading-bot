# Import necessary libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

# Download the VADER lexicon if it hasn't been downloaded already
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Add custom words to VADER's lexicon to improve analysis of financial news
custom_lexicon = {
    'breaking all records': 3.0,
    'on the rise': 2.0,
    'record earnings': 2.5,
    'significant losses': -2.5,
    'disappointing': -2.0
}
# Update VADER with the custom lexicon
sid.lexicon.update(custom_lexicon)

# Finnhub API key (replace 'your_finnhub_api_key' with your actual API key)
API_KEY = 'crep6r9r01qnd5d02ne0crep6r9r01qnd5d02neg'

def fetch_news_articles(ticker):
    """
    Fetches the last 5 news articles about the given company ticker from Finnhub API.
    Returns a list of news headlines.
    """
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-09-01&to=2024-09-08&token={API_KEY}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        
        # Get the last 5 news articles (or less if fewer are available)
        latest_news = [article['headline'] for article in news_data[:5]]
        return latest_news
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return []
    except Exception as err:
        print(f"Other error occurred: {err}")
        return []

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text and returns a score:
    +1 for positive sentiment, -1 for negative sentiment, and 0 for neutral.
    Also prints detailed sentiment scores for each article.
    """
    # Get the sentiment scores for the text
    sentiment_scores = sid.polarity_scores(text)
    
    # Print detailed sentiment scores for debugging
    print(f"Sentiment scores for '{text}': {sentiment_scores}")

    # Determine the sentiment based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        return 1  # Positive sentiment
    elif sentiment_scores['compound'] <= -0.05:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Function to analyze a list of news articles and return the overall sentiment score
def analyze_news_sentiment(news_articles):
    """
    Analyzes the sentiment of a list of news articles and returns the overall sentiment score.
    For each article, +1 is added for positive sentiment, -1 for negative, and 0 for neutral.
    """
    overall_score = 0
    for article in news_articles:
        sentiment_score = analyze_sentiment(article)
        overall_score += sentiment_score
    
    return overall_score
