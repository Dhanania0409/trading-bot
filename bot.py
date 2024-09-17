from traderLib import *
from logger import *

def get_ticker():
    ticker = input('Write the ticker you want to operate with: ')
    return ticker

def main():
    ticker = get_ticker()
    trader = Trader(ticker)
    df = trader.get_historical_data()
    print(f"OHLC data for {ticker} over the last period:\n{df}\n")
    trader.plot_stock_data(df)
    print(f"\nPerforming sentiment analysis for {ticker} over different periods...\n")
    if trader.should_buy(df):
        print(f"BUY signal for {ticker}.")
    else:
        print(f"NO BUY signal for {ticker}.")

if __name__ == "__main__":
    main()
