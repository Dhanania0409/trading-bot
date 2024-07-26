# encoding: utf-8
#OHLC Data: Open High Low Close
#Intra-day trading

import sys
import logger as lg

class Trader:
    def __init__(self, ticker):
        lg.info('Trader initialised with ticker %s' % ticker)
    
    def is_tradable(self, ticker):
    #check if tradable: ask the API is asset is tradable
            #IN: asset (string)
            #OUT: true/false
        try:
            #asset = get asset from alpaca wrapper
            if not asset.tradable:
                lg.info('The asset %s is not tradable' % ticker)
                return False
            else:
                lg.info('The asset %s is tradable' % ticker)
                return True
        except:
            lg.error('The asset %s is not answering well' % ticker)
            return False
        
    def set_stoploss(self, entryPrice, direction):
    # set stoploss: takes price as input and sets the stop loss depending on the direction
        #IN: buying price, direction (long/ short)
        #OUT: stoploss
        stopLossMargin = 0.05 #percentage
        try:
            if direction == 'Long':
                stopLoss = entryPrice - entryPrice*stopLossMargin
                return stopLoss
            elif direction == 'Short':
                stopLoss = entryPrice + entryPrice*stopLossMargin
                return stopLoss
            else:
                raise ValueError
        
        except Exception as e:
            lg.error('The direction value is not understood: %s' % str(direction))
            sys.exit()

        return stopLoss

    # set take profit: takes price as input and sets the take profit
        #IN: buying price
        #OUT: take profit

    # load historical stock data:
        #IN: ticker, interval, entries limit
        #OUT: array with the stock data (OHLC)

    #Get open positions
        #IN: ticker
        #OUT: boolean (True = already open)
    
    #submit orders: gets our order to the API (retry)
        #IN: Order data
        #OUT: boolean (True = order went through)
    
    #Cancel order:
        #IN: Order ID
        #OUT: boolean (True = order cancelled)

    #check positions: CHECK If possition is open
        #IN: ticker
        #OUT: boolean (True = order is there)

    #get general trend: detect intersting trend (UP/DOWN/NO TREND)
        #IN: 30 min candle data (close data)
        #OUT: Trend (string)
        #If  no trend defined, go back to beginning (POINT 1)

    #get stochastic analysis:
        #IN: 5 min candles data(OHLC data), output of the gt analysis(up/ down/ no trend)
        #OUT: True/ False (confirm)
        #If failed, go back to POINT 2

    #enter position mode: check the conditions in parallel once inside the position 
    #IF check take profit -> close position
            #IN: current gains (earning $)
            #OUT: True/ False

        #ELIF check stop loss, if true -> close position
            #IN: current gains (loosing $)
            #OUT: True/ False

        #ELIF check stock crossing, if true -> close position
            #STEP 1: pull 5 mknites OHLC data
                #IN: asset
                #OUT: OHLC data ( 5 min candles)

            #STEP 2: check if stochastic curves are crossing
                #IN: OHLC data ( 5 min candles)
                #OUT: True/ False

     def run():

        #LOOP until timeout (eg: 2 hours)
        #POINT 1: INITIAL CHECK
        #check the position: ask the API if the position is open
            #IN: asset (string)
            #OUT: true/false

        #TREND ANALYSIS:
        #load historical candle
        #load 30 mins cand  le
            #IN: asset (string), demad  the API for 30 min candles, time range, candle size
            #OUT: 30 min candles, (OHLC for every candle)

        #get general trend analysis: detect intersting trend (UP/DOWN/NP TREND)

            #LOOP until timeout (eg: 30 minutes) or go back to beginning
            #POINT 2
            #STEP 1: load historical data - load 5 min candeles 
                #IN: asset (string), demad  the API for 5 min candles, time range, candle size
                #OUT: 30 min candles, (OHLC for every candle)
                #If failed, go back to POINT 2

            #STEP 2: perform instant trend analysis: confirk the trend detected by the GT analysis
                #IN: 5 min candles data(close data), output of the gt analysis(up/ down/ no trend)
                #OUT: True/ False (confirm)
                #If failed, go back to POINT 2

            # STEP 3:perform RSI analysis
                #IN: 5 min candles data(CLose data), output of the gt analysis(up/ down/ no trend)
                #OUT: True/ False (confirm)
                #If failed, go back to POINT 2

            #STEP 4: perform stockhastic analysis
                
        #SUBMIT ORDER
        #submit order (limit order): Interact with API
            #if false, abort go back to point 1    

        #check position: see if tbe position exists
            #if false, abort go back to point 1 

        #LOOP until timeout reached (eg: 8 hours)
        #ENTER POSTTION MODE: check the positions in parallel
        

        #GET OUT
        #submit order(market): Interact with API
            #IN: Number of shares, asset (String), position ID
            #OUT: True/ False
            #if false, retry until it works 

        #check position: see if tbe position exists
            #IN: position iD
            #OUT: True/ False (confirmed)
            #if false, abort/ go back to SUBMIT ORDER

        #wait 15 min

#end

