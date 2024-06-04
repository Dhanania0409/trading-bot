# encoding: utf-8
#OHLC Data: Open High Low Close
#Intra-day trading

class Trader:
    def __init__(self):
        lg.info('Trader initialised')
        
#LOOP until timeout (eg: 2 hours)
#POINT 1: INITIAL CHECK
#check the position: ask the API if the position is open
    #IN: asset (string)
    #OUT: true/false

#check if tradable: ask the API is asset is tradable
    #IN: asset (string)
    #OUT: true/false

#GENERAL TRADE
#load 30 mins canfle
    #IN: asset (string), demad  the API for 30 min candles, time range, candle size
    #OUT: 30 min candles, (OHLC for every candle)

#perform general trade anamlysis: detect intersting trend (UP/DOWN/NP TREND)
    #IN: 30 min candle data (close data)
    #OUT: Trend (string)
    #If  no trend defined, go back to beginning (POINT 1)

    #LOOP until timeout (eg: 30 minutes) or go back to beginning
    #POINT 2
    #STEP 1: load 5 min candeles 
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
        #IN: 5 min candles data(OHLC data), output of the gt analysis(up/ down/ no trend)
        #OUT: True/ False (confirm)
        #If failed, go back to POINT 2

#SUBMIT ORDER
#submit order (limit order): Interact with API
    #IN: Number of shares, asset (String), desired price
    #OUT: True/ False (confirmed), position ID
    #if false, abort go back to point 1 

#check position: see if tbe position exists
    #IN: position iD
    #OUT: True/ False (confirmed)
    #if false, abort go back to point 1 

#LOOP until timeout reached (eg: 8 hours)
#ENTER POSTTION MODE: check the positions in parallel
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

    #STEP 2: check if stochastic curves are crossing
        #IN: OHLC data ( 5 min candles)
        #OUT: True/ False

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
 