#OHLC Data: Open High Low Close

#define asset
#OUT: string

#INTITIAL CHECK
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

#LOOP
#STEP 1: load 5 min candeles 
    #IN: asset (string), demad  the API for 5 min candles, time range, candle size
    #OUT: 30 min candles, (OHLC for every candle)

#STEP 2: perform instant trend analysis: confirk the trend detected by the GT analysis
    #IN: 5 min candles data(close data), output of the gt analysis(up/ down/ no trend)
    #OUT: True/ False (confirm)

# STEP 3:perform RSI analysis
    #IN: 5 min candles data(CLose data), output of the gt analysis(up/ down/ no trend)
    #OUT: True/ False (confirm)

#STEP 4: perform stockhastic analysis
    #IN: 5 min candles data(OHLC data), output of the gt analysis(up/ down/ no trend)
    #OUT: True/ False (confirm)

#SUBMIT ORDER
#submit order
#check position

#ENTER POSTTION MODE
#check take profit
#check stop loss
#check stock crossing

#GET OUT
#check order
#check position

#wait 15 min
#back to beginning
 