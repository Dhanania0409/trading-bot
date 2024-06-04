#encoding: utf-8

#import needed libraries
from traderLib import *
from logger import *
import sys

# initialise logger
initialise_logger()

#check trading account(blocked, total amount)
def check_account_ok():
    try:
        #get account info
    except Exception as e:
        lg.error("Could not get account info")
        lg.info(str(e))
        sys.exit()

#close current orders
def clean_open_orders():
    #open_orders = list of open orders
    lg.info("List of open orders")
    lg.info(str(open_orders))

    for i in open_orders:
        #close order
        lg.info('Order is closed' % str(i.id))

    lg.info('Closing orders complete')

#define asset
def get_ticker():
    #enter ticker 
    ticker = input('Write the ticker you want to operate with: ')
    #OUT: string
    return ticker

#execute trading bot
def main():
    #initialise the logger
    initialise_logger()
    check_account_ok()
    clean_open_orders()
    ticker = input('Write the ticker you want to operate with: ')

    trader = Trader
    #run trading bot:
        #IN: String (ticker)
        #OUT: boolean (True=success) 

if __name__ == 'main':
    main()