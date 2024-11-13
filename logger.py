import logging as lg
import os
from datetime import datetime

def initialise_logger():
    logs_path = './logs'
    if not os.path.exists(logs_path):
        try:
            os.makedirs(logs_path)
        except OSError as e:
            print(f"Creation of the directory {logs_path} failed due to: {e}")
        else:
            print(f"Successfully created log directory: {logs_path}")
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f'{date}.log'
    currentlog_path = os.path.join(logs_path, log_name)
    lg.basicConfig(
        filename=currentlog_path,
        format='%(asctime)s - %(levelname)s: %(message)s',
        level=lg.DEBUG
    )
    console_handler = lg.StreamHandler()
    console_handler.setLevel(lg.DEBUG)  
    console_handler.setFormatter(lg.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    lg.getLogger().addHandler(console_handler)
    lg.info('Logger initialized successfully.')
    print(f"Log files are being created and stored at: {currentlog_path}")