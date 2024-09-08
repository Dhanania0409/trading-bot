import logging as lg
import os
from datetime import datetime

def initialise_logger():
    # Creating a folder for the logs
    logs_path = './logs'
    if not os.path.exists(logs_path):
        try:
            os.makedirs(logs_path)
        except OSError as e:
            print(f"Creation of the directory {logs_path} failed due to: {e}")
        else:
            print(f"Successfully created log directory: {logs_path}")

    # Renaming each log depending on the time
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f'{date}.log'
    currentlog_path = os.path.join(logs_path, log_name)

    # Basic logger configuration
    lg.basicConfig(
        filename=currentlog_path,
        format='%(asctime)s - %(levelname)s: %(message)s',
        level=lg.DEBUG
    )
    
    # Adding a StreamHandler to log to the console
    console_handler = lg.StreamHandler()
    console_handler.setLevel(lg.DEBUG)  # You can adjust the level for console logging
    console_handler.setFormatter(lg.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    
    # Add the console handler to the root logger
    lg.getLogger().addHandler(console_handler)

    # Test logging output
    lg.info('Logger initialized successfully.')
    print(f"Log files are being created and stored at: {currentlog_path}")
