import logging as lg
import os
from datetime import datetime

def initialise_logger():

    # Creating a folder for the logs
    logs_path = './logs'
    try:
        os.mkdir(logs_path)
    except OSError:
        print(f"Creation of the directory {logs_path} failed - it does not have to be bad")
    else:
        print("Successfully created log directory")

    # Renaming each log depending on the time
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = date + '.log'
    currentlog_path = os.path.join(logs_path, log_name)

    lg.basicConfig(filename=currentlog_path, format='%(asctime)s - %(levelname)s: %(message)s', level=lg.DEBUG)

    # Logging levels: DEBUG, INFO, WARNING, ERROR
    lg.info('This is an info message')
    lg.error('This is an error message')

    lg.getLogger().addFilter(lg.StreamHandler())

    print(f"Log files are being created and stored at: {currentlog_path}")
