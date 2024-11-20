import logging

def initialise_logger():
    """
    Initialize and return a logger for the application.
    """
    logger = logging.getLogger('trading_bot')

    # Check if the logger has handlers already, to avoid adding multiple handlers
    if not logger.hasHandlers():
        # Set the logging level
        logger.setLevel(logging.INFO)
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        
        # Define a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the console handler to the logger
        logger.addHandler(console_handler)

        # Optional: Add a file handler if you want to log to a file
        file_handler = logging.FileHandler('trading_bot.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Test log message
if __name__ == "__main__":
    logger = initialise_logger()
    logger.info("Logger initialized successfully.")
