# logger.py
import logging

LOGGING_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Set up logging configuration
logger = logging.getLogger('KG-logger')  # Create a logger object
logger.setLevel(logging.INFO) 

file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(file_handler)