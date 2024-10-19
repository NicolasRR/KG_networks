# logger.py
import logging

# Set up logging configuration
logger = logging.getLogger('KG-logger')  # Create a logger object
logger.setLevel(logging.INFO) 

LOGGING_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}