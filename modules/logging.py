import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)
# Create a file handler
fh = logging.FileHandler('logs/app.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

logger.addHandler(fh)

# logger.info("This is an info level log message.")