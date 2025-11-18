import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# file handler
fh = logging.FileHandler('logs/training.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

logger.addHandler(fh)