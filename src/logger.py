
import logging

class logger:
    logger = None
    def create_logger(self,log_level):

        numeric_log_level = getattr(logging, log_level, None)

        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)

        logging.basicConfig(filename='myapp.log', level=numeric_log_level, format='%(asctime)s %(name)-12s %(lineno)d %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logger.logger = logging.getLogger(__name__)
        logger.logger.info('Initalized logs')

    def __init__(self, log_level):
        if (logger.logger == None):
            self.create_logger(log_level)

    def get_logger():
        logger.logger = logging.getLogger(__name__)
        return logger.logger
