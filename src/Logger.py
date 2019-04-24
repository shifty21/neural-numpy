import yaml
import logging
import logging.config
import os

class Logger:

    def __init__(self):
        if (os.path.exists("config.yaml")):
            with open ("config.yaml","r") as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
        else:
            logging.basicConfig(logging.INFO)

    def get_logger(name):
        return logging.getLogger(name)
