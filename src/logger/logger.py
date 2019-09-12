import yaml
import logging
import logging.config
import os


class Logger:
    def __init__(self):
        file = os.path.dirname(os.path.abspath(__file__)) + "/config.yaml"
        print(dir)
        if (os.path.exists(file)):
            with open(file, "r") as f:
                print("loading config file")
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
        else:
            logging.basicConfig()

    def get_logger(name):
        return logging.getLogger(name)
