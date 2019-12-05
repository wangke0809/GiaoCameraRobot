import logging, logging.config
import json, config

class Logger(object):
    
    @staticmethod
    def getLogger(name):
        conf = json.loads(config.LoggerJsonConfig)
        logging.config.dictConfig(conf)     
        return logging.getLogger(name)

if __name__ == '__main__':
    l = Logger.getLogger("test")
    l.info("aaa")
    l = Logger.getLogger("testa")
    l.info("aaa")
    l = Logger.getLogger("test.a")
    l.info("aaa")
    l.debug("ddd")
    l.warning("w")
    l.error('e')