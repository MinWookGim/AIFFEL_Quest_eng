import logging, sys

def setup_logger(name="ml_api", level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger
