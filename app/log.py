from loguru import logger 

def print_log(level_log, text):
    logger.remove() 

    logger.add(
        sink=lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{message}</cyan>"
    )
    return logger.log(level_log.upper(), text)
