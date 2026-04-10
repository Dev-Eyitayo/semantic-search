import sys
from loguru import logger
from core.config import settings

def setup_logging():
    logger.remove()

    logger.add(
        sys.stdout,
        enqueue=True,
        backtrace=True,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


    logger.add(
        "logs/sheltly_dev.log",
        rotation="10 MB",
        retention="7 days", 
        level="INFO",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger

custom_logger = setup_logging()