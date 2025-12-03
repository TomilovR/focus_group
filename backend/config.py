import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "100/minute")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Email Validation
MAX_SUBJECT_LENGTH = 200
MAX_BODY_LENGTH = 10000
MAX_CTA_LENGTH = 500

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

def setup_logging():
    """Setup application logging with rotation"""
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("email_ai_predictor")
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=LOG_MAX_BYTES, 
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()
