"""
Advanced Configuration for Pathhole Detection System
"""

import os
from typing import Dict, Any
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

class Config:
    """Base configuration"""
    
    # API Settings
    API_TITLE = "Pathhole Detection API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Production-ready API for detecting potholes in road images"
    
    # Model Settings
    MODEL_PATH = os.getenv("MODEL_PATH", "./best(1).pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
    
    # Image Processing
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1024))
    MIN_IMAGE_SIZE = 64
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    
    # Processing
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda:0, etc
    
    # Server Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # Streamlit Settings
    STREAMLIT_PORT = 8501
    STREAMLIT_THEME = "light"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = BASE_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    
    # Upload Settings
    UPLOAD_DIR = BASE_DIR / "uploads"
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Temporary Files
    TEMP_DIR = BASE_DIR / "temp"
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Cache Settings
    ENABLE_CACHE = True
    CACHE_DIR = BASE_DIR / ".cache"
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Security
    MAX_BATCH_SIZE = 100
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
    ]
    
    # CORS
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # Inference Settings
    INFERENCE_TIMEOUT = 120
    MAX_PREDICTIONS = 1000
    AUGMENT = False
    VISUALIZE = False
    SAVE_TXT = False
    SAVE_CROP = False
    SAVE_CONF = True
    
    # Response Format
    RESPONSE_FORMAT = {
        "include_image": False,
        "include_raw": False,
        "include_metadata": True,
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    API_DEBUG = True
    ENABLE_CACHE = False
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""
    API_DEBUG = False
    ENABLE_CACHE = True
    LOG_LEVEL = "INFO"
    MAX_BATCH_SIZE = 50  # Lower for production


class TestingConfig(Config):
    """Testing configuration"""
    API_DEBUG = True
    TESTING = True
    MAX_IMAGE_SIZE = 512
    ENABLE_CACHE = False
    ALLOWED_EXTENSIONS = {".jpg", ".png"}


def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Export config instance
config = get_config()

# Export all settings as dict for easy access
SETTINGS: Dict[str, Any] = {
    "model_path": config.MODEL_PATH,
    "confidence": config.CONFIDENCE_THRESHOLD,
    "iou": config.IOU_THRESHOLD,
    "max_image_size": config.MAX_IMAGE_SIZE,
    "max_file_size": config.MAX_FILE_SIZE,
    "allowed_extensions": config.ALLOWED_EXTENSIONS,
    "device": config.DEVICE,
    "api_host": config.API_HOST,
    "api_port": config.API_PORT,
    "upload_dir": str(config.UPLOAD_DIR),
    "temp_dir": str(config.TEMP_DIR),
    "cache_enabled": config.ENABLE_CACHE,
    "cache_dir": str(config.CACHE_DIR),
}
