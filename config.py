
import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'pawcare_ai_secret_key_2024'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model configuration
    MODEL_PATH = 'models/pawcare_model.h5'
    IMAGE_SIZE = (224, 224)
    
    # Database configuration (for future use)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///pawcare.db'
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/pawcare.log'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
