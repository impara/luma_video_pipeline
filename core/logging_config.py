import logging
import logging.handlers
import os
from pathlib import Path

def configure_logging(debug_mode=False):
    """
    Configure logging for the application.
    
    Args:
        debug_mode: Whether to enable debug logging
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs - DEBUG level
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "luma_pipeline.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Special handler for caption synchronization debugging
    sync_logger = logging.getLogger("captions")
    sync_handler = logging.FileHandler(log_dir / "caption_sync.log", mode="w")
    sync_handler.setLevel(logging.DEBUG)
    sync_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sync_handler.setFormatter(sync_format)
    sync_logger.addHandler(sync_handler)
    
    logging.info("Logging configured successfully")
    if debug_mode:
        logging.info("Debug logging enabled") 