"""
Unified error handling for media generation.
Provides common exception types and retry decorators for media generation operations.
"""

import logging
import functools
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_log, after_log
import requests
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaGenerationError(Exception):
    """Base exception class for media generation errors."""
    pass

class VideoGenerationError(MediaGenerationError):
    """Exception for video generation errors."""
    pass

class ImageGenerationError(MediaGenerationError):
    """Exception for image generation errors."""
    pass

class AudioGenerationError(MediaGenerationError):
    """Exception for audio generation errors."""
    pass

class TTSError(MediaGenerationError):
    """Raised when TTS generation fails"""
    pass

class SceneBuilderError(Exception):
    """Base exception for scene builder errors"""
    pass

class CacheError(Exception):
    """Exception for cache operations"""
    pass

def create_retry_decorator(max_attempts=3, min_wait=4, max_wait=10, exception_types=(MediaGenerationError,)):
    """Create a retry decorator with the specified parameters.
    
    Args:
        max_attempts: Maximum number of attempts
        min_wait: Minimum wait time between attempts (seconds)
        max_wait: Maximum wait time between attempts (seconds)
        exception_types: Exception types to retry on
        
    Returns:
        A retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_types),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )

# Pre-configured decorators for common use cases
retry_media_generation = create_retry_decorator(
    exception_types=(MediaGenerationError,)
)

retry_scene_building = create_retry_decorator(
    exception_types=(MediaGenerationError, TTSError)
)

retry_api_call = create_retry_decorator(
    exception_types=(Exception,),
    max_attempts=5,
    min_wait=2,
    max_wait=30
)

retry_download = create_retry_decorator(
    exception_types=(requests.RequestException, IOError),
    max_attempts=4,
    min_wait=2,
    max_wait=15
)

retry_cache_operation = create_retry_decorator(
    exception_types=(CacheError, IOError),
    max_attempts=2,
    min_wait=1,
    max_wait=5
)

# Error logging utility
def log_error(func):
    """Decorator to log errors before raising them."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper 