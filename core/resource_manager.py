"""
Resource management utilities for memory optimization.
Provides centralized resource tracking and cleanup functionality.
"""

import logging
from pathlib import Path
from typing import List, Any
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip

# Configure logging
logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages resources like clips and temporary files to ensure proper cleanup.
    Uses context manager pattern for automatic resource management.
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self.active_clips: List[Any] = []  # Stores active video/audio clips
        self.temp_files: List[Path] = []   # Stores temporary file paths
        self.registered_objects: List[Any] = []  # Stores other objects that need cleanup
        
    def register_clip(self, clip) -> Any:
        """
        Register a clip for automatic cleanup.
        
        Args:
            clip: Any MoviePy clip type (VideoFileClip, AudioFileClip, CompositeVideoClip, etc.)
            
        Returns:
            The registered clip
        """
        # Check if the object has the critical moviepy clip attributes/methods
        if hasattr(clip, 'close') and hasattr(clip, 'duration'):
            self.active_clips.append(clip)
            logger.debug(f"Registered clip: {type(clip).__name__}")
        else:
            logger.warning(f"Attempted to register unknown clip type: {type(clip)}")
        return clip
        
    def register_temp_file(self, path: str | Path) -> Path:
        """
        Register a temporary file for cleanup.
        
        Args:
            path: Path to the temporary file
            
        Returns:
            Path object for the registered file
        """
        path = Path(path)
        self.temp_files.append(path)
        logger.debug(f"Registered temporary file: {path}")
        return path
        
    def register_object(self, obj: Any) -> Any:
        """
        Register any object that needs cleanup.
        
        Args:
            obj: Object with a cleanup/close method
            
        Returns:
            The registered object
        """
        self.registered_objects.append(obj)
        logger.debug(f"Registered object: {type(obj).__name__}")
        return obj
        
    def cleanup(self):
        """Clean up all registered resources."""
        # Clean up clips
        for clip in self.active_clips:
            try:
                if hasattr(clip, 'close'):
                    clip.close()
                logger.debug(f"Closed clip: {type(clip).__name__}")
            except Exception as e:
                logger.warning(f"Failed to close clip {type(clip).__name__}: {e}")
        self.active_clips.clear()
        
        # Clean up temporary files
        for path in self.temp_files:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Deleted temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {path}: {e}")
        self.temp_files.clear()
        
        # Clean up registered objects
        for obj in self.registered_objects:
            try:
                if hasattr(obj, 'cleanup'):
                    obj.cleanup()
                elif hasattr(obj, 'close'):
                    obj.close()
                logger.debug(f"Cleaned up object: {type(obj).__name__}")
            except Exception as e:
                logger.warning(f"Failed to clean up object {type(obj).__name__}: {e}")
        self.registered_objects.clear()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
        if exc_type is not None:
            logger.error(f"Error during resource management: {exc_val}")
            return False  # Re-raise the exception
        return True 