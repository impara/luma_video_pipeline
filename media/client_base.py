"""
Base class defining the interface for all media generation clients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class MediaClient(ABC):
    """
    Abstract base class for media generation clients.
    All media clients (SDXL, Replicate, Gemini) should implement this interface.
    """
    
    @abstractmethod
    def generate_media(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate media (image or video) based on the provided prompt.
        
        Args:
            prompt: The text prompt for media generation
            **kwargs: Additional parameters specific to the media client
            
        Returns:
            Dictionary containing media generation result with at least:
            - 'url': URL to the generated media
            - 'local_path': Path where the media is saved locally
        """
        pass
    
    @abstractmethod
    def download_media(self, url: str, output_path: str) -> str:
        """
        Download media from URL to local storage.
        
        Args:
            url: URL of the media to download
            output_path: Path where to save the downloaded media
            
        Returns:
            Path to the downloaded media file
        """
        pass
    
    @abstractmethod
    def get_media_dimensions(self, path: str) -> Dict[str, int]:
        """
        Get the dimensions of the media.
        
        Args:
            path: Path to the media file
            
        Returns:
            Dictionary containing width and height of the media
        """
        pass 