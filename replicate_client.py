"""
Handles interactions with the Replicate API for video generation.
"""
import os
import uuid
import requests
import hashlib
import json
from pathlib import Path
import replicate
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
import logging

from media_client import GenerativeMediaClient
from cache_handler import CacheHandler
from error_handling import VideoGenerationError, retry_media_generation
from utils import download_video, ensure_dir_exists

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplicateClient(GenerativeMediaClient):
    """Base client for interacting with Replicate API."""
    
    def __init__(self, api_token: str = None):
        """Initialize the Replicate client.
        
        Args:
            api_token: Optional Replicate API token. If not provided, will look for REPLICATE_API_TOKEN env var.
        """
        load_dotenv()
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable or api_token parameter is required")
            
        # Set the API token for the replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
    def check_auth(self) -> bool:
        """Verify authentication is working.
        
        Returns:
            bool: True if authentication is successful
        """
        try:
            # Simple API call to verify token works
            client = replicate.Client(api_token=self.api_token)
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False


class ReplicateRayClient(ReplicateClient):
    """Client for interacting with Luma Ray model on Replicate."""
    
    # Use luma/ray without version hash to get latest version
    MODEL_ID = "luma/ray"
    DOWNLOAD_DIR = Path("downloads/replicate_segments")
    
    def __init__(self, api_token: str = None, dev_mode: bool = False):
        """Initialize the Replicate Ray client.
        
        Args:
            api_token: Optional Replicate API token. If not provided, will look for REPLICATE_API_TOKEN env var.
            dev_mode: If True, forces cache usage and skips API calls when possible, useful for development/testing.
        """
        super().__init__(api_token)
        
        # Ensure download directory exists
        ensure_dir_exists(self.DOWNLOAD_DIR)
        
        # Initialize cache handler
        self.cache_handler = CacheHandler(
            cache_dir=self.DOWNLOAD_DIR,
            cache_file="cache.json",
            dev_mode=dev_mode
        )
        
        self.dev_mode = dev_mode
        self._used_videos = []  # Track sequence of used videos in dev mode
        
        # Define core parameters for cache key generation
        self.core_cache_params = ["prompt", "aspect_ratio", "loop"]
    
    @retry_media_generation
    def generate_media(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate a video using Replicate's Ray model.
        
        Args:
            prompt: Text prompt describing the desired video
            config: Configuration parameters for Ray model
                - aspect_ratio: "16:9", "9:16", or "1:1" 
                - width: Video width in pixels
                - height: Video height in pixels
                - continue_from_file: Path to previous video to continue from
                - loop: Whether the video should loop seamlessly
                - youtube_optimized: Whether to use YouTube-optimized resolutions
            
        Returns:
            List of paths to generated video files
        """
        # Ensure config is not None
        config = config or {}
        
        # Normalize aspect ratio to allowed values
        aspect_ratio = config.get("aspect_ratio", "16:9")
        
        # Check if we should use YouTube-optimized resolutions
        youtube_optimized = config.get("youtube_optimized", True)
        
        # Map dimensions based on aspect ratio if not explicitly provided
        width = config.get("width")
        height = config.get("height")
        
        if not width or not height:
            if youtube_optimized:
                # Use YouTube-optimized resolutions
                if aspect_ratio == "16:9":
                    width, height = 1920, 1080  # Full HD for landscape YouTube videos
                elif aspect_ratio == "9:16":  # Short format
                    width, height = 1080, 1920  # Full HD for YouTube Shorts
                elif aspect_ratio == "1:1":   # Square format
                    width, height = 1080, 1080  # 1080x1080 for square videos
                else:
                    width, height = 1920, 1080  # Default to 16:9 Full HD
            else:
                # Use original lower resolutions
                if aspect_ratio == "16:9":
                    width, height = 1024, 576
                elif aspect_ratio == "9:16":  # Short format
                    width, height = 576, 1024
                elif aspect_ratio == "1:1":   # Square format
                    width, height = 1024, 1024
                else:
                    width, height = 1024, 576  # Default to 16:9
        
        # Prepare parameters for Ray model
        params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "width": width,
            "height": height,
            "loop": config.get("loop", False),
            # Can include additional params here as needed
        }
        
        # Set the file to continue from if provided
        continue_from = config.get("continue_from_file")
        if continue_from:
            params["continue_from"] = continue_from
        
        # Generate a unique identifier for this request
        request_id = str(uuid.uuid4())
        
        # Dev mode handling - force cache usage
        if self.dev_mode:
            # We use dynamic cache key generation here to handle dev mode
            def generate_video_func():
                # Attempt generation if no cached video found
                return self._call_replicate_api(params, request_id)
                
            # Try to get from cache or generate if needed
            file_paths, from_cache = self.cache_handler.get_or_add_cached_media(
                params=params,
                core_params=self.core_cache_params,
                generator_func=generate_video_func,
                result_key='file_paths'
            )
            
            return file_paths
            
        # Production mode - check cache first
        def generate_video_func():
            return self._call_replicate_api(params, request_id)
            
        # Try to get from cache or generate if needed
        file_paths, from_cache = self.cache_handler.get_or_add_cached_media(
            params=params,
            core_params=self.core_cache_params,
            generator_func=generate_video_func,
            result_key='file_paths'
        )
        
        return file_paths

    def _call_replicate_api(self, params: Dict[str, Any], request_id: str) -> List[str]:
        """Call the Replicate API to generate a video.
        
        Args:
            params: Model parameters
            request_id: Unique ID for this request
            
        Returns:
            List of paths to downloaded video files
        """
        try:
            # Call the Replicate API
            client = replicate.Client(api_token=self.api_token)
            
            # Run the model
            output = client.run(
                self.MODEL_ID,
                input=params
            )
            
            # Process the output
            video_urls = output if isinstance(output, list) else [output]
            
            # Check if we have valid URLs
            if not video_urls or not all(isinstance(url, str) for url in video_urls):
                raise VideoGenerationError(f"Invalid response from Replicate: {output}")
                
            # Download the videos
            video_paths = []
            for i, url in enumerate(video_urls):
                # Use request_id plus index for multiple videos
                file_id = f"{request_id}_{i}" if len(video_urls) > 1 else request_id
                try:
                    video_path = download_video(url, self.DOWNLOAD_DIR, file_id)
                    video_paths.append(video_path)
                except Exception as e:
                    logger.error(f"Failed to download video {i} from {url}: {e}")
                    raise VideoGenerationError(f"Video download failed: {str(e)}")
            
            return [str(path) for path in video_paths]
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise VideoGenerationError(f"Video generation failed: {str(e)}") 