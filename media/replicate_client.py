"""
Replicate client for video generation using Ray model.
"""

import os
import time
import json
import replicate
import logging
import httpx
import requests
from pathlib import Path
import subprocess
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple

from media.client_base import MediaClient
from core.error_handling import retry_api_call, retry_download, MediaGenerationError, VideoGenerationError, retry_media_generation
from core.cache_handler import CacheHandler
from core.utils import ensure_directory_exists, download_file_to_path, download_video
from core.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplicateRayClient(MediaClient):
    """Client for interacting with Luma Ray model on Replicate."""
    
    # Use luma/ray without version hash to get latest version
    MODEL_ID = "luma/ray"
    
    def __init__(self, api_token: str = None, dev_mode: bool = False):
        """Initialize the Replicate Ray client.
        
        Args:
            api_token: Optional Replicate API token. If not provided, will look for REPLICATE_API_TOKEN env var.
            dev_mode: If True, forces cache usage and skips API calls when possible, useful for development/testing.
        """
        # Store the API token
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable or api_token parameter is required")
        
        # Set the API token for the replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
        # Load config for output directories
        self.config = Config()
        
        # Create replicate_segments directly under output
        self.DOWNLOAD_DIR = Path(self.config.base_output_dir) / "replicate_segments"
        
        # Ensure download directory exists
        ensure_directory_exists(self.DOWNLOAD_DIR)
        
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
                # Use 720p resolution for non-optimized mode to ensure YouTube 720p quality
                if aspect_ratio == "16:9":
                    width, height = 1280, 720  # HD Ready (720p)
                elif aspect_ratio == "9:16":  # Short format
                    width, height = 720, 1280  # Vertical HD
                elif aspect_ratio == "1:1":   # Square format
                    width, height = 720, 720  # Square HD
                else:
                    width, height = 1280, 720  # Default to 16:9 HD
        
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
            # Check if we're using a test API key
            if self.api_token.lower() == "test" or "test" in self.api_token.lower():
                logger.info("Test mode: Generating test video file instead of calling Replicate API")
                
                # Create a test video file using ffmpeg if available
                width = params.get("width", 1280)
                height = params.get("height", 720)
                
                # Simulate video generation
                try:
                    # Create a test video path
                    test_video_path = self.DOWNLOAD_DIR / f"ray_test_{request_id}.mp4"
                    
                    # Try to create a test video with ffmpeg if available
                    try:
                        duration = 5  # 5 seconds test video
                        
                        # Create a text label showing the prompt and dimensions
                        label = f"{params.get('prompt', 'Test Video')[:50]}...\n{width}x{height}"
                        
                        # Command to create a test video with text
                        cmd = [
                            "ffmpeg", "-y",  # Overwrite output files
                            "-f", "lavfi",   # Use lavfi input
                            "-i", f"color=c=cornflowerblue:s={width}x{height}:d={duration}",  # Blue background
                            "-vf", f"drawtext=text='{label}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5",  # Add text
                            "-c:v", "libx264",  # H.264 codec
                            "-pix_fmt", "yuv420p",  # Standard pixel format
                            "-r", "24",  # 24 fps
                            str(test_video_path)
                        ]
                        
                        logger.info(f"Creating test video with dimensions {width}x{height}")
                        subprocess.run(cmd, check=True, capture_output=True)
                        logger.info(f"Generated test video at {test_video_path}")
                        
                    except (subprocess.SubprocessError, FileNotFoundError) as e:
                        # Fallback: create an empty file if ffmpeg fails
                        logger.warning(f"Failed to create test video with ffmpeg: {e}")
                        logger.info("Creating empty test video file instead")
                        test_video_path.touch()
                        
                    return [str(test_video_path)]
                    
                except Exception as e:
                    # Final fallback: return a fake path
                    logger.error(f"Error creating test video: {e}")
                    return [str(self.DOWNLOAD_DIR / f"ray_error_{request_id}.mp4")]
            
            # Real API call (if not in test mode)
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

    def download_media(self, url: str, output_path: str) -> str:
        """
        Download media from URL to local storage.
        
        Args:
            url: URL of the media to download
            output_path: Path where to save the downloaded media
            
        Returns:
            Path to the downloaded media file
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Downloaded media to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
            raise VideoGenerationError(f"Failed to download media: {e}")
    
    def get_media_dimensions(self, path: str) -> Dict[str, int]:
        """
        Get the dimensions of the media.
        
        Args:
            path: Path to the media file
            
        Returns:
            Dictionary containing width and height of the media
        """
        try:
            # For videos, we need to use a video library to get dimensions
            import cv2
            video = cv2.VideoCapture(path)
            if not video.isOpened():
                raise VideoGenerationError(f"Could not open video file: {path}")
                
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video.release()
            
            return {"width": width, "height": height}
        except ImportError:
            # Fallback if OpenCV is not available
            logger.warning("OpenCV not available, using default video dimensions")
            return {"width": 1920, "height": 1080}
        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            raise VideoGenerationError(f"Failed to get video dimensions: {e}") 