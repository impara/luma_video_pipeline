"""
Common utility functions for media generation.
Provides shared functionality for downloading files, path handling, etc.
"""

import os
import uuid
import requests
from pathlib import Path
import logging
from typing import Optional, Union

from core.error_handling import retry_download, log_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry_download
@log_error
def download_file(url: str, output_dir: Path, file_id: Optional[str] = None, extension: str = "") -> Path:
    """Download a file from a URL to the specified directory.
    
    Args:
        url: URL to download from
        output_dir: Directory to save the file to
        file_id: Optional ID for the file, if not provided a UUID will be generated
        extension: File extension to append
        
    Returns:
        Path to the downloaded file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a filename
    if not file_id:
        file_id = str(uuid.uuid4())
    
    # Add extension if provided
    filename = f"{file_id}{extension}"
    output_path = output_dir / filename
    
    # Download the file
    try:
        logger.info(f"Downloading from {url} to {output_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Download complete: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

# Add alias for consistency with new naming convention
download_file_to_path = download_file

def download_image(url: str, output_dir: Path, file_id: Optional[str] = None) -> Path:
    """Specialized function for downloading images.
    
    Args:
        url: URL of the image to download
        output_dir: Directory to save the image to
        file_id: Optional ID for the file
        
    Returns:
        Path to the downloaded image
    """
    extension = get_extension_from_url(url, default=".png")
    return download_file(url, output_dir, file_id, extension)

def download_video(url: str, output_dir: Path, file_id: Optional[str] = None) -> Path:
    """Specialized function for downloading videos.
    
    Args:
        url: URL of the video to download
        output_dir: Directory to save the video to
        file_id: Optional ID for the file
        
    Returns:
        Path to the downloaded video
    """
    extension = get_extension_from_url(url, default=".mp4")
    return download_file(url, output_dir, file_id, extension)

def get_extension_from_url(url: str, default: str = "") -> str:
    """Extract the file extension from a URL.
    
    Args:
        url: URL to extract extension from
        default: Default extension to use if none can be extracted
        
    Returns:
        File extension (including the dot) or default if none found
    """
    # Parse URL path
    from urllib.parse import urlparse
    path = urlparse(url).path
    
    # Get extension
    ext = os.path.splitext(path)[1]
    
    # Return extension or default
    return ext if ext else default

def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

# Add alias for consistency with new naming convention
ensure_directory_exists = ensure_dir_exists 