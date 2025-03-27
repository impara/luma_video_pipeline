#!/usr/bin/env python3
"""
YouTube Uploader - Standalone tool for uploading videos to YouTube.
Can be used independently after generating videos with the Video Pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from core.config import Config
from integrations.youtube.client import YouTubeClient
from media.sdxl_client import SDXLClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Upload videos to YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a standard landscape video:
  python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt
  
  # Upload a YouTube Shorts video:
  python integrations/youtube/uploader.py --video-path output/videos/my_shorts.mp4 --metadata youtube_metadata.txt --shorts
  
  # Use custom credentials file:
  python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt --credentials my_credentials.json
  
  # Generate and use a thumbnail from the metadata description:
  python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt --generate-thumbnail
  
  # Use an existing thumbnail image:
  python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt --thumbnail my_thumbnail.jpg
"""
    )
    
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the video file to upload"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to YouTube metadata file (title, description, tags, etc.)"
    )
    
    parser.add_argument(
        "--credentials",
        type=str,
        default="client_secret.json",
        help="Path to YouTube API credentials file (default: client_secret.json)"
    )
    
    parser.add_argument(
        "--shorts",
        action="store_true",
        help="Indicate that this is a YouTube Shorts video (adds #Shorts tag)"
    )
    
    # Thumbnail options (mutually exclusive)
    thumbnail_group = parser.add_mutually_exclusive_group()
    thumbnail_group.add_argument(
        "--thumbnail",
        type=str,
        help="Path to existing thumbnail image file"
    )
    
    thumbnail_group.add_argument(
        "--generate-thumbnail",
        action="store_true",
        help="Generate a thumbnail from the description in the metadata file"
    )
    
    # Thumbnail generation options
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=1280,
        help="Width of the generated thumbnail (default: 1280)"
    )
    
    parser.add_argument(
        "--thumbnail-height",
        type=int,
        default=720,
        help="Height of the generated thumbnail (default: 720)"
    )
    
    return parser

def validate_files(video_path: str, metadata_path: str, credentials_path: str) -> bool:
    """Validate that all required files exist"""
    missing_files = []
    
    if not os.path.exists(video_path):
        missing_files.append(f"Video file: {video_path}")
    
    if not os.path.exists(metadata_path):
        missing_files.append(f"Metadata file: {metadata_path}")
    
    if not os.path.exists(credentials_path):
        missing_files.append(f"Credentials file: {credentials_path}")
    
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"- {file}")
        return False
    
    return True

def generate_thumbnail(prompt: str, width: int = 1280, height: int = 720) -> Optional[str]:
    """
    Generate a thumbnail image using SDXL based on the provided prompt
    
    Args:
        prompt: Text description for the thumbnail
        width: Width of the thumbnail
        height: Height of the thumbnail
        
    Returns:
        Path to the generated thumbnail image, or None if generation failed
    """
    try:
        # Load configuration
        config = Config()
        
        # Initialize SDXL client
        logger.info("Initializing SDXL client for thumbnail generation...")
        sdxl_client = SDXLClient(
            api_token=config.replicate_token,
            dev_mode=config.is_dev_mode
        )
        
        # Prepare thumbnail directory
        thumbnail_dir = Path("thumbnails")
        thumbnail_dir.mkdir(exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        thumbnail_path = thumbnail_dir / f"thumbnail_{timestamp}.jpg"
        
        # Enhanced prompt for better thumbnails
        enhanced_prompt = f"YouTube thumbnail: {prompt}. High quality, vibrant colors, clear composition, professional looking, eye-catching, 4K quality"
        
        # Generate the image
        logger.info(f"Generating thumbnail with prompt: {prompt}")
        logger.info(f"Dimensions: {width}x{height}")
        
        # Configure SDXL parameters
        sdxl_config = {
            "width": width,
            "height": height,
            "num_outputs": 1,
            "guidance_scale": 8.0,  # Higher guidance scale for more prompt adherence
            "apply_watermark": False,
            "high_noise_frac": 0.8,  # Better quality
            "negative_prompt": "blurry, low quality, low resolution, poorly composed, amateur, text, watermark"
        }
        
        # Generate the image
        image_paths = sdxl_client.generate_media(enhanced_prompt, sdxl_config)
        
        if not image_paths or len(image_paths) == 0:
            logger.error("Failed to generate thumbnail: No images returned")
            return None
        
        # Use the first generated image
        source_path = image_paths[0]
        
        # Copy to thumbnail directory with proper name
        import shutil
        shutil.copy(source_path, thumbnail_path)
        
        logger.info(f"Thumbnail generated successfully: {thumbnail_path}")
        return str(thumbnail_path)
        
    except Exception as e:
        logger.error(f"Thumbnail generation failed: {str(e)}")
        return None

def upload_to_youtube(args: argparse.Namespace) -> bool:
    """Upload a video to YouTube using the provided arguments"""
    # Validate files
    if not validate_files(args.video_path, args.metadata, args.credentials):
        return False
    
    # Initialize YouTube client
    youtube_client = YouTubeClient(credentials_file=args.credentials)
    
    # Authenticate
    logger.info("Authenticating with YouTube...")
    if not youtube_client.authenticate():
        logger.error("YouTube authentication failed. Please check your credentials.")
        return False
    
    # Parse metadata file
    try:
        logger.info(f"Parsing metadata from {args.metadata}...")
        metadata = youtube_client.parse_metadata_file(args.metadata)
        
        # Log parsed metadata for debugging
        logger.debug(f"Parsed metadata: {metadata}")
        
        # Handle Shorts format
        if args.shorts:
            logger.info("Processing as YouTube Shorts video")
            
            # Ensure #Shorts tag is included
            if "tags" in metadata and "#Shorts" not in metadata["tags"]:
                metadata["tags"].append("#Shorts")
                
            # Add #Shorts to the title if not already present
            if "title" in metadata and "#Shorts" not in metadata["title"]:
                metadata["title"] += " #Shorts"
        
        # Handle educational metadata
        if "educational_type" in metadata:
            logger.info(f"Setting educational type: {metadata['educational_type']}")
            valid_types = ["concept_overview", "problem_walkthrough", "tutorial", "lecture", "study_guide"]
            if metadata["educational_type"].lower() in valid_types:
                metadata["education_type"] = metadata["educational_type"].lower()
            else:
                logger.warning(f"Invalid educational type: {metadata['educational_type']}. Must be one of: {', '.join(valid_types)}")
                del metadata["educational_type"]
        
        # Handle thumbnail generation if requested
        thumbnail_path = None
        if args.generate_thumbnail:
            if "thumbnail_prompt" in metadata and metadata["thumbnail_prompt"]:
                logger.info(f"Generating thumbnail with prompt: {metadata['thumbnail_prompt']}")
                thumbnail_path = generate_thumbnail(
                    metadata["thumbnail_prompt"],
                    width=args.thumbnail_width,
                    height=args.thumbnail_height
                )
                if not thumbnail_path:
                    logger.warning("Thumbnail generation failed. Continuing without thumbnail.")
                else:
                    logger.info(f"Thumbnail successfully generated at: {thumbnail_path}")
            else:
                logger.warning("No thumbnail prompt found in metadata. Skipping thumbnail generation.")
        elif args.thumbnail:
            thumbnail_path = args.thumbnail
            if not os.path.exists(thumbnail_path):
                logger.warning(f"Thumbnail file not found: {thumbnail_path}")
                thumbnail_path = None
        
        # Upload the video
        logger.info(f"Uploading {args.video_path} to YouTube...")
        success, video_id, error = youtube_client.upload_video(
            video_path=args.video_path,
            metadata=metadata,
            is_shorts=args.shorts
        )
        
        if success:
            logger.info(f"Video uploaded successfully!")
            logger.info(f"Video ID: {video_id}")
            logger.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            
            # Set thumbnail if available
            if thumbnail_path:
                logger.info(f"Setting thumbnail from {thumbnail_path}...")
                if youtube_client.set_thumbnail(video_id, thumbnail_path):
                    logger.info("Thumbnail set successfully")
                else:
                    logger.warning("Failed to set thumbnail")
            
            # Check upload status
            logger.info("Checking video processing status...")
            status = youtube_client.get_upload_status(video_id)
            if status["success"]:
                logger.info(f"Video status: {status['processing_status']}")
                logger.info(f"Privacy: {status['privacy_status']}")
            else:
                logger.warning(f"Could not retrieve video status: {status.get('error', 'Unknown error')}")
            
            return True
        else:
            logger.error(f"Upload failed: {error}")
            return False
    
    except Exception as e:
        logger.error(f"YouTube upload error: {str(e)}")
        return False

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    logger.info("=== YouTube Uploader ===")
    logger.info(f"Video: {args.video_path}")
    logger.info(f"Metadata: {args.metadata}")
    logger.info(f"Shorts format: {'Yes' if args.shorts else 'No'}")
    
    if args.generate_thumbnail:
        logger.info("Will generate thumbnail from metadata description")
    elif args.thumbnail:
        logger.info(f"Will use thumbnail: {args.thumbnail}")
    
    if upload_to_youtube(args):
        logger.info("Upload process completed successfully")
        return 0
    else:
        logger.error("Upload process failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 