"""
YouTube Client module for the Video Pipeline.
Handles authentication, uploads, and metadata management for YouTube videos.
"""

import os
import time
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, Union
import http.client
import httplib2
import random
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from PIL import Image

from core.error_handling import retry_api_call

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YouTube API scopes required for uploading videos
SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 
          'https://www.googleapis.com/auth/youtube']

# Default metadata if not specified
DEFAULT_CATEGORY = "27"  # Education (was 22 for People & Blogs)
DEFAULT_PRIVACY = "private"  # Start with private for safety
DEFAULT_TAGS = ["AI generated", "Video Pipeline"]

# YouTube category ID mapping
CATEGORY_MAPPING = {
    "film & animation": "1",
    "autos & vehicles": "2",
    "music": "10",
    "pets & animals": "15",
    "sports": "17",
    "short movies": "18",
    "travel & events": "19",
    "gaming": "20",
    "videoblogging": "21",
    "people & blogs": "22",
    "comedy": "23",
    "entertainment": "24",
    "news & politics": "25",
    "howto & style": "26",
    "education": "27",
    "science & technology": "28",
    "nonprofits & activism": "29",
    "movies": "30",
    "anime/animation": "31",
    "action/adventure": "32",
    "classics": "33",
    "documentary": "35",
    "drama": "36",
    "family": "37",
    "foreign": "38",
    "horror": "39",
    "sci-fi/fantasy": "40",
    "thriller": "41",
    "shorts": "42",
    "shows": "43",
    "trailers": "44"
}

class YouTubeUploadError(Exception):
    """Exception raised for YouTube upload errors"""
    pass

class YouTubeClient:
    """Client for interacting with YouTube API"""
    
    def __init__(self, credentials_file: str = "client_secret.json", token_file: str = "youtube_token.json"):
        """
        Initialize the YouTube client
        
        Args:
            credentials_file: Path to the OAuth client secrets file
            token_file: Path to save/load the user's access and refresh tokens
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.youtube = None
        self.authenticated = False
        
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API using OAuth2
        
        Returns:
            bool: True if authentication was successful
        """
        creds = None
        
        # Check if token file exists
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_info(
                    json.loads(open(self.token_file).read()), SCOPES)
            except Exception as e:
                logger.warning(f"Error loading credentials from token file: {e}")
        
        # If credentials don't exist or are invalid, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Error refreshing credentials: {e}")
                    creds = None
            
            # If still no valid credentials, need user to authenticate
            if not creds:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.error(f"Authentication failed: {e}")
                    return False
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        try:
            # Build the YouTube API client
            self.youtube = build('youtube', 'v3', credentials=creds)
            self.authenticated = True
            logger.info("Successfully authenticated with YouTube API")
            return True
        except Exception as e:
            logger.error(f"Failed to build YouTube API client: {e}")
            return False
    
    def parse_metadata_file(self, metadata_file: str) -> Dict[str, Any]:
        """
        Parse YouTube metadata from a text file
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            Dict containing parsed metadata
        """
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        metadata = {
            "title": "",
            "description": "",
            "tags": [],
            "category": DEFAULT_CATEGORY,
            "privacy": DEFAULT_PRIVACY,
            "thumbnail_prompt": "",
            "educational_type": ""
        }
        
        current_section = None
        section_content = ""
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for section headers
                if line.startswith("# Title"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "title"
                    section_content = ""
                elif line.startswith("# Description"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "description"
                    section_content = ""
                elif line.startswith("# Tags"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "tags"
                    section_content = ""
                elif line.startswith("# Category"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "category"
                    section_content = ""
                elif line.startswith("# Privacy"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "privacy"
                    section_content = ""
                elif line.startswith("# Educational Type"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "educational_type"
                    section_content = ""
                elif line.startswith("# Thumbnail"):
                    if current_section and section_content:
                        self._process_section(metadata, current_section, section_content)
                    current_section = "thumbnail"
                    section_content = ""
                elif current_section:
                    # Add content to current section
                    if section_content:
                        section_content += "\n"
                    section_content += line
        
        # Process the last section
        if current_section and section_content:
            self._process_section(metadata, current_section, section_content)
        
        return metadata
    
    def _process_section(self, metadata: Dict[str, Any], section: str, content: str) -> None:
        """Process a section of the metadata file"""
        content = content.strip()
        
        if section == "title":
            metadata["title"] = content
        elif section == "description":
            metadata["description"] = content
        elif section == "tags":
            # Split tags by commas and strip whitespace
            metadata["tags"] = [tag.strip() for tag in content.split(",")]
        elif section == "category":
            # Check if the category is a name that needs to be converted to an ID
            content_lower = content.lower()
            if content_lower in CATEGORY_MAPPING:
                metadata["category"] = CATEGORY_MAPPING[content_lower]
                logger.info(f"Converted category name '{content}' to ID: {metadata['category']}")
            else:
                # Assume it's already a valid ID or will be handled by error checking later
                metadata["category"] = content
                logger.warning(f"Unknown category name: '{content}'. Using as-is. Valid categories are: {', '.join(CATEGORY_MAPPING.keys())}")
        elif section == "privacy":
            if content.lower() in ["public", "private", "unlisted"]:
                metadata["privacy"] = content.lower()
        elif section == "educational_type":
            valid_types = ["concept_overview", "problem_walkthrough", "tutorial", "lecture", "study_guide"]
            if content.lower() in valid_types:
                metadata["educational_type"] = content.lower()
        elif section == "thumbnail":
            metadata["thumbnail_prompt"] = content
    
    def upload_video(self, video_path: str, metadata: Dict[str, Any], is_shorts: bool = False) -> Tuple[bool, str, str]:
        """
        Upload a video to YouTube
        
        Args:
            video_path: Path to the video file
            metadata: Dictionary containing video metadata
            is_shorts: Whether this is a YouTube Shorts video
            
        Returns:
            Tuple of (success, video_id, error_message)
        """
        if not self.authenticated:
            if not self.authenticate():
                return False, "", "Authentication failed"
        
        if not os.path.exists(video_path):
            return False, "", f"Video file not found: {video_path}"
        
        # Prepare video metadata
        title = metadata.get("title", os.path.basename(video_path))
        description = metadata.get("description", "")
        
        # Add #Shorts hashtag for Shorts videos
        tags = metadata.get("tags", DEFAULT_TAGS)
        if is_shorts and "#Shorts" not in tags:
            tags.append("#Shorts")
        
        # Verify category ID validity
        category = metadata.get("category", DEFAULT_CATEGORY)
        valid_category_ids = list(CATEGORY_MAPPING.values())
        if category not in valid_category_ids:
            logger.warning(f"Category ID '{category}' may not be valid. Valid IDs are: {', '.join(valid_category_ids)}")
            if is_shorts:
                # For shorts, use Entertainment category (24) as fallback
                category = "24"  # Entertainment
                logger.info(f"Using Entertainment (24) category for Shorts video")
            else:
                # Use Education (27) as fallback
                category = "27"  # Education
                logger.info(f"Using Education (27) category as fallback")
        
        # Create video insert request
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category
            },
            "status": {
                "privacyStatus": metadata.get("privacy", DEFAULT_PRIVACY),
                "selfDeclaredMadeForKids": False
            }
        }
        
        # Add educational metadata if present
        if "educational_type" in metadata and metadata["educational_type"]:
            body["snippet"]["educationalDetails"] = {
                "type": metadata["educational_type"]
            }
        
        # Upload the video
        try:
            logger.info(f"Starting upload of {video_path} to YouTube...")
            
            # Create MediaFileUpload object
            media = MediaFileUpload(
                video_path,
                mimetype="video/mp4",
                resumable=True,
                chunksize=1024*1024
            )
            
            # Create the video insert request
            insert_request = self.youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute the request with progress tracking
            response = None
            last_progress = 0
            
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    if progress > last_progress + 5:  # Report every 5% progress
                        logger.info(f"Upload progress: {progress}%")
                        last_progress = progress
            
            video_id = response["id"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Video uploaded successfully! Video ID: {video_id}")
            logger.info(f"Video URL: {video_url}")
            
            return True, video_id, ""
            
        except HttpError as e:
            error_content = json.loads(e.content)
            error_message = error_content.get("error", {}).get("message", str(e))
            logger.error(f"YouTube upload failed: {error_message}")
            return False, "", f"YouTube API error: {error_message}"
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return False, "", f"Upload error: {str(e)}"
    
    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """
        Set a custom thumbnail for a YouTube video
        
        Args:
            video_id: YouTube video ID
            thumbnail_path: Path to thumbnail image file
            
        Returns:
            bool: True if successful
        """
        if not self.authenticated:
            logger.warning("Not authenticated. Attempting to authenticate...")
            if not self.authenticate():
                logger.error("Failed to authenticate for setting thumbnail")
                return False
        
        if not os.path.exists(thumbnail_path):
            logger.error(f"Thumbnail file not found: {thumbnail_path}")
            return False
        
        try:
            # Log thumbnail details
            logger.debug(f"Setting thumbnail for video {video_id}")
            logger.debug(f"Thumbnail path: {thumbnail_path}")
            
            # Verify the image file is readable
            try:
                img = Image.open(thumbnail_path)
                logger.debug(f"Thumbnail dimensions: {img.size}")
                img.close()
            except Exception as e:
                logger.error(f"Failed to read thumbnail image: {str(e)}")
                return False
            
            # Upload the thumbnail
            media = MediaFileUpload(
                thumbnail_path,
                mimetype="image/jpeg",
                resumable=True
            )
            
            # Set the thumbnail
            logger.debug("Executing thumbnail set request...")
            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            
            logger.info(f"Thumbnail set successfully for video {video_id}")
            return True
            
        except HttpError as e:
            error_content = json.loads(e.content)
            error_message = error_content.get("error", {}).get("message", str(e))
            logger.error(f"Setting thumbnail failed: {error_message}")
            return False
            
        except Exception as e:
            logger.error(f"Setting thumbnail failed: {str(e)}")
            return False
    
    def get_upload_status(self, video_id: str) -> Dict[str, Any]:
        """
        Get the status of an uploaded video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict containing video status information
        """
        if not self.authenticated:
            if not self.authenticate():
                return {"success": False, "error": "Authentication failed"}
        
        try:
            # Get video status
            response = self.youtube.videos().list(
                part="status,processingDetails",
                id=video_id
            ).execute()
            
            if not response.get("items"):
                return {
                    "success": False,
                    "error": f"Video {video_id} not found"
                }
            
            video = response["items"][0]
            status = video.get("status", {})
            processing = video.get("processingDetails", {})
            
            return {
                "success": True,
                "video_id": video_id,
                "privacy_status": status.get("privacyStatus"),
                "upload_status": status.get("uploadStatus"),
                "processing_status": processing.get("processingStatus"),
                "processing_progress": processing.get("processingProgress", {}),
                "processing_failure_reason": processing.get("processingFailureReason")
            }
            
        except Exception as e:
            logger.error(f"Failed to get video status: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get video status: {str(e)}"
            } 