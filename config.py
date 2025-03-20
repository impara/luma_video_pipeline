"""
Centralized configuration management for the video pipeline.
Handles environment variables and configuration settings.
"""

import os
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

class ConfigError(Exception):
    """Raised when there's a configuration error"""
    pass

class Config:
    """Centralized configuration management"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration, optionally from a specific .env file
        
        Args:
            env_file: Optional path to .env file. If None, uses default .env in current directory
        """
        # Load environment variables
        load_dotenv(env_file)
        
        # Required API tokens
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        self.elevenlabs_token = os.getenv("ELEVENLABS_API_KEY")
        
        # UnrealSpeech API token (optional if using ElevenLabs)
        self.unrealspeech_token = os.getenv("API_KEY")
        
        # Gemini API key (for Google Generative AI)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Optional development flags
        self.dev_mode = os.getenv("VIDEO_PIPELINE_DEV_MODE", "").lower() in ("true", "1", "yes")
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required configuration is present"""
        missing = []
        
        # Validate based on possible use cases (some API keys may only be needed depending on options)
        if not self.replicate_token:
            # Replicate token is not required if using only Gemini for image generation
            if not self.gemini_api_key:
                missing.append("REPLICATE_API_TOKEN or GEMINI_API_KEY")
        
        # Either ElevenLabs or UnrealSpeech API key is required
        if not self.elevenlabs_token and not self.unrealspeech_token:
            missing.append("ELEVENLABS_API_KEY or API_KEY (for UnrealSpeech)")
            
        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set them in your .env file or environment."
            )
    
    @property
    def is_dev_mode(self) -> bool:
        """Whether the pipeline is running in development mode"""
        return self.dev_mode 