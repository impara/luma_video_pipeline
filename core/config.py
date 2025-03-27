"""
Centralized configuration management for the video pipeline.
Handles environment variables and configuration settings.
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Raised when there's a configuration error"""
    pass

class Config:
    """Centralized configuration management"""
    
    def __init__(self, env_file: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize configuration, optionally from a specific .env file and a config file
        
        Args:
            env_file: Optional path to .env file. If None, uses default .env in current directory
            config_file: Optional path to JSON config file
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
        
        # Set up output directories
        self.base_output_dir = Path("output")
        self.video_output_dir = self.base_output_dir / "videos"
        self.image_output_dir = self.base_output_dir / "images"
        self.audio_output_dir = self.base_output_dir / "audio"
        self.captions_output_dir = self.base_output_dir / "captions"
        self.temp_output_dir = self.base_output_dir / "temp"
        
        # Memory management settings
        available_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        recommended_max = int(available_memory * 0.7)  # Use up to 70% of system memory
        
        self.memory_settings = {
            'window_size': 3,  # Number of scenes to process at once
            'max_memory_bytes': recommended_max,
            'max_memory_mb': recommended_max // (1024 * 1024),
            'cleanup_threshold': 0.8,  # Cleanup when 80% of max memory is used
            'temp_dir': self.temp_output_dir / 'processing',
            'use_ram_disk': True,  # Attempt to use RAM disk for temp files
            'min_free_memory_mb': 1024  # Minimum free memory to maintain (1GB)
        }
        
        # Load custom config if provided
        if config_file:
            self.load_config(config_file)
        
        # Ensure all output directories exist
        self._ensure_output_dirs()
        
        # Validate required configuration
        self._validate_config()
    
    def _ensure_output_dirs(self):
        """Ensure all output directories exist"""
        self.base_output_dir.mkdir(exist_ok=True)
        self.video_output_dir.mkdir(exist_ok=True)
        self.image_output_dir.mkdir(exist_ok=True)
        self.audio_output_dir.mkdir(exist_ok=True)
        self.captions_output_dir.mkdir(exist_ok=True)
        self.temp_output_dir.mkdir(exist_ok=True)
    
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

    def load_config(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON config file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update memory settings if provided
            if 'memory_settings' in config_data:
                self.memory_settings.update(config_data['memory_settings'])
                
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            raise ConfigError(f"Failed to load config file {config_file}: {e}")
            
    def get_memory_settings(self) -> Dict[str, Any]:
        """
        Get current memory management settings.
        
        Returns:
            Dictionary of memory management settings
        """
        return self.memory_settings.copy()
        
    def update_memory_settings(self, settings: Dict[str, Any]):
        """
        Update memory management settings.
        
        Args:
            settings: Dictionary of settings to update
        """
        self.memory_settings.update(settings)
        logger.info("Updated memory management settings")
        
    def get_temp_dir(self) -> Path:
        """
        Get the appropriate temporary directory based on settings.
        
        Returns:
            Path to temporary directory
        """
        if self.memory_settings['use_ram_disk']:
            from core.memory_utils import create_memory_efficient_temp_dir
            ram_dir = create_memory_efficient_temp_dir()
            if ram_dir:
                return ram_dir
                
        return self.memory_settings['temp_dir'] 