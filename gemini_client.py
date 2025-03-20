"""
Client for generating images using Google's Gemini API.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    import google.generativeai as genai
    HAS_GENAI = False
    
from dotenv import load_dotenv
import logging

from media_client import GenerativeMediaClient
from cache_handler import CacheHandler
from error_handling import MediaGenerationError, retry_media_generation
from utils import ensure_dir_exists

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient(GenerativeMediaClient):
    """Client for generating images using Google's Gemini API.
    
    This client uses Gemini 2.0 Flash Experimental for multimodal text-to-image generation
    which should work with the free tier API key.
    """
    
    # Gemini experimental model that supports image generation
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    DOWNLOAD_DIR = Path("generated_images")
    
    def __init__(self, api_key: str = None, dev_mode: bool = False):
        """Initialize the Gemini client.
        
        Args:
            api_key: Optional Google API key. If not provided, will look for GEMINI_API_KEY env var.
            dev_mode: If True, forces cache usage and skips API calls when possible.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter is required")
            
        # Configure the Gemini API
        if HAS_GENAI:
            # Using newer google.genai package
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Using older google.generativeai package
            genai.configure(api_key=self.api_key)
        
        # Ensure download directory exists
        ensure_dir_exists(self.DOWNLOAD_DIR)
        
        # Initialize cache handler
        self.cache_handler = CacheHandler(
            cache_dir=self.DOWNLOAD_DIR,
            cache_file="gemini_cache.json",  # Use a separate cache file for Gemini
            dev_mode=dev_mode
        )
        
        self.dev_mode = dev_mode
        
        # Define core parameters for cache key generation
        self.core_cache_params = ["prompt", "model", "width", "height", "num_outputs", "video_format"]
    
    @retry_media_generation
    def generate_media(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate images using Gemini 2.0 Flash Experimental model with multimodal capabilities.
        
        Args:
            prompt: Text prompt for image generation
            config: Optional configuration parameters including:
                - width: Image width (for context in the prompt)
                - height: Image height (for context in the prompt)
                - num_outputs: Number of images to generate (default: 1)
                - video_format: Format string ("landscape", "portrait", "square")
                
        Returns:
            List of paths to generated image files
        """
        config = config or {}
        
        # Prepare all parameters for caching and generation
        params = {
            "prompt": prompt,
            "model": self.GEMINI_MODEL,
            "width": config.get("width", 1024),
            "height": config.get("height", 576),
            "num_outputs": config.get("num_outputs", 1),
            "video_format": config.get("video_format", "landscape"),
            "temperature": config.get("temperature", 0.8)
        }
        
        # Add any additional parameters from config
        for key, value in config.items():
            if key not in params:
                params[key] = value
        
        # Define generation function for cache handler
        def generate_images_func():
            if self.dev_mode:
                logger.warning("Dev mode enabled but no cached images found, attempting generation...")
            return self._generate_with_gemini_multimodal(prompt, params)
        
        # Use the cache handler for both dev and production modes
        file_paths, from_cache = self.cache_handler.get_or_add_cached_media(
            params=params,
            core_params=self.core_cache_params,
            generator_func=generate_images_func,
            result_key='file_paths'
        )
        
        return file_paths
    
    def _generate_with_gemini_multimodal(self, prompt: str, params: Dict[str, Any]) -> List[str]:
        """Generate images using Gemini 2.0 Flash Experimental model's multimodal capabilities.
        
        Args:
            prompt: Text prompt for image generation
            params: Generation parameters
            
        Returns:
            List of paths to generated image files
        """
        # Get dimensions and format from params
        width = params.get("width", 1024)
        height = params.get("height", 576)
        num_outputs = params.get("num_outputs", 1)
        video_format = params.get("video_format", "landscape")
        temperature = params.get("temperature", 0.8)
        
        # Format ratio for better results
        if video_format == "landscape":
            ratio = "16:9"
        elif video_format == "short":
            ratio = "9:16"
        else:  # square
            ratio = "1:1"
            
        # Enhanced prompt with more context for better generation
        enhanced_prompt = (
            f"Generate a high-quality, detailed image for: {prompt}. "
            f"Create an image with {width}x{height} pixels in {ratio} ratio. "
            f"The image should be photorealistic with vibrant colors and sharp details. "
            f"Do not include any text or watermarks."
        )
        
        logger.info(f"Generating images with Gemini {self.GEMINI_MODEL} using multimodal output")
        
        image_paths = []
        max_attempts_per_image = 3
        
        # Generate the requested number of images
        for i in range(num_outputs):
            success = False
            attempts = 0
            
            # Try multiple times for each image if needed
            while not success and attempts < max_attempts_per_image:
                attempts += 1
                try:
                    # Generate a unique request ID for this image
                    request_id = str(uuid.uuid4())
                    
                    logger.info(f"Image generation attempt {attempts} for image {i+1}")
                    
                    # Generate content with multimodal output (requesting an image in response)
                    response = self.client.models.generate_content(
                        model=self.GEMINI_MODEL,
                        contents=enhanced_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["Text", "Image"],
                            temperature=temperature,
                            # Use a different seed for each image if generating multiple
                            seed=int.from_bytes(os.urandom(2), byteorder="big") if i > 0 else None
                        )
                    )
                    
                    # Validate response structure
                    if not hasattr(response, 'candidates') or not response.candidates:
                        logger.error(f"Invalid response: No candidates returned")
                        continue
                        
                    if not response.candidates[0]:
                        logger.error(f"Invalid response: First candidate is empty")
                        continue
                        
                    if not hasattr(response.candidates[0], 'content') or response.candidates[0].content is None:
                        logger.error(f"Invalid response: No content in first candidate")
                        continue
                        
                    if not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
                        logger.error(f"Invalid response: No parts in content")
                        continue
                    
                    # Extract image from response
                    image_extracted = False
                    
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            if part.inline_data.mime_type.startswith('image/'):
                                # Save the image data directly with the request ID
                                file_path = self.DOWNLOAD_DIR / f"gemini_{request_id}.png"
                                with open(file_path, 'wb') as f:
                                    f.write(part.inline_data.data)
                                
                                # Verify the image
                                try:
                                    with Image.open(file_path) as img:
                                        logger.info(f"Successfully saved image {i+1}: {img.format}, size: {img.size}")
                                        
                                        # Resize the image to match the requested dimensions
                                        if img.size != (width, height):
                                            logger.info(f"Resizing image from {img.size} to {width}x{height}")
                                            resized_img = img.resize((width, height), Image.LANCZOS)
                                            resized_img.save(file_path)
                                            logger.info(f"Image resized to {width}x{height}")
                                        
                                        image_paths.append(str(file_path))
                                        image_extracted = True
                                        success = True
                                        break  # Found and saved an image, exit the loop
                                except Exception as e:
                                    logger.error(f"Failed to verify image {i+1}: {str(e)}")
                                    if file_path.exists():
                                        file_path.unlink()  # Delete invalid image
                    
                    if not image_extracted:
                        logger.warning(f"No image data found in response for generation {i+1}, attempt {attempts}")
                        
                except Exception as e:
                    logger.error(f"Error generating image {i+1}, attempt {attempts}: {str(e)}")
            
            if not success:
                logger.error(f"Failed to generate image {i+1} after {max_attempts_per_image} attempts")
        
        if not image_paths:
            # Fall back to a default image if available
            default_image_path = self.DOWNLOAD_DIR / "default_placeholder.png"
            if default_image_path.exists():
                logger.warning(f"Using default placeholder image as fallback")
                return [str(default_image_path)]
            else:
                raise MediaGenerationError("Failed to generate any images with Gemini and no fallback available")
        
        logger.info(f"Successfully generated {len(image_paths)} images with Gemini")
        return image_paths


# Example usage:
"""
client = GeminiClient()
images = client.generate_media(
    prompt="A serene landscape at sunset",
    config={
        "width": 1024,
        "height": 576,
        "num_outputs": 1,
        "video_format": "landscape"
    }
)
print(f"Generated images: {images}")
"""
