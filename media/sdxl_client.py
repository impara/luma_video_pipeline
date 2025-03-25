"""
SDXL client for image generation via Replicate.
"""

import os
import uuid
import requests
import hashlib
import json
import replicate
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from PIL import Image
from media.client_base import MediaClient
from core.cache_handler import CacheHandler
from core.error_handling import ImageGenerationError, retry_media_generation
from core.utils import ensure_directory_exists, download_file_to_path
from core.config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDXLClient(MediaClient):
    """Client for generating images using Replicate's SDXL model."""
    
    MODEL_ID = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    
    def __init__(self, api_token: str = None, dev_mode: bool = False):
        """Initialize the SDXL client.
        
        Args:
            api_token: Optional Replicate API token. If not provided, will look for REPLICATE_API_TOKEN env var.
            dev_mode: If True, forces cache usage and skips API calls when possible.
        """
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable or api_token parameter is required")
            
        # Load config for output directories
        self.config = Config()
        self.DOWNLOAD_DIR = self.config.image_output_dir
            
        # Set the API token for the replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
        # Ensure download directory exists
        ensure_directory_exists(self.DOWNLOAD_DIR)
        
        # Initialize cache handler
        self.cache_handler = CacheHandler(
            cache_dir=self.DOWNLOAD_DIR,
            cache_file="cache.json",
            dev_mode=dev_mode
        )
        
        self.dev_mode = dev_mode
        
        # Define core parameters for cache key generation
        self.core_cache_params = ["prompt", "width", "height", "num_outputs", "guidance_scale"]
        
        # Enhanced scene-specific settings
        self.scene_settings = {
            "landscape": {
                "guidance_scale": 8.5,
                "num_inference_steps": 50,
                "negative_prompt": "blurry, low quality, distorted, deformed",
                "width": 1024,
                "height": 576
            },
            "portrait": {
                "guidance_scale": 9.0,
                "num_inference_steps": 45,
                "negative_prompt": "blurry, low quality, distorted, deformed faces, asymmetric eyes",
                "width": 768,
                "height": 1024
            },
            "short": {  # 9:16 aspect ratio for social media shorts
                "guidance_scale": 9.0,
                "num_inference_steps": 50,
                "negative_prompt": "blurry, low quality, distorted, deformed",
                "width": 576,
                "height": 1024
            },
            "square": {  # 1:1 aspect ratio
                "guidance_scale": 8.0,
                "num_inference_steps": 50,
                "negative_prompt": "blurry, low quality, distorted, deformed",
                "width": 1024,
                "height": 1024
            },
            "abstract": {
                "guidance_scale": 7.0,
                "num_inference_steps": 40,
                "negative_prompt": "realistic, photographic",
                "width": 1024,
                "height": 576
            }
        }
        
        # Style consistency tracking
        self.style_consistency = {
            "base_style": {
                "prompt_prefix": "high quality, detailed, ",
                "prompt_suffix": ", vivid colors, sharp focus",
                "negative_prompt": "blurry, low resolution, pixelated, cropped, frame, border, watermark",
                "base_guidance_scale": 7.5,
                "style_strength": 0.8
            },
            "color_palette": {
                "forest": "verdant greens, earthy browns, dappled sunlight",
                "magical": "ethereal blues, purples, soft glowing highlights",
                "urban": "neon accents, steel blues, concrete grays",
                "sunset": "golden yellows, deep oranges, muted purples"
            }
        }
        
        # Enhanced scene context tracking with character and object memory
        self.scene_context = {
            "previous_scenes": [],
            "current_settings": {},
            "consistent_elements": [],
            "character_descriptions": {},  # Store character descriptions for consistency
            "established_elements": [],    # Track visually established elements
            "last_image_path": None,       # Track the last generated image for img2img chaining
            "color_schemes": {},           # Track color schemes used in different scenes
            "style_keywords": [],          # Track style keywords for consistent aesthetics
            "environment_details": {}      # Track environmental details across scenes
        }
        
        # Initialize default scene analysis
        self._scene_keywords = {
            "landscape": ["mountains", "valley", "forest", "ocean", "nature", "landscape", "sky", "sunset", "outdoor"],
            "portrait": ["person", "character", "face", "portrait", "woman", "man", "figure", "standing"],
            "abstract": ["abstract", "surreal", "digital", "fractal", "geometric", "fantasy", "dreamlike"]
        }
        
        # Refiner options based on schema
        self.refiner_options = {
            "no_refiner": {
                "enabled": False
            },
            "expert_ensemble_refiner": {
                "enabled": True,
                "high_noise_frac": 0.8
            },
            "base_image_refiner": {
                "enabled": True,
                "refine_steps": None  # Will use num_inference_steps if None
            }
        }
        
        # Available schedulers
        self.schedulers = ["DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM", 
                          "K_EULER_ANCESTRAL", "K_EULER", "PNDM"]
        
        # Maximum scenes to remember for context
        self.max_context_scenes = 3
    
    def _load_cache(self):
        """Load the cache from disk."""
        try:
            if self.CACHE_FILE.exists():
                with open(self.CACHE_FILE, 'r') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            logger.warning(f"Failed to load SDXL cache: {e}")
            self.cache = {}
            
    def _save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save SDXL cache: {e}")
            
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a cache key from the input parameters."""
        # Normalize prompt (strip whitespace, lowercase)
        prompt = params.get("prompt", "").strip().lower()
        
        # Create key parts with core parameters
        key_parts = [
            prompt,
            str(params.get("width", 1024)),
            str(params.get("height", 576)),
            str(params.get("num_outputs", 1)),
            str(params.get("guidance_scale", 7.5))
        ]
        
        # Add optional parameters if they exist
        for opt_param in ["seed", "negative_prompt"]:
            if opt_param in params:
                key_parts.append(f"{opt_param}={params[opt_param]}")
                
        # Join all parts and hash
        key_str = "|".join(key_parts)
        # Use full hash for exact matching
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_abs_path(self, rel_path: Union[str, Path]) -> Path:
        """Convert a relative path to absolute path."""
        path = Path(rel_path)
        if path.is_absolute():
            return path
        # Handle case where generated_images prefix is missing
        if self.DOWNLOAD_DIR.name not in path.parts:
            path = self.DOWNLOAD_DIR / path
        return path.resolve()
        
    def _get_rel_path(self, abs_path: Union[str, Path]) -> Path:
        """Convert an absolute path to relative path from DOWNLOAD_DIR."""
        path = Path(abs_path)
        try:
            return path.relative_to(self.download_dir_abs)
        except ValueError:
            # If path is already relative to DOWNLOAD_DIR, ensure it has the prefix
            if self.DOWNLOAD_DIR.name not in path.parts:
                return self.DOWNLOAD_DIR / path
            return path
    
    def _get_cached_images(self, params: Dict[str, Any]) -> Optional[List[str]]:
        """Check if images exist in cache for given parameters."""
        # Get cache key first
        cache_key = self._get_cache_key(params)
        
        # In dev mode, try to find any cached images
        if self.dev_mode:
            prompt = params.get("prompt", "").strip().lower()
            available_images = []
            
            # Collect all available images
            for key, entry_data in self.cache.items():
                # Check if prompts match
                entry_prompt = entry_data["params"].get("prompt", "").strip().lower()
                if prompt == entry_prompt:
                    image_paths = [self._get_abs_path(p) for p in entry_data["image_paths"]]
                    if all(p.exists() for p in image_paths):
                        available_images.extend(str(p) for p in image_paths)
            
            if available_images:
                # Return a subset matching the requested num_outputs
                num_outputs = params.get("num_outputs", 1)
                selected_images = available_images[:num_outputs]
                return selected_images
                
            return None
            
        # Production mode - check exact cache match
        if cache_key in self.cache:
            entry_data = self.cache[cache_key]
            # Convert relative paths to absolute
            image_paths = [self._get_abs_path(p) for p in entry_data["image_paths"]]
            
            # Verify all files exist
            if all(p.exists() for p in image_paths):
                logger.info(f"Found {len(image_paths)} cached images for key {cache_key}")
                return [str(p) for p in image_paths]
            else:
                # Clean up invalid cache entry
                logger.warning(f"Cache entry {cache_key} has missing files, removing")
                del self.cache[cache_key]
                self._save_cache()
                
        logger.info(f"No cache entry found for key {cache_key}")
        return None
    
    def _cache_images(self, params: Dict[str, Any], image_paths: List[str]):
        """Add images to the cache."""
        # Use the same cache key generation as in _get_cached_images
        cache_key = self._get_cache_key(params)
        
        # Convert absolute paths to relative for storage
        rel_paths = [str(self._get_rel_path(p)) for p in image_paths]
        
        # Store images with their parameters
        self.cache[cache_key] = {
            "image_paths": rel_paths,
            "params": params.copy()  # Store a copy of the parameters
        }
        self._save_cache()
        logger.info(f"Cached {len(image_paths)} images with key: {cache_key}")
    
    def _download_image(self, image_uri: str) -> Path:
        """Download image from URI to local file.
        
        Args:
            image_uri: URI of the image to download
            
        Returns:
            Path: Path to the downloaded image file
        """
        # Use our utility function
        return download_image(image_uri, self.DOWNLOAD_DIR)
    
    def _analyze_scene(self, prompt: str) -> str:
        """
        Analyze scene description to determine the appropriate scene type.
        
        Args:
            prompt: The scene description prompt
            
        Returns:
            str: Scene type (landscape, portrait, or abstract)
        """
        # Convert prompt to lowercase for matching
        text_lower = prompt.lower()
        
        # Count matches for each scene type
        matches = {
            scene_type: sum(1 for keyword in keywords if keyword in text_lower)
            for scene_type, keywords in self._scene_keywords.items()
        }
        
        # Find the scene type with the most keyword matches
        best_match = max(matches.items(), key=lambda x: x[1])
        
        # If no matches or very weak matches, default to landscape
        if best_match[1] == 0:
            return "landscape"
            
        return best_match[0]

    def _extract_entities_from_prompt(self, prompt: str) -> Dict[str, List[str]]:
        """Extract characters, objects, and environment details from prompt.
        
        Args:
            prompt: The scene description prompt
            
        Returns:
            Dict containing extracted entities
        """
        entities = {
            "characters": [],
            "objects": [],
            "environment": []
        }
        
        # Simple extraction based on keywords
        words = prompt.lower().split()
        
        # Check for character indicators
        character_indicators = ["person", "man", "woman", "boy", "girl", "character", 
                               "protagonist", "hero", "heroine", "figure"]
        
        # Check for object indicators
        object_indicators = ["holding", "with a", "carries", "wearing", "beside", "next to"]
        
        # Environment indicators
        environment_indicators = ["inside", "outside", "in a", "at the", "background", 
                                 "surrounded by", "forest", "city", "room", "building"]
        
        # Extract potential character names (capitalize first letter words)
        potential_names = [word for word in prompt.split() 
                          if word[0].isupper() and len(word) > 1 and word.isalpha()]
        
        if potential_names:
            entities["characters"].extend(potential_names)
        
        # Extract based on indicators
        for indicator in character_indicators:
            if indicator in prompt.lower():
                start_idx = prompt.lower().find(indicator)
                # Get surrounding context
                context = prompt[max(0, start_idx-20):min(len(prompt), start_idx+30)]
                entities["characters"].append(context)
                
        for indicator in object_indicators:
            if indicator in prompt.lower():
                start_idx = prompt.lower().find(indicator)
                # Get following context
                context = prompt[start_idx:min(len(prompt), start_idx+30)]
                entities["objects"].append(context)
                
        for indicator in environment_indicators:
            if indicator in prompt.lower():
                start_idx = prompt.lower().find(indicator)
                # Get following context
                context = prompt[start_idx:min(len(prompt), start_idx+40)]
                entities["environment"].append(context)
        
        return entities

    def _update_scene_context(self, params: Dict[str, Any], generated_image_path: Optional[str] = None):
        """Update the scene context with the current parameters.
        
        This helps maintain style consistency between scenes if requested.
        
        Args:
            params: The parameters used for the current generation
            generated_image_path: Path to the generated image (if available)
        """
        # Store a copy of the relevant style parameters
        style_params = {
            "guidance_scale": params.get("guidance_scale"),
            "num_inference_steps": params.get("num_inference_steps"),
            "negative_prompt": params.get("negative_prompt"),
            "scheduler": params.get("scheduler", "K_EULER"),
            "prompt_strength": params.get("prompt_strength", 0.8),
            "width": params.get("width", 1024),
            "height": params.get("height", 576)
        }
        
        # Extract entities from prompt
        prompt = params.get("prompt", "")
        entities = self._extract_entities_from_prompt(prompt)
        
        # Update character descriptions
        for char_desc in entities["characters"]:
            # Use first word as key if it starts with uppercase (likely a name)
            potential_name = char_desc.split()[0]
            if potential_name[0].isupper():
                char_key = potential_name.lower()
                self.scene_context["character_descriptions"][char_key] = char_desc
        
        # Update established elements
        for obj in entities["objects"]:
            if obj not in self.scene_context["established_elements"]:
                self.scene_context["established_elements"].append(obj)
                
        # Update environment details
        for env in entities["environment"]:
            env_key = env.split()[0].lower()
            self.scene_context["environment_details"][env_key] = env
        
        # Store color palette indicators
        for palette_name, palette_desc in self.style_consistency["color_palette"].items():
            if any(word in prompt.lower() for word in palette_desc.split(",")):
                if palette_name not in self.scene_context["color_schemes"]:
                    self.scene_context["color_schemes"][palette_name] = palette_desc
        
        # Update current settings
        self.scene_context["current_settings"] = style_params
        
        # Store last image path for potential img2img chaining
        if generated_image_path:
            self.scene_context["last_image_path"] = generated_image_path
        
        # Add to previous scenes (limited history)
        self.scene_context["previous_scenes"].append({
            "prompt": prompt,
            "params": style_params,
            "entities": entities
        })
        
        # Keep only the last N scenes in history
        if len(self.scene_context["previous_scenes"]) > self.max_context_scenes:
            self.scene_context["previous_scenes"] = self.scene_context["previous_scenes"][-self.max_context_scenes:]

    def _enhance_prompt_for_consistency(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Enhance the prompt and settings to maintain visual consistency.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Tuple of (enhanced_prompt, scene_settings)
        """
        # Analyze scene type first
        scene_type = self._analyze_scene(prompt)
        
        # Get scene-specific settings
        scene_settings = self.scene_settings.get(scene_type, self.scene_settings["landscape"])
        
        # Start with base style prompt modifications
        base_style = self.style_consistency["base_style"]
        enhanced_prompt = f"{base_style['prompt_prefix']}{prompt}{base_style['prompt_suffix']}"
        
        # Add character consistency for recurring characters
        entities = self._extract_entities_from_prompt(prompt)
        for char_desc in entities["characters"]:
            potential_name = char_desc.split()[0].lower() if char_desc and char_desc.split() else ""
            if potential_name in self.scene_context["character_descriptions"]:
                # Ensure character appearance consistency
                prev_desc = self.scene_context["character_descriptions"][potential_name]
                # Only add if not completely in the prompt already
                if prev_desc.lower() not in prompt.lower():
                    enhanced_prompt += f", {potential_name} consistent with previous appearances"
        
        # Add color palette consistency
        active_palettes = []
        for palette_name, palette_desc in self.scene_context["color_schemes"].items():
            # Only add palettes relevant to the current scene
            if any(word in prompt.lower() for word in palette_name.split("_")):
                enhanced_prompt += f", {palette_desc}"
                active_palettes.append(palette_name)
                
        # Add environment consistency
        for env_key, env_desc in self.scene_context["environment_details"].items():
            if env_key in prompt.lower():
                # Only add environment details if not completely in the prompt already
                if env_desc.lower() not in prompt.lower():
                    enhanced_prompt += f", {env_desc}"
                    
        # Reference established elements for consistency
        relevant_elements = [elem for elem in self.scene_context["established_elements"] 
                           if any(word in prompt.lower() for word in elem.split()[:2])]
        if relevant_elements:
            enhanced_prompt += f", maintaining consistent appearance of {', '.join(relevant_elements[:3])}"
            
        # Enhance negative prompt in settings
        if "negative_prompt" in scene_settings:
            negative_prompt = scene_settings["negative_prompt"]
            if self.scene_context["previous_scenes"]:
                negative_prompt += ", inconsistent lighting, style shift, inconsistent character design"
                # Add active color palettes to avoid
                other_palettes = [p for p in self.style_consistency["color_palette"].keys() 
                                if p not in active_palettes]
                if other_palettes and active_palettes:
                    avoid_colors = ", ".join(self.style_consistency["color_palette"][p].split(",")[0] 
                                          for p in other_palettes)
                    negative_prompt += f", {avoid_colors}"
            scene_settings["negative_prompt"] = negative_prompt
        
        # Return the enhanced prompt and scene settings
        return enhanced_prompt, scene_settings

    @retry_media_generation
    def generate_media(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate images using SDXL model.
        
        Args:
            prompt: Text prompt for image generation
            config: Optional configuration parameters including:
                - width: Image width (default: 1024)
                - height: Image height (default: 576)
                - num_outputs: Number of images to generate (default: 1)
                - guidance_scale: Guidance scale (default: 7.5)
                - num_inference_steps: Number of inference steps (default: 50)
                - seed: Random seed (default: random)
                - negative_prompt: Text to avoid in generation (default: "")
                - scene_type: Use preset settings for "landscape", "portrait", "short", "square", or "abstract"
                - aspect_ratio: Use preset settings for "9:16", "1:1", or "16:9"
                - image: URI for input image (img2img or inpainting mode)
                - mask: URI for input mask (inpainting mode only)
                - prompt_strength: Strength of prompt when using img2img/inpainting (default: 0.8)
                - refine: Refiner to use ("no_refiner", "expert_ensemble_refiner", "base_image_refiner")
                - refine_steps: Number of steps for base_image_refiner
                - high_noise_frac: Fraction of noise for expert_ensemble_refiner
                - scheduler: Scheduler to use (default: "K_EULER")
                - lora_scale: Scale for LoRA models (default: 0.6)
                - apply_watermark: Whether to apply watermark (default: True)
                - disable_safety_checker: Whether to disable safety checker (default: False)
                - maintain_context: Maintain visual context with previous scenes (default: True)
                - chain_images: Use last generated image as input for img2img (default: False)
                
        Returns:
            List of paths to generated image files
        """
        config = config or {}
        
        # Chain images if requested (use last image for img2img)
        if config.get("chain_images", False) and self.scene_context["last_image_path"]:
            # Only set if not already explicitly set
            if "image" not in config:
                config["image"] = self.scene_context["last_image_path"]
                logger.info(f"Chaining from previous image: {config['image']}")
                
                # Set a default prompt_strength if not specified for smooth transitions
                if "prompt_strength" not in config:
                    config["prompt_strength"] = 0.6  # Gentler transition between scenes
        
        # Map aspect_ratio to scene_type if scene_type is not specified
        if "scene_type" not in config and "aspect_ratio" in config:
            aspect_ratio = config["aspect_ratio"]
            if aspect_ratio == "9:16":
                config["scene_type"] = "short"
            elif aspect_ratio == "1:1":
                config["scene_type"] = "square"
            elif aspect_ratio == "16:9":
                config["scene_type"] = "landscape"
        
        # Apply scene-specific settings if provided
        scene_type = config.get("scene_type")
        if scene_type and scene_type in self.scene_settings:
            # Start with scene-specific settings
            scene_config = self.scene_settings[scene_type].copy()
            # Override with any explicitly provided settings
            scene_config.update({k: v for k, v in config.items() if k != "scene_type"})
            config = scene_config
            
        # Apply style consistency if enabled
        maintain_context = config.get("maintain_context", True)
        if maintain_context and self.scene_context["current_settings"]:
            logger.info("Maintaining style consistency with previous scenes")
            
            # Enhance prompt for consistency if we have previous context
            if self.scene_context["previous_scenes"]:
                enhanced_prompt, enhanced_settings = self._enhance_prompt_for_consistency(prompt)
                prompt = enhanced_prompt
                
                # Only override settings if not explicitly specified
                for k, v in enhanced_settings.items():
                    if k not in config:
                        config[k] = v
        
        # Set up parameters with defaults
        params = {
            "prompt": prompt,
            "width": config.get("width", 1024),
            "height": config.get("height", 576),
            "num_outputs": config.get("num_outputs", 1),
            "guidance_scale": config.get("guidance_scale", 7.5),
            "num_inference_steps": config.get("num_inference_steps", 50),
            "negative_prompt": config.get("negative_prompt", ""),
            "scheduler": config.get("scheduler", "K_EULER"),
        }
        
        # Add refiners if specified
        refiner = config.get("refine", "no_refiner")
        if refiner in self.refiner_options and self.refiner_options[refiner]["enabled"]:
            params["refine"] = refiner
            
            if refiner == "expert_ensemble_refiner":
                params["high_noise_frac"] = config.get("high_noise_frac", 
                                                     self.refiner_options[refiner]["high_noise_frac"])
            elif refiner == "base_image_refiner":
                if config.get("refine_steps"):
                    params["refine_steps"] = config.get("refine_steps")
                    
        # Add img2img or inpainting parameters if provided
        if "image" in config:
            params["image"] = config["image"]
            params["prompt_strength"] = config.get("prompt_strength", 0.8)
            
            # If mask is provided, we're doing inpainting
            if "mask" in config:
                params["mask"] = config["mask"]
                
        # Add seed if provided (for reproducibility)
        if "seed" in config:
            params["seed"] = config["seed"]
            
        # Add LoRA scale if provided
        if "lora_scale" in config:
            params["lora_scale"] = config["lora_scale"]
            
        # Add watermark setting if provided
        if "apply_watermark" in config:
            params["apply_watermark"] = config["apply_watermark"]
            
        # Add safety checker setting if provided
        if "disable_safety_checker" in config:
            params["disable_safety_checker"] = config["disable_safety_checker"]
            
        # Generate a unique identifier for this request
        request_id = str(uuid.uuid4())
        
        # In dev mode, try to use cached images
        if self.dev_mode:
            def generate_images_func():
                logger.warning("Dev mode enabled but no cached images found, attempting generation...")
                return self._call_replicate_api(params, request_id)
                
            # Try to get from cache or generate if needed
            file_paths, from_cache = self.cache_handler.get_or_add_cached_media(
                params=params,
                core_params=self.core_cache_params,
                generator_func=generate_images_func,
                result_key='file_paths'
            )
            
            # Update scene context if this was successful
            if file_paths:
                # Use the first image path for context updates
                first_image = file_paths[0] if file_paths else None
                self._update_scene_context(params, first_image)
            
            return file_paths
            
        # Production mode - check cache first
        def generate_images_func():
            return self._call_replicate_api(params, request_id)
            
        # Try to get from cache or generate if needed
        file_paths, from_cache = self.cache_handler.get_or_add_cached_media(
            params=params,
            core_params=self.core_cache_params,
            generator_func=generate_images_func,
            result_key='file_paths'
        )
        
        # Update scene context if this was successful
        if file_paths:
            # Use the first image path for context updates
            first_image = file_paths[0] if file_paths else None
            self._update_scene_context(params, first_image)
        
        return file_paths
            
    def _call_replicate_api(self, params: Dict[str, Any], request_id: str) -> List[str]:
        """Call the Replicate API to generate images.
        
        Args:
            params: Model parameters
            request_id: Unique ID for this request
            
        Returns:
            List of paths to downloaded image files
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
            image_urls = output if isinstance(output, list) else [output]
            
            # Check if we have valid URLs
            if not image_urls or not all(isinstance(url, str) for url in image_urls):
                raise ImageGenerationError(f"Invalid response from Replicate: {output}")
                
            # Download the images
            image_paths = []
            for i, url in enumerate(image_urls):
                # Use request_id plus index for multiple images
                file_id = f"{request_id}_{i}" if len(image_urls) > 1 else request_id
                try:
                    image_path = download_image(url, self.DOWNLOAD_DIR, file_id)
                    image_paths.append(image_path)
                except Exception as e:
                    logger.error(f"Failed to download image {i} from {url}: {e}")
                    raise ImageGenerationError(f"Image download failed: {str(e)}")
            
            logger.info(f"Successfully generated and downloaded {len(image_paths)} images")
            
            return [str(path) for path in image_paths]
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise ImageGenerationError(f"Image generation failed: {str(e)}")

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
            raise ImageGenerationError(f"Failed to download media: {e}")
    
    def get_media_dimensions(self, path: str) -> Dict[str, int]:
        """
        Get the dimensions of the media.
        
        Args:
            path: Path to the media file
            
        Returns:
            Dictionary containing width and height of the media
        """
        try:
            with Image.open(path) as img:
                width, height = img.size
                return {"width": width, "height": height}
        except Exception as e:
            logger.error(f"Failed to get image dimensions: {e}")
            raise ImageGenerationError(f"Failed to get image dimensions: {e}")


# Example usage:
"""
client = SDXLClient()
images = client.generate_media(
    prompt="A serene landscape at sunset",
    config={
        "width": 1024,
        "height": 768,
        "num_outputs": 2,
        "guidance_scale": 7.5,
        "num_inference_steps": 50
    }
)
print(f"Generated images: {images}")
""" 