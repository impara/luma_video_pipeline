"""
Centralized cache handling for media generation clients.
Provides a common interface for caching and retrieving media files.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheHandler:
    """Centralized cache handling for media generation clients."""
    
    def __init__(self, cache_dir: Path, cache_file: str, dev_mode: bool = False):
        """Initialize the cache handler.
        
        Args:
            cache_dir: Directory to store cached files
            cache_file: Name of the cache file
            dev_mode: Whether to operate in development mode
        """
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / cache_file
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir_abs = self.cache_dir.resolve()
        self.dev_mode = dev_mode
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load the cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save the cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_cache_key(self, params: Dict[str, Any], core_params: List[str]) -> str:
        """Generate a cache key from the input parameters.
        
        Args:
            params: Parameters to generate cache key from
            core_params: List of core parameter names to include in the key
            
        Returns:
            String cache key
        """
        # Normalize prompt (strip whitespace, lowercase)
        prompt = params.get("prompt", "").strip().lower()
        
        # Create key parts with core parameters
        key_parts = [prompt]
        for param in core_params:
            if param in params and param != "prompt":  # Avoid duplicating prompt
                key_parts.append(str(params.get(param)))
        
        # Join all parts and hash
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_abs_path(self, rel_path: Union[str, Path]) -> Path:
        """Convert a relative path to absolute path."""
        path = Path(rel_path)
        if path.is_absolute():
            return path
        # Handle case where cache_dir prefix is missing
        if self.cache_dir.name not in path.parts:
            path = self.cache_dir / path
        return path.resolve()
    
    def get_rel_path(self, abs_path: Union[str, Path]) -> Path:
        """Convert an absolute path to relative path from cache_dir."""
        path = Path(abs_path)
        try:
            return path.relative_to(self.cache_dir_abs)
        except ValueError:
            # If path is already relative to cache_dir, ensure it has the prefix
            if self.cache_dir.name not in path.parts:
                return self.cache_dir / path
            return path
    
    def get_cached_item(self, params: Dict[str, Any], core_params: List[str], 
                         item_key: str, check_exact: bool = True) -> Optional[Union[str, List[str]]]:
        """Check if an item exists in cache for given parameters.
        
        Args:
            params: Parameters to check against
            core_params: List of core parameter names for cache key generation
            item_key: Key to look up in the cache entry (e.g., 'file_path', 'file_paths')
            check_exact: Whether to check for exact match of all params vs just core params
            
        Returns:
            Cached item if found, None otherwise
        """
        cache_key = self.get_cache_key(params, core_params)
        
        # Check if entry exists in cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            
            # If we need exact match, verify all params match
            if check_exact:
                # Check if all parameters match (ignoring non-deterministic ones)
                for k, v in params.items():
                    if k in cache_entry.get('params', {}) and str(cache_entry['params'][k]) != str(v):
                        if k not in ['seed', 'uuid', 'id']:  # Ignore non-deterministic params
                            logger.info(f"Parameter mismatch for {k}: {cache_entry['params'][k]} != {v}")
                            return None
            
            # Return the requested item if it exists
            if item_key in cache_entry:
                if isinstance(cache_entry[item_key], list):
                    # Convert all paths to absolute paths
                    return [str(self.get_abs_path(p)) for p in cache_entry[item_key]]
                else:
                    # Convert to absolute path
                    return str(self.get_abs_path(cache_entry[item_key]))
        
        return None
    
    def add_to_cache(self, params: Dict[str, Any], core_params: List[str], 
                     result_items: Dict[str, Any]) -> str:
        """Add an item to the cache.
        
        Args:
            params: Parameters that were used to generate the item
            core_params: List of core parameter names for cache key generation
            result_items: Dictionary of items to store (e.g., {'file_path': path, 'file_paths': [paths]})
            
        Returns:
            Cache key that was used
        """
        cache_key = self.get_cache_key(params, core_params)
        
        # Convert all paths to relative paths for storage
        processed_items = {}
        for k, v in result_items.items():
            if isinstance(v, list) and all(isinstance(x, (str, Path)) for x in v):
                processed_items[k] = [str(self.get_rel_path(p)) for p in v]
            elif isinstance(v, (str, Path)):
                processed_items[k] = str(self.get_rel_path(v))
            else:
                processed_items[k] = v
        
        # Create cache entry
        cache_entry = {
            'params': params,
            **processed_items
        }
        
        # Add to cache and save
        self.cache[cache_key] = cache_entry
        self._save_cache()
        
        return cache_key
    
    def get_or_add_cached_media(self, 
                               params: Dict[str, Any], 
                               core_params: List[str], 
                               generator_func, 
                               result_key: str = 'file_paths') -> Tuple[List[str], bool]:
        """Get media from cache or generate it and add to cache if not found.
        
        This method standardizes the pattern of checking cache, generating if needed, and caching results.
        
        Args:
            params: Parameters for media generation
            core_params: Core parameters for cache key generation
            generator_func: Function to call if item not in cache (should return paths or list of paths)
            result_key: Key to use when storing result in cache
            
        Returns:
            Tuple of (file_paths, from_cache) where from_cache indicates if result was from cache
        """
        # First check if we have it in cache
        cached_paths = self.get_cached_item(params, core_params, result_key)
        
        if cached_paths:
            if self.dev_mode:
                logger.info(f"Using cached media in dev mode: {cached_paths}")
            else:
                logger.info(f"Using cached media: {cached_paths}")
            
            # Convert to list if it's a single path
            if not isinstance(cached_paths, list):
                cached_paths = [cached_paths]
                
            return cached_paths, True
        
        # Not in cache, need to generate
        if self.dev_mode:
            logger.warning("Dev mode enabled but no cached media found. Need actual media generation.")
        
        # Generate the media
        generated_paths = generator_func()
        
        # Convert to list if it's a single path
        if not isinstance(generated_paths, list):
            generated_paths = [generated_paths]
        
        # Add to cache
        self.add_to_cache(params, core_params, {result_key: generated_paths})
        
        return generated_paths, False 