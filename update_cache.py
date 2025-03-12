"""
Script to update the cache.json to the new format using the improved caching mechanism.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

def generate_new_key(params: Dict[str, Any]) -> str:
    """Generate a cache key using the new mechanism."""
    # Get the prompt and normalize it
    prompt = params.get("prompt", "").strip().lower().rstrip(".,!?")
    
    # Create key parts (only core parameters)
    key_parts = [
        prompt,
        params.get("aspect_ratio", "16:9"),
        str(params.get("loop", False))
    ]
            
    # Join all parts and hash
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()

def clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean parameters to only include core video generation params."""
    return {
        "prompt": params["prompt"].strip(),
        "aspect_ratio": params.get("aspect_ratio", "16:9"),
        "loop": params.get("loop", False)
    }

def update_cache():
    """Update the cache.json to the new format."""
    cache_path = Path("downloads/replicate_segments/cache.json")
    
    if not cache_path.exists():
        print("Cache file not found")
        return
        
    try:
        # Read existing cache
        with open(cache_path, 'r') as f:
            old_cache = json.load(f)
            
        # Create new cache with updated keys
        new_cache = {}
        seen_prompts = set()
        
        # First pass: collect non-continuation videos
        for entry_id, entry_data in old_cache.items():
            params = entry_data["params"]
            
            # Skip continuation videos
            if "start_video_id" in params or "end_video_id" in params:
                continue
                
            prompt = params["prompt"].strip().lower().rstrip(".,!?")
            
            # Skip if we've already processed this prompt
            if prompt in seen_prompts:
                continue
                
            seen_prompts.add(prompt)
            
            # Clean parameters and generate new key
            cleaned_params = clean_params(params)
            new_key = generate_new_key(cleaned_params)
            
            # Add to new cache with cleaned parameters
            new_cache[new_key] = {
                "video_path": entry_data["video_path"],
                "params": cleaned_params
            }
        
        # Backup old cache
        backup_path = cache_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(old_cache, f)
            
        # Write new cache
        with open(cache_path, 'w') as f:
            json.dump(new_cache, f)
            
        print(f"Cache updated successfully!")
        print(f"Old cache backed up to: {backup_path}")
        print(f"Original entries: {len(old_cache)}")
        print(f"New entries (non-continuation only): {len(new_cache)}")
        print(f"Removed {len(old_cache) - len(new_cache)} continuation/duplicate entries")
        
    except Exception as e:
        print(f"Error updating cache: {e}")

if __name__ == "__main__":
    update_cache() 