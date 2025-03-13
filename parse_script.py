"""
Script parsing utilities for the video generation pipeline.
Handles the two-part format where each scene consists of a visual prompt and narration text.
"""
import logging
import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_two_part_script(script_path: str) -> List[Tuple[str, str, Dict]]:
    """
    Parse a script file where each scene has labeled visual and narration sections.
    Now supports split narrations with Part 1/Part 2 markers.
    
    # Scene 1 (Visual) [continue=1]  # Optional continuation flag
    [Visual prompt for AI video generation]
    
    # Scene 1 (Narration - Part 1)
    [First part of narration text]
    
    # Scene 1 (Narration - Part 2)
    [Second part of narration text]
    
    Args:
        script_path: Path to the script file
        
    Returns:
        List of tuples, each containing (visual_prompt, narration_text, metadata)
        where metadata includes continuation and split narration information
    """
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")
    
    # Read all lines
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Process lines
    scenes = []
    current_visual = None
    current_metadata = {}
    current_narration_parts = []
    current_scene = 0
    
    # Regex to extract continuation flag and part number
    continue_pattern = re.compile(r'\[continue=(\d+)\]')
    part_pattern = re.compile(r'Narration - Part (\d+)')
    
    for line in lines:
        # Skip empty lines
        if not line:
            continue
            
        # Check for scene headers and extract metadata
        if line.startswith('#'):
            if '(Visual)' in line:
                # If we have a complete scene, store it
                if current_visual is not None and current_narration_parts:
                    # Join narration parts with a special marker for splitting later
                    narration = "<SPLIT>".join(current_narration_parts)
                    current_metadata['split_narration'] = len(current_narration_parts) > 1
                    scenes.append((current_visual, narration, current_metadata))
                    current_visual = None
                    current_narration_parts = []
                    current_metadata = {}
                
                # Extract continuation flag if present
                match = continue_pattern.search(line)
                if match:
                    current_metadata = {'continue_from': int(match.group(1))}
                else:
                    current_metadata = {}
                    
            elif '(Narration' in line:
                # Check for part number
                match = part_pattern.search(line)
                if match:
                    current_metadata['has_parts'] = True
            continue
        
        # If we have content, store it
        if line:
            if current_visual is None:
                current_visual = line
            else:
                current_narration_parts.append(line)
    
    # Handle last scene if complete
    if current_visual is not None and current_narration_parts:
        narration = "<SPLIT>" .join(current_narration_parts)
        current_metadata['split_narration'] = len(current_narration_parts) > 1
        scenes.append((current_visual, narration, current_metadata))
    
    # Validate we have complete scenes
    if not scenes:
        raise ValueError("Script file is empty or contains only comments")
    
    # Log the parsed scenes
    logger.info(f"Successfully parsed {len(scenes)} scenes from {script_path}")
    
    return scenes

if __name__ == "__main__":
    # Example usage
    script_content = """
    # Scene 1 (Visual)
    A futuristic cityscape with hovercars and neon lights
    
    # Scene 1 (Narration - Part 1)
    Welcome to the future, where innovation never sleeps
    
    # Scene 1 (Narration - Part 2)
    Behold the beauty of a new dawn, as nature awakens
    
    # Scene 2 (Visual) [continue=1]
    A serene mountain lake at sunrise with mist rolling over the water
    
    # Scene 2 (Narration - Part 1)
    Behold the beauty of a new dawn, as nature awakens
    
    # Scene 2 (Narration - Part 2)
    Behold the beauty of a new dawn, as nature awakens
    """
    
    # Write example to a file
    with open("example_script.txt", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    try:
        scenes = parse_two_part_script("example_script.txt")
        print("\nParsed scenes:")
        for i, (visual, narration, metadata) in enumerate(scenes, 1):
            print(f"\nScene {i}:")
            print(f"Visual: {visual}")
            print(f"Narration: {narration}")
            if metadata.get('continue_from'):
                print(f"Continues from scene: {metadata['continue_from']}")
            if metadata.get('split_narration'):
                print(f"Has split narration: {len(narration.split('<SPLIT>'))} parts")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    finally:
        Path("example_script.txt").unlink(missing_ok=True) 