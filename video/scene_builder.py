"""
Scene Builder for coordinating the generation of individual scenes.
"""

import os
import time
import json
import logging
import tempfile
import numpy as np
import uuid as uuid_lib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip, VideoClip, concatenate_audioclips
from PIL import Image

from core.error_handling import SceneBuilderError, retry_scene_building, MediaGenerationError
from core.utils import ensure_directory_exists
from core.config import Config
from core.memory_utils import FrameGenerator
from audio.tts import TextToSpeech
from video.captions import create_caption_clip, add_karaoke_captions_to_video, overlay_caption_on_video
from media.client_base import MediaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneBuilder:
    def __init__(
        self,
        media_client: MediaClient,
        tts_client: Optional[TextToSpeech] = None,
        media_type: str = "video",  # Can be "video" or "image"
        use_smart_voice: bool = True  # Enable smart voice by default
    ):
        """
        Initialize the scene builder with media client and optional TTS client
        
        Args:
            media_client: MediaClient instance for generating media (video/images)
            tts_client: Optional TextToSpeech instance. If not provided, creates a new one.
            media_type: Type of media to generate ("video" or "image")
            use_smart_voice: Whether to use smart voice optimization
        """
        self.media_client = media_client
        self.tts = tts_client or TextToSpeech(use_smart_voice=use_smart_voice)
        self.media_type = media_type
        self.use_smart_voice = use_smart_voice
        
        # Load config for output directories
        self.config = Config()
        self.output_dir = self.config.temp_output_dir
        ensure_directory_exists(self.output_dir)
        
        logger.info(f"SceneBuilder initialized with media_type={media_type}, use_smart_voice={use_smart_voice}")
    
    @retry_scene_building
    def build_scene(
        self,
        visual_text: str,
        narration_text: str,
        style_config: Optional[Dict] = None,
        continue_from_scene: Optional[Dict] = None,
        video_format: str = "landscape",
        animation_style: str = "ken_burns",
        **kwargs
    ) -> Dict:
        """
        Builds a complete scene by orchestrating media generation, TTS, and captions.
        Now supports split narrations marked with <SPLIT> delimiter.
        
        Args:
            visual_text: The text prompt for media generation
            narration_text: The text for TTS and captions (may contain <SPLIT> for multiple parts)
            style_config: Configuration for media and caption styling
            continue_from_scene: Optional scene to continue from (for video mode)
            video_format: Format/aspect ratio of the video ("landscape", "short", "square")
            animation_style: Animation style for image mode
            **kwargs: Additional style parameters
                - font: Font for captions
                - font_size: Font size for captions
                - color: Text color for captions
                - highlight_color: Color for highlighted words in karaoke captions
                - visible_lines: Number of lines to show at once in karaoke captions (default: 2)
                - bottom_padding: Padding from bottom of screen for captions (default: 80)
                
        Returns:
            Dict containing paths to generated assets and caption clips:
            {
                "media_path": str,
                "media_paths": list[str],
                "audio_paths": list[str],
                "audio_path": str,
                "caption_clips": list[MoviePy clip objects],  # One per narration part
                "duration": float,
                "media_id": str,
                "final_clip": VideoFileClip,
                "video_format": str
            }
        """
        # Input validation
        if not visual_text or not visual_text.strip():
            raise ValueError("Visual text prompt cannot be empty")
        if not narration_text or not narration_text.strip():
            raise ValueError("Narration text cannot be empty")
            
        # Set default style config if none provided
        if style_config is None:
            style_config = {
                "aspect_ratio": "16:9",
                "duration": 4.0,
                "font": "Arial-Bold",
                "font_size": 18,
                "color": "white",
                "image_selection_mode": "first",
                "animation_style": "ken_burns"
            }
            
        logger.info(f"Building scene - Visual: {visual_text}")
        logger.info(f"Narration: {narration_text}")
        
        try:
            # Check if narration needs to be split
            narration_parts = narration_text.split("<SPLIT>") if "<SPLIT>" in narration_text else [narration_text]
            logger.info(f"Processing {len(narration_parts)} narration part(s)")
            
            # Step 1: Generate TTS audio for each part
            audio_paths = []
            audio_durations = []
            total_duration = 0
            
            # Store word timings for all parts
            all_word_timings = []
            combined_duration = 0
            
            # Store TTS results for reuse
            tts_results = {}
            
            for i, part in enumerate(narration_parts, 1):
                part_text = part.strip()
                logger.info(f"Generating voiceover for part {i}: '{part_text[:50]}...'")
                
                # Use smart voice optimization based on narration content
                tts_result = self.tts.generate_speech(
                    part_text,
                    use_smart_voice=self.use_smart_voice
                )
                
                # Store the result for potential reuse
                tts_results[part_text] = tts_result
                
                audio_path = tts_result["path"]
                audio_paths.append(audio_path)
                
                # Verify audio file
                if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
                    raise MediaGenerationError(f"Generated audio file for part {i} is invalid or empty")
                
                # Get audio duration
                with AudioFileClip(audio_path) as audio:
                    duration = audio.duration
                    audio_durations.append(duration)
                    total_duration += duration
                logger.info(f"Part {i} duration: {duration:.2f} seconds")
                
                # Adjust word timings to account for previous parts
                if "word_timings" in tts_result:
                    adjusted_timings = []
                    for word, start, end in tts_result["word_timings"]:
                        adjusted_timings.append((word, start + combined_duration, end + combined_duration))
                    all_word_timings.extend(adjusted_timings)
                    combined_duration += duration
            
            # Step 2: Generate media
            logger.info("Generating media...")
            
            # Set dimensions based on aspect ratio
            aspect_ratio = style_config.get("aspect_ratio", "16:9")
            
            # Use YouTube-optimized resolutions by default
            youtube_optimized = style_config.get("youtube_optimized", True)
            
            if youtube_optimized:
                # Use YouTube-optimized resolutions
                if aspect_ratio == "16:9":
                    width, height = 1920, 1080  # Full HD for landscape YouTube videos
                elif aspect_ratio == "9:16":  # Short format
                    width, height = 1080, 1920  # Full HD for YouTube Shorts
                elif aspect_ratio == "1:1":   # Square format
                    width, height = 1080, 1080  # 1080x1080 for square videos
                else:
                    width, height = 1920, 1080  # Default to 16:9 Full HD
                
                logger.info(f"Using YouTube-optimized resolution: {width}x{height}")
            else:
                # Use 720p resolution for non-optimized mode to ensure YouTube 720p quality
                if aspect_ratio == "16:9":
                    width, height = 1280, 720  # HD Ready (720p)
                elif aspect_ratio == "9:16":  # Short format
                    width, height = 720, 1280  # Vertical HD
                elif aspect_ratio == "1:1":   # Square format
                    width, height = 720, 720  # Square HD
                else:
                    width, height = 1280, 720  # Default to 16:9 HD
                
                logger.info(f"Using HD resolution (720p): {width}x{height}")
            
            media_config = {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "youtube_optimized": youtube_optimized
            }
            
            if self.media_type == "video":
                if continue_from_scene:
                    media_config["start_video_id"] = continue_from_scene.get("media_id")
                media_config["loop"] = False
            
            media_paths = self.media_client.generate_media(visual_text, config=media_config)
            if not media_paths:
                raise MediaGenerationError("No media generated")
            
            # Handle multiple images if in image mode
            if self.media_type == "image" and len(media_paths) > 1:
                selection_mode = style_config.get("image_selection_mode", "first")
                
                if selection_mode == "first":
                    # Simply use the first image
                    logger.info(f"Multiple images generated, using first of {len(media_paths)}")
                    media_path = media_paths[0]
                    
                elif selection_mode == "slideshow":
                    # Create a slideshow from all images
                    logger.info(f"Creating slideshow from {len(media_paths)} images")
                    from moviepy.editor import ImageSequenceClip
                    import shutil
                    
                    # Create a temporary directory for the slideshow
                    slideshow_dir = self.output_dir / "slideshows" / str(uuid_lib.uuid4())
                    slideshow_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all images to the slideshow directory
                    slideshow_images = []
                    for i, img_path in enumerate(media_paths):
                        new_path = slideshow_dir / f"slide_{i}.png"
                        shutil.copy2(img_path, new_path)
                        slideshow_images.append(str(new_path))
                    
                    # Create a video from the images
                    media_path = str(slideshow_dir / "slideshow.mp4")
                    slideshow = ImageSequenceClip(
                        slideshow_images,
                        durations=[total_duration / len(media_paths)] * len(media_paths)
                    )
                    # Write video file
                    slideshow.write_videofile(
                        media_path,
                        fps=24,
                        audio=False
                    )
                    slideshow.close()
                    
                    # Update media type since we converted to video
                    self.media_type = "video"
                else:
                    # Default to first image if mode is unknown
                    logger.warning(f"Unknown image selection mode: {selection_mode}, using first image")
                    media_path = media_paths[0]
            else:
                media_path = media_paths[0]
            
            # Step 3: Create base clip (either video or animated image)
            if self.media_type == "image":
                logger.info("Creating animated clip from image...")
                base_clip = animate_image_clip(
                    media_path,
                    duration=total_duration,  # Use total duration of all parts
                    animation_style=animation_style
                )
                video_path = media_path
            else:
                logger.info("Loading video clip...")
                base_clip = VideoFileClip(str(media_path))
                video_path = media_path
            
            # Step 4: Create caption clips for each part
            logger.info("Creating captions...")
            font_size = style_config.get("font_size", 18)  # Default to 24 if not specified
            logger.info(f"Using font size: {font_size}")
            
            # We're now using karaoke captions for all formats, so we don't need to create standard captions
            caption_clips = []
            current_time = 0
            
            # Step 5: Combine everything into final clip
            logger.info("Assembling final clip...")
            
            # Concatenate audio clips
            audio_clips = [AudioFileClip(path) for path in audio_paths]
            final_audio = concatenate_audioclips(audio_clips)
            
            # Create composite with all caption parts (if any)
            if caption_clips:
                final_clip = CompositeVideoClip(
                    [base_clip] + caption_clips,
                    size=base_clip.size
                )
            else:
                final_clip = base_clip
            
            # Set audio
            final_clip = final_clip.set_audio(final_audio)
            
            # Add fade effects
            final_clip = final_clip.fadein(0.5).fadeout(0.5)
            
            # Generate a unique ID for this media segment
            media_id = str(uuid_lib.uuid4())
            
            # Clean up audio clips
            for clip in audio_clips:
                clip.close()
            
            # Apply karaoke captions for all video formats
            use_karaoke_captions = True
            
            # Generate narration audio if narration text is provided
            narration_audio = None
            narration_data = None
            if narration_text:
                try:
                    # Instead of regenerating the entire narration, use the already generated parts
                    # Create a combined audio file if needed
                    if len(audio_paths) > 1:
                        logger.info("Combining audio parts for full narration...")
                        audio_clips = [AudioFileClip(path) for path in audio_paths]
                        combined_audio = concatenate_audioclips(audio_clips)
                        
                        # Generate a unique filename for the combined audio
                        combined_filename = f"voiceover_combined_{uuid_lib.uuid4()}.mp3"
                        combined_path = self.output_dir / combined_filename
                        combined_audio.write_audiofile(str(combined_path))
                        
                        # Clean up individual audio clips
                        for clip in audio_clips:
                            clip.close()
                        
                        narration_audio = AudioFileClip(str(combined_path))
                        
                        # Create narration data structure similar to what generate_speech would return
                        narration_data = {
                            "path": str(combined_path),
                            "word_timings": all_word_timings,
                            "duration": combined_duration
                        }
                    else:
                        # If there's only one part, use it directly
                        narration_audio = AudioFileClip(audio_paths[0])
                        
                        # Reuse the first TTS result instead of regenerating it
                        first_part = narration_parts[0].strip()
                        if first_part in tts_results:
                            logger.info(f"Reusing existing TTS result for single part")
                            tts_result = tts_results[first_part]
                        else:
                            # This should never happen since we already generated all parts
                            logger.warning(f"TTS result not found for part, regenerating")
                            tts_result = self.tts.generate_speech(
                                first_part,
                                use_smart_voice=self.use_smart_voice
                            )
                        
                        narration_data = {
                            "path": tts_result["path"],
                            "word_timings": tts_result.get("word_timings", []),
                            "duration": audio_durations[0]
                        }
                    
                    logger.info(f"Narration prepared: {narration_data['path']}")
                except Exception as e:
                    logger.error(f"Failed to prepare narration: {e}")
                    raise SceneBuilderError(f"Failed to prepare narration: {e}")
            
            # Apply captions if narration is available
            if narration_text and narration_data:
                if use_karaoke_captions:
                    if "word_timings" in narration_data and narration_data["word_timings"]:
                        logger.info("Applying TikTok-style karaoke captions")
                        # Create style dictionary for karaoke captions
                        caption_style = {
                            "font": kwargs.get("font", "Arial-Bold"),
                            "font_size": kwargs.get("font_size", 60),
                            "color": kwargs.get("color", "white"),
                            "highlight_color": kwargs.get("highlight_color", "#ff5c5c"),
                            "highlight_bg_color": kwargs.get("highlight_bg_color", "white"),
                            "stroke_color": kwargs.get("stroke_color", "black"),
                            "stroke_width": kwargs.get("stroke_width", 2),
                            "bg_color": kwargs.get("bg_color", "rgba(0,0,0,0.5)"),
                            "use_background": kwargs.get("use_background", False),
                            "highlight_use_box": kwargs.get("highlight_use_box", False),
                            "visible_lines": kwargs.get("visible_lines", 2),  # Number of lines to show at once
                            "bottom_padding": kwargs.get("bottom_padding", 80),  # Padding from bottom of screen
                            "tts_provider": getattr(self.tts, "provider_name", "unrealspeech")  # Pass TTS provider name
                        }
                        
                        # Apply karaoke captions
                        final_clip = add_karaoke_captions_to_video(
                            video=final_clip,
                            word_timings=narration_data["word_timings"],
                            style=caption_style
                        )
                        
                        # Set the duration of the final clip to match the audio duration
                        if "duration" in narration_data:
                            final_clip = final_clip.set_duration(narration_data["duration"])
                        else:
                            # If duration is not in narration_data, use the audio clip duration
                            audio_clip = AudioFileClip(narration_data["path"])
                            final_clip = final_clip.set_duration(audio_clip.duration)
                            audio_clip.close()
                    else:
                        logger.warning("Word timings not available for karaoke captions, falling back to standard captions")
                        # Clean <SPLIT> markers from narration text
                        clean_text = narration_text.replace("<SPLIT>", " ").strip()
                        caption_clip = create_caption_clip(
                            text=clean_text,
                            duration=final_clip.duration,
                            video_height=final_clip.h,
                            video_width=final_clip.w,
                            font=kwargs.get("font", "Arial-Bold"),
                            font_size=kwargs.get("font_size", 60),
                            color=kwargs.get("color", "white"),
                            position="bottom"
                        )
                        
                        # Overlay caption on video
                        final_clip = overlay_caption_on_video(final_clip, caption_clip)
                        
                        # Set the duration of the final clip to match the audio duration
                        if "duration" in narration_data:
                            final_clip = final_clip.set_duration(narration_data["duration"])
                        else:
                            # If duration is not in narration_data, use the audio clip duration
                            audio_clip = AudioFileClip(narration_data["path"])
                            final_clip = final_clip.set_duration(audio_clip.duration)
                            audio_clip.close()
                else:
                    logger.info("Applying standard captions")
                    # Clean <SPLIT> markers from narration text
                    clean_text = narration_text.replace("<SPLIT>", " ").strip()
                    caption_clip = create_caption_clip(
                        text=clean_text,
                        duration=final_clip.duration,
                        video_height=final_clip.h,
                        video_width=final_clip.w,
                        font=kwargs.get("font", "Arial"),
                        font_size=kwargs.get("font_size", 24),
                        color=kwargs.get("color", "white"),
                        position="bottom"
                    )
                    
                    # Overlay caption on video
                    final_clip = overlay_caption_on_video(final_clip, caption_clip)
                    
                    # Set the duration of the final clip to match the audio duration
                    if "duration" in narration_data:
                        final_clip = final_clip.set_duration(narration_data["duration"])
                    else:
                        # If duration is not in narration_data, use the audio clip duration
                        audio_clip = AudioFileClip(narration_data["path"])
                        final_clip = final_clip.set_duration(audio_clip.duration)
                        audio_clip.close()
            
            return {
                "media_path": video_path,
                "media_paths": media_paths,
                "audio_paths": audio_paths,
                "audio_path": audio_paths[0] if audio_paths else None,  # For backward compatibility
                "caption_clips": caption_clips,
                "duration": total_duration,
                "media_id": media_id,
                "final_clip": final_clip,
                "video_format": video_format
            }
            
        except Exception as e:
            logger.error(f"Error building scene: {e}")
            raise SceneBuilderError(f"Failed to build scene: {e}") from e
            
    def verify_output(self, scene: Dict) -> bool:
        """
        Verify the output of a scene build
        
        Args:
            scene: Dictionary containing scene components
            
        Returns:
            bool: True if all components are valid
        """
        try:
            # Check media file
            media_path = Path(scene["media_path"])
            if not media_path.exists() or media_path.stat().st_size == 0:
                logger.error("Media file is invalid or empty")
                return False
                
            # Check audio files
            if "audio_paths" in scene:
                for audio_path in scene["audio_paths"]:
                    path = Path(audio_path)
                    if not path.exists() or path.stat().st_size == 0:
                        logger.error(f"Audio file {audio_path} is invalid or empty")
                        return False
            else:
                # Backward compatibility check
                audio_path = Path(scene["audio_path"])
                if not audio_path.exists() or audio_path.stat().st_size == 0:
                    logger.error("Audio file is invalid or empty")
                    return False
                
            # Check caption clips - skip this check for videos using karaoke captions
            logger.info("Skipping caption clips check (using karaoke captions)")
            
            # Test overlay to ensure compatibility - skip for videos using karaoke captions
            logger.info("Skipping caption overlay test (using karaoke captions)")
            
            return True
            
        except Exception as e:
            logger.error(f"Scene verification failed: {str(e)}")
            return False 

def animate_image_clip(
    image_path: Union[str, Path],
    duration: float,
    animation_style: str = "ken_burns",
    zoom_range: Tuple[float, float] = (1.0, 1.3),
    pan_range: Tuple[float, float] = (-0.1, 0.1)
) -> VideoFileClip:
    """
    Create an animated video clip from a static image using memory-efficient generator.
    
    Args:
        image_path: Path to the image file
        duration: Duration of the animation in seconds
        animation_style: Type of animation effect
        zoom_range: Min and max zoom values
        pan_range: Min and max pan offset values
        
    Returns:
        Animated video clip
    """
    logger.info(f"Creating {animation_style} animation for image: {image_path}")
    
    # Create a frame generator instead of generating all frames in memory at once
    frame_gen = FrameGenerator(
        image_path=image_path,
        duration=duration,
        fps=24,  # Standard frame rate
        animation_style=animation_style,
        zoom_range=zoom_range,
        pan_range=pan_range
    )
    
    # Create a custom clip using the frame generator's get_frame_at_time method
    clip = VideoClip(lambda t: frame_gen.get_frame_at_time(t), duration=duration)
    
    # We don't need to add fade effects here as they're handled in the generator
    logger.info(f"Created memory-efficient animated clip with duration={duration:.2f}s")
    
    return clip 