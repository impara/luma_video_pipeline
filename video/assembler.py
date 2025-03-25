"""
Final video assembly module.
Responsible for:
- Combining generated video with voiceover audio
- Ensuring proper synchronization of all components
- Handling final video export with all elements combined
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    VideoFileClip,
    ImageClip,
    concatenate_videoclips,
    concatenate_audioclips,
    ColorClip,
    clips_array,
    transfx
)
from moviepy.video.fx import all as vfx

from core.error_handling import MediaGenerationError
from core.config import Config
from core.utils import ensure_directory_exists

# Configure logging
logger = logging.getLogger(__name__)

class VideoAssembler:
    def __init__(self):
        """Initialize the video assembler"""
        # Load config for output directories
        self.config = Config()
        self.output_dir = self.config.video_output_dir
        ensure_directory_exists(self.output_dir)
        
    def _prepare_output_path(self, output_path: str) -> str:
        """
        Prepare the output path for saving a video.
        
        Args:
            output_path: Path to save the final video
            
        Returns:
            str: Normalized output path with proper extension
        """
        # Check if the output path already starts with output/videos
        if output_path.startswith(str(self.output_dir)):
            # If it's already a properly formatted path, use it as is
            output_path_obj = Path(output_path)
        else:
            # Otherwise, place it in the output directory
            output_path_obj = self.output_dir / Path(output_path).name
            
        # Ensure output path has .mp4 extension
        if not str(output_path_obj).endswith('.mp4'):
            output_path_obj = output_path_obj.with_suffix('.mp4')
            
        # Create parent directories if needed
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        return str(output_path_obj)
        
    def _write_video_file(self, clip, output_path: str, is_high_quality: bool = True) -> None:
        """
        Write video clip to file with optimized parameters.
        
        Args:
            clip: The video clip to write
            output_path: Path to save the video
            is_high_quality: Whether to use higher quality settings (YouTube compatible)
        """
        # Set video and audio encoding parameters based on quality setting
        if is_high_quality:
            # High quality settings (YouTube optimized)
            bitrate = "8000k"        # High quality bitrate
            audio_bitrate = "192k"   # Good audio quality
            format_name = "High Quality (YouTube compatible)"
        else:
            # Standard quality settings
            bitrate = "4000k"        # Standard bitrate
            audio_bitrate = "128k"   # Standard audio quality
            format_name = "Standard Quality"
            
        logger.info(f"Writing video with {format_name} settings to {output_path}")
        logger.info(f"- Video bitrate: {bitrate}")
        logger.info(f"- Audio bitrate: {audio_bitrate}")
            
        # Write the video file with selected settings
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate=bitrate,
            audio_bitrate=audio_bitrate,
            fps=24,
            threads=4,        # Use multiple threads for faster encoding
            preset="medium",  # Balance between speed and quality
            remove_temp=True  # Clean up temporary files
        )

    def _create_scene_clip(self, scene: Dict, index: int) -> CompositeVideoClip:
        """
        Creates a composite clip from a single scene with video, audio, and captions.
        Now supports multiple caption parts per scene.
        
        Args:
            scene: Dictionary containing scene components
            index: Scene index for logging
        
        Returns:
            CompositeVideoClip: The assembled scene clip
        """
        logger.info(f"Rendering scene {index + 1}...")
        
        # Load video/image and audio
        if scene.get("final_clip"):
            # Use the pre-rendered clip if available (from animated images)
            base_clip = scene["final_clip"]
        else:
            # Load from file
            base_clip = VideoFileClip(scene["media_path"])
        
        # Process audio
        if "audio_paths" in scene:
            # Handle multiple audio parts
            audio_clips = [AudioFileClip(path) for path in scene["audio_paths"]]
            audio = concatenate_audioclips(audio_clips)
        else:
            # Backwards compatibility for single audio path
            audio = AudioFileClip(scene["audio_path"])
        
        # Process captions
        caption_clips = scene.get("caption_clips", [])
        
        # Combine base video with caption clips
        components = [base_clip] + caption_clips
        composite = CompositeVideoClip(components, size=base_clip.size)
        
        # Ensure composite duration matches audio
        if composite.duration < audio.duration:
            # Extend video duration to match audio
            composite = composite.set_duration(audio.duration)
        elif composite.duration > audio.duration:
            # Log warning about mismatched durations but don't trim 
            # (let the video assembler handle this at the scene level if needed)
            logger.info(f"Adjusting scene {index + 1} duration from {composite.duration:.2f}s to {audio.duration:.2f}s")
            composite = composite.set_duration(audio.duration)
        
        # Log detailed information about the assembled clip
        logger.info(f"Scene {index + 1} details:")
        logger.info(f"- Video duration: {base_clip.duration:.2f}s")
        logger.info(f"- Audio duration: {audio.duration:.2f}s")
        logger.info(f"- Number of caption parts: {len(caption_clips)}")
        for i, cap in enumerate(caption_clips, 1):
            logger.info(f"  - Caption part {i} duration: {cap.duration:.2f}s")
        logger.info(f"- Final duration: {composite.duration:.2f}s")
        
        # Set audio and return
        return composite.set_audio(audio)

    def assemble_scenes(
        self,
        scenes: List[Dict],
        output_path: str,
        fps: int = 24,  # Changed default to 24 fps to match source videos
        transition_duration: float = 0.5,  # Duration for cross-dissolve transitions
        optimize_for_youtube: bool = True  # New parameter to optimize for YouTube
    ) -> str:
        """
        Assembles all scenes into a final video with proper transitions.
        Now handles scenes with multiple narration parts properly.
        
        Args:
            scenes: List of scene dictionaries with components
            output_path: Path to save the final video
            fps: Frames per second for the video (default: 24)
            transition_duration: Duration of transitions between scenes (default: 0.5s)
            optimize_for_youtube: Whether to use settings optimized for YouTube (default: True)
            
        Returns:
            str: Path to the assembled video file
        """
        try:
            logger.info("Starting video assembly...")
            
            # Prepare output path
            output_path = self._prepare_output_path(output_path)
                
            # Render all scenes
            scene_clips = []
            for i, scene in enumerate(scenes):
                clip = self._create_scene_clip(scene, i)
                scene_clips.append(clip)
                logger.info(f"Scene {i + 1} rendered successfully")
                logger.info(f"Scene duration: {clip.duration:.2f}s")
            
            # Calculate total duration
            total_duration = sum(clip.duration for clip in scene_clips)
            logger.info(f"Total video duration will be: {total_duration:.2f}s")
            
            # Add transitions between scenes if requested
            if len(scene_clips) > 1 and transition_duration > 0:
                logger.info("Adding cross-dissolve transitions between scenes...")
                
                # Create clips with transitions using a cross-dissolve effect
                # In older MoviePy versions, we need to apply transitions manually to clips
                clips_with_transitions = []
                
                for i, clip in enumerate(scene_clips):
                    if i == 0:  # First clip stays as is
                        clips_with_transitions.append(clip)
                    else:
                        # Apply cross dissolve effect to transition from previous clip
                        # We'll keep full clip duration but add the transition effect
                        with_transition = clip.crossfadein(transition_duration)
                        clips_with_transitions.append(with_transition)
                
                # Use 'compose' method to overlap clips at their transition points
                final_clip = concatenate_videoclips(
                    clips_with_transitions,
                    method="compose",
                    padding=-transition_duration
                )
                
                logger.info(f"Added {len(scene_clips)-1} transitions of {transition_duration}s each")
            else:
                # Simple concatenation without transitions
                final_clip = concatenate_videoclips(scene_clips, method="chain")
            
            # Set video and audio encoding parameters (for better quality/compatibility)
            # These settings are optimized for good quality/size balance on YouTube
            if optimize_for_youtube:
                # YouTube recommended settings
                width, height = 1920, 1080  # Full HD
                format_name = "Full HD (1080p)"
            else:
                # Default HD settings
                width, height = 1280, 720   # HD Ready
                format_name = "HD (720p)"
            
            logger.info(f"Optimizing for {format_name} format:")
            logger.info(f"- Video dimensions: {width}x{height}")
            logger.info(f"- FPS: {fps}")
            
            # Check if we need to resize the video
            if final_clip.size != (width, height):
                final_clip = final_clip.resize(height=height)
            
            # Write the final video with optimized settings
            self._write_video_file(final_clip, output_path, optimize_for_youtube)
            
            # Clean up
            for clip in scene_clips:
                clip.close()
            final_clip.close()
            
            logger.info("Video assembly completed successfully!")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during video assembly: {str(e)}")
            raise
        
    def combine_video_and_audio_only(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> str:
        """
        Combines a video with an audio track without modifying the video content.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path where to save the final video
            
        Returns:
            str: Path to the assembled video
        """
        try:
            logger.info("Combining video and audio...")
            
            # Prepare output path
            output_path = self._prepare_output_path(output_path)
                
            # Load video and audio
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Set audio to video
            final_clip = video.set_audio(audio)
            
            # Write the final clip using high quality settings
            logger.info(f"Writing final video to {output_path}...")
            self._write_video_file(final_clip, output_path, is_high_quality=True)
            
            # Clean up
            video.close()
            audio.close()
            final_clip.close()
            
            logger.info("Video assembly completed successfully!")
            return output_path




            
        except Exception as e:
            logger.error(f"Error during video assembly: {str(e)}")
            raise 