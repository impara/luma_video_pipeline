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
from core.resource_manager import ResourceManager
from core.memory_utils import MemoryMonitor

# Configure logging
logger = logging.getLogger(__name__)

class VideoAssembler:
    def __init__(self):
        """Initialize the video assembler"""
        # Load config for output directories
        self.config = Config()
        self.output_dir = self.config.video_output_dir
        
        # Create required directories
        self._create_required_directories()
        
        # Initialize memory management
        memory_settings = self.config.get_memory_settings()
        self.window_size = memory_settings['window_size']
        self.memory_monitor = MemoryMonitor(
            max_memory_mb=memory_settings['max_memory_mb'],
            cleanup_threshold=memory_settings['cleanup_threshold']
        )
        self.resource_manager = ResourceManager()
        
        # Initialize encoder detection
        self.tested_encoders = set()
        self.available_encoders = {}
        
    def _create_required_directories(self):
        """Create all required directories for video processing"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'temp'),
            os.path.join(self.output_dir, 'temp', 'processing'),
            os.path.join(self.output_dir, 'temp', 'scenes'),
            os.path.join(self.output_dir, 'temp', 'audio')
        ]
        
        for directory in directories:
            ensure_directory_exists(directory)
            logger.info(f"Ensured directory exists: {directory}")
        
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
        
    def _get_encoding_parameters(self, is_intermediate=True, is_high_quality=True):
        """
        Get optimized encoding parameters based on whether this is an intermediate
        or final file, and desired quality level.
        
        Args:
            is_intermediate: Whether this is an intermediate file (True) or final output (False)
            is_high_quality: Whether high quality encoding is requested
            
        Returns:
            Dict of encoding parameters
        """
        # Get CPU count for optimal thread usage
        cpu_count = os.cpu_count() or 4
        threads = max(1, min(cpu_count - 1, 16))  # Leave one CPU free, max 16 threads
        
        # Base parameters - always use these
        base_params = {
            "codec": "libx264",
            "audio_codec": "aac",
            "fps": 24,
            "threads": threads
        }
        
        # For intermediate files, prioritize speed over quality
        if is_intermediate:
            # Always use fast settings for intermediate files regardless of quality setting
            params = {
                **base_params,
                "audio_bitrate": "128k",
                "ffmpeg_params": [
                    "-preset", "ultrafast" if not is_high_quality else "veryfast",
                    "-crf", "28",
                    "-tune", "fastdecode",
                    "-profile:v", "baseline",
                    "-movflags", "+faststart"
                ]
            }
            logger.debug("Using fast encoding preset for intermediate file")
        else:
            # Final output - use quality settings based on is_high_quality
            if is_high_quality:
                # High quality for final output
                params = {
                    **base_params,
                    "audio_bitrate": "192k",
                    "ffmpeg_params": [
                        "-preset", "medium",
                        "-crf", "18",
                        "-tune", "film",
                        "-profile:v", "high",
                        "-level", "4.2",
                        "-movflags", "+faststart"
                    ]
                }
                logger.info("Using high quality encoding for final output")
            else:
                # Standard quality for final output (faster)
                params = {
                    **base_params,
                    "audio_bitrate": "128k",
                    "ffmpeg_params": [
                        "-preset", "faster",
                        "-crf", "23",
                        "-tune", "film",
                        "-profile:v", "main",
                        "-level", "4.0",
                        "-movflags", "+faststart"
                    ]
                }
                logger.info("Using standard quality encoding for final output")
        
        # Try to use hardware acceleration if available
        if not self.available_encoders and hasattr(self, 'tested_encoders'):
            # Only detect encoders once
            self.available_encoders = self._detect_available_encoders()
            
        # Use hardware acceleration if available
        if self.available_encoders:
            encoder_name = next(iter(self.available_encoders))
            encoder_codec = self.available_encoders[encoder_name]
            
            logger.info(f"Using hardware acceleration: {encoder_name} ({encoder_codec})")
            params["codec"] = encoder_codec
            
            # Adjust parameters for specific hardware encoders
            if encoder_codec == 'h264_nvenc':
                if is_intermediate:
                    # Fast encoding for intermediate files
                    params["ffmpeg_params"] = [
                        "-preset", "p1",     # Speed-focused preset
                        "-rc", "vbr",        # Variable bitrate
                        "-qmin", "1", 
                        "-qmax", "35",       # Lower quality for speed
                        "-b:v", "0"
                    ]
                else:
                    # Quality for final output
                    if is_high_quality:
                        params["ffmpeg_params"] = [
                            "-preset", "p4",     # Quality-focused preset
                            "-rc", "vbr_hq",     # High quality mode
                            "-qmin", "1",
                            "-qmax", "17",       # Higher quality (lower value)
                            "-b:v", "0"
                        ]
                    else:
                        params["ffmpeg_params"] = [
                            "-preset", "p2",     # Balanced preset
                            "-rc", "vbr",
                            "-qmin", "1",
                            "-qmax", "24",       # Medium quality
                            "-b:v", "0"
                        ]
            elif encoder_codec == 'h264_qsv':
                # Intel QuickSync-specific parameters
                params["ffmpeg_params"] = [
                    "-global_quality", "18" if is_high_quality and not is_intermediate else "28",
                    "-movflags", "+faststart"
                ]
            elif encoder_codec == 'h264_amf':
                # AMD-specific parameters
                quality_value = "quality" if is_high_quality and not is_intermediate else "speed"
                params["ffmpeg_params"] = [
                    "-quality", quality_value,
                    "-movflags", "+faststart"
                ]
        
        return params

    def _write_video_file(self, clip, output_path: str, is_intermediate=True, is_high_quality=True, use_two_pass=False) -> None:
        """
        Write video clip to file with optimized parameters based on whether it's an
        intermediate file or final output.
        
        Args:
            clip: The video clip to write
            output_path: Path to save the video
            is_intermediate: Whether this is an intermediate file (not the final output)
            is_high_quality: Whether high quality encoding is requested
            use_two_pass: Whether to use two-pass encoding (only for final files)
        """
        # Get optimized encoding parameters
        video_params = self._get_encoding_parameters(
            is_intermediate=is_intermediate,
            is_high_quality=is_high_quality
        )
        
        # For two-pass encoding (final files only)
        format_name = "Intermediate" if is_intermediate else "Final"
        format_name += " " + ("High Quality" if is_high_quality else "Standard Quality")
        
        if use_two_pass and not is_intermediate:
            logger.info(f"Using two-pass encoding for {format_name} output")
            
            # For two-pass encoding, we need a target bitrate
            # Calculate based on resolution: higher for high quality
            width, height = clip.size
            pixels = width * height
            
            if is_high_quality:
                # High quality bitrate formula (simplified)
                video_bitrate = f"{int(pixels * 0.11 / 1000)}k"  # Approx formula
            else:
                # Standard quality bitrate formula
                video_bitrate = f"{int(pixels * 0.075 / 1000)}k"
            
            logger.info(f"Using target bitrate: {video_bitrate} for two-pass encoding")
            
            # Remove any existing rate control params that might conflict
            conflict_params = ["-crf", "-qmin", "-qmax", "-rc"]
            for param in conflict_params:
                if param in video_params["ffmpeg_params"]:
                    idx = video_params["ffmpeg_params"].index(param)
                    # Remove param and its value
                    if idx < len(video_params["ffmpeg_params"]) - 1:
                        video_params["ffmpeg_params"].pop(idx+1)  # Remove value
                    video_params["ffmpeg_params"].remove(param)  # Remove param
            
            # Set up two-pass encoding
            video_params["bitrate"] = video_bitrate
            video_params["ffmpeg_params"].extend(["-pass", "1"])
        
        # Log encoding parameters for visibility
        logger.info(f"Encoding {format_name} file:")
        logger.info(f"- Output: {output_path}")
        logger.info(f"- Video codec: {video_params['codec']}")
        logger.info(f"- Audio codec: {video_params['audio_codec']} ({video_params['audio_bitrate']})")
        
        if "ffmpeg_params" in video_params and len(video_params["ffmpeg_params"]) > 1:
            preset_idx = video_params["ffmpeg_params"].index("-preset") + 1 if "-preset" in video_params["ffmpeg_params"] else -1
            if preset_idx > 0 and preset_idx < len(video_params["ffmpeg_params"]):
                logger.info(f"- Encoding preset: {video_params['ffmpeg_params'][preset_idx]}")
        
        # Write the video file
        try:
            clip.write_videofile(output_path, **video_params)
        except Exception as e:
            logger.error(f"Failed to write video file: {e}")
            # Fall back to default encoding if hardware acceleration fails
            if video_params['codec'] != 'libx264':
                logger.warning("Hardware encoding failed. Falling back to software encoding (libx264)")
                # Get new parameters with software encoding
                video_params = self._get_encoding_parameters(is_intermediate, is_high_quality)
                video_params['codec'] = 'libx264'  # Force software encoding
                clip.write_videofile(output_path, **video_params)

    def _calculate_optimal_bitrate(self, width: int, height: int, fps: float) -> str:
        """
        Calculate optimal bitrate based on resolution and frame rate.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            
        Returns:
            String representation of bitrate in kbps
        """
        # Base formula: width * height * fps * 0.07 / 1000
        # The 0.07 factor is a quality coefficient that can be adjusted
        
        # Calculate resolution-based bitrate
        resolution = width * height
        
        if resolution >= 1920 * 1080:  # 1080p or higher
            base_bitrate = 8000  # 8Mbps base for 1080p at 30fps
        elif resolution >= 1280 * 720:  # 720p
            base_bitrate = 5000  # 5Mbps base for 720p at 30fps
        elif resolution >= 854 * 480:  # 480p
            base_bitrate = 2500  # 2.5Mbps base for 480p at 30fps
        else:  # lower resolutions
            base_bitrate = 1500  # 1.5Mbps base for low res at 30fps
            
        # Adjust for framerate
        fps_factor = fps / 30.0
        bitrate = int(base_bitrate * fps_factor)
        
        # Ensure bitrate is within reasonable bounds
        bitrate = max(1000, min(bitrate, 20000))
        
        return f"{bitrate}k"  # Return as string with k suffix for kbps

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

    def _process_scene_window(
        self,
        scenes: List[Dict],
        start_idx: int,
        temp_dir: Path,
        **kwargs
    ) -> List[VideoFileClip]:
        """
        Process a window of scenes and save to temporary files.
        
        Args:
            scenes: List of all scenes
            start_idx: Starting index for this window
            temp_dir: Directory for temporary files
            **kwargs: Additional arguments for scene processing
            
        Returns:
            List of VideoFileClips for the processed window
        """
        window_clips = []
        end_idx = min(start_idx + self.window_size, len(scenes))
        
        logger.info(f"Processing scene window {start_idx + 1} to {end_idx}")
        
        try:
            for i in range(start_idx, end_idx):
                # Monitor memory before processing each scene
                self.memory_monitor.log_memory_status(f"Before scene {i + 1}")
                
                # Create scene clip
                logger.info(f"Creating scene {i + 1}")
                clip = None
                
                try:
                    # Create the scene clip
                    clip = self._create_scene_clip(scenes[i], i)
                    
                    # Safely calculate memory estimate with fallback for None values
                    try:
                        # Get clip properties with fallbacks for None values
                        width = getattr(clip, 'w', 0) or 1920  # Default to HD width
                        height = getattr(clip, 'h', 0) or 1080  # Default to HD height
                        fps = getattr(clip, 'fps', 0) or 24     # Default to 24fps
                        duration = getattr(clip, 'duration', 0) or 10  # Default to 10 seconds
                        
                        # Calculate memory estimate
                        clip_memory_estimate = width * height * 4 * fps * duration / (1024 * 1024)  # MB estimate
                        logger.info(f"Clip memory estimate: {clip_memory_estimate:.1f}MB (w={width}, h={height}, fps={fps}, duration={duration:.1f}s)")
                    except Exception as est_error:
                        # If calculation fails, use a conservative default estimate
                        clip_memory_estimate = 500  # Conservative estimate in MB
                        logger.warning(f"Error calculating memory estimate for scene {i + 1}: {str(est_error)}. Using default: {clip_memory_estimate}MB")
                    
                    # Get memory usage safely
                    try:
                        memory_usage = self.memory_monitor.get_memory_usage()
                        memory_percentage = memory_usage.get('percent', 0)
                    except Exception:
                        # Default to a high value to be conservative
                        memory_percentage = 70
                        logger.warning(f"Error getting memory usage. Using default percentage: {memory_percentage}%")
                    
                    # Only write to disk if memory usage is high or clip is large
                    if memory_percentage > 60 or clip_memory_estimate > 200:  # Thresholds
                        # Save to temporary file
                        temp_path = temp_dir / f"temp_scene_{i}.mp4"
                        logger.info(f"Saving scene {i + 1} to disk (memory: {memory_percentage}%, estimate: {clip_memory_estimate:.1f}MB)")
                        
                        # Write the clip to a file
                        self._write_video_file(clip, str(temp_path), is_intermediate=True, is_high_quality=False)
                        
                        # Register the temporary file
                        self.resource_manager.register_temp_file(temp_path)
                        
                        # Close the original clip to free memory
                        clip.close()
                        
                        # Create a new clip from the saved file and register it
                        new_clip = VideoFileClip(str(temp_path))
                        window_clips.append(self.resource_manager.register_clip(new_clip))
                    else:
                        # Keep clip in memory for faster processing
                        logger.info(f"Keeping scene {i + 1} in memory (memory: {memory_percentage}%, estimate: {clip_memory_estimate:.1f}MB)")
                        window_clips.append(self.resource_manager.register_clip(clip))
                    
                except Exception as scene_error:
                    logger.error(f"Error processing scene {i + 1}: {str(scene_error)}")
                    # Clean up if there was an error
                    if clip is not None:
                        try:
                            clip.close()
                        except Exception:
                            pass
                    raise
                
                # Check memory after processing
                self.memory_monitor.log_memory_status(f"After scene {i + 1}")
                if self.memory_monitor.needs_cleanup():
                    logger.info("Memory threshold exceeded, performing cleanup")
                    self.memory_monitor.force_cleanup()
            
            return window_clips
            
        except Exception as window_error:
            logger.error(f"Error processing scene window {start_idx + 1} to {end_idx}: {str(window_error)}")
            # Clean up any clips we've created
            for clip in window_clips:
                try:
                    clip.close()
                except Exception:
                    pass
            raise MediaGenerationError(f"Failed to process scene window: {str(window_error)}")
        
    def _concatenate_window_clips(
        self,
        clips: List[VideoFileClip],
        temp_dir: Path,
        transition_duration: float = 0.5
    ) -> VideoFileClip:
        """
        Concatenate clips in a memory-efficient way using FFMPEG's concat demuxer when possible.
        
        Args:
            clips: List of clips to concatenate
            temp_dir: Directory for temporary files
            transition_duration: Duration for transitions between clips
            
        Returns:
            Concatenated video clip
        """
        if not clips:
            raise MediaGenerationError("No clips to concatenate")
            
        if len(clips) == 1:
            # For single clips, just return the original (no need to copy)
            logger.info("Only one clip to process, returning directly")
            return clips[0]
            
        logger.info(f"Concatenating {len(clips)} clips with {transition_duration}s transitions")
        
        # Check if we can use direct FFMPEG concatenation (no transitions requested)
        if transition_duration <= 0:
            try:
                # Use FFMPEG's concat demuxer for more efficient concatenation
                # First make sure all clips are saved to disk
                video_files = []
                for i, clip in enumerate(clips):
                    # If the clip is not a VideoFileClip or not from a file, save it
                    if not hasattr(clip, 'filename') or not clip.filename:
                        temp_path = temp_dir / f"clip_{i}.mp4"
                        logger.info(f"Saving clip {i} to disk for efficient concatenation")
                        self._write_video_file(clip, str(temp_path), is_high_quality=False)
                        video_files.append(str(temp_path))
                        # Register the temp file for cleanup
                        self.resource_manager.register_temp_file(temp_path)
                    else:
                        video_files.append(clip.filename)
                
                # Create a manifest file for ffmpeg concat demuxer
                manifest_path = temp_dir / "concat_manifest.txt"
                with open(manifest_path, 'w') as f:
                    for video_file in video_files:
                        f.write(f"file '{video_file}'\n")
                
                # Register the manifest file for cleanup
                self.resource_manager.register_temp_file(manifest_path)
                
                # Prepare output path
                output_path = temp_dir / f"concat_result_{id(clips)}.mp4"
                
                # Use ffmpeg directly with concat demuxer (no re-encoding)
                import subprocess
                command = [
                    "ffmpeg",
                    "-y",                       # Overwrite output files without asking
                    "-f", "concat",             # Use concat demuxer
                    "-safe", "0",               # Don't restrict file paths
                    "-i", str(manifest_path),   # Input manifest file
                    "-c", "copy",               # Copy streams without re-encoding
                    "-movflags", "+faststart",  # Optimize for streaming
                    str(output_path)            # Output file
                ]
                
                logger.info("Executing FFMPEG concat demuxer command")
                process = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if process.returncode != 0:
                    logger.warning(f"FFMPEG concat demuxer failed: {process.stderr}")
                    # Fall back to traditional MoviePy concatenation
                    raise RuntimeError("FFMPEG concat failed, falling back to MoviePy")
                
                # Create a clip from the concatenated file
                logger.info(f"Successfully concatenated using FFMPEG demuxer to {output_path}")
                result_clip = VideoFileClip(str(output_path))
                
                # Register temporary files and clip for cleanup
                self.resource_manager.register_temp_file(output_path)
                return self.resource_manager.register_clip(result_clip)
                
            except Exception as e:
                logger.warning(f"Efficient concatenation failed, falling back to MoviePy: {str(e)}")
                # Fall back to traditional MoviePy concatenation
        
        # Traditional MoviePy concatenation (for transitions or if FFMPEG concat failed)
        try:
            # Add transitions between clips
            if transition_duration > 0:
                clips_with_transitions = []
                
                for i, clip in enumerate(clips):
                    if i == 0:
                        clips_with_transitions.append(clip)
                    else:
                        with_transition = clip.crossfadein(transition_duration)
                        clips_with_transitions.append(with_transition)
                
                # Use 'compose' method for transitions
                logger.info("Concatenating clips with transitions")
                final_clip = concatenate_videoclips(
                    clips_with_transitions,
                    method="compose",
                    padding=-transition_duration
                )
            else:
                # Simple concatenation without transitions
                logger.info("Concatenating clips without transitions")
                final_clip = concatenate_videoclips(clips, method="chain")
            
            # Calculate total duration safely
            try:
                # Safely calculate total duration with fallbacks for None
                total_duration = 0
                for clip in clips:
                    duration = getattr(clip, 'duration', 0)
                    if duration is None:
                        duration = 10  # Default duration if unknown
                    total_duration += duration
                logger.info(f"Total duration of all clips: {total_duration:.1f}s")
            except Exception as est_error:
                # If calculation fails, use a conservative default estimate
                total_duration = len(clips) * 30  # Conservative estimate: 30s per clip
                logger.warning(f"Error calculating total duration: {str(est_error)}. Using default: {total_duration:.1f}s")
            
            # Only write to disk for large concatenations (more than 3 clips)
            # or if the resulting clip is large (estimated by total duration)
            if len(clips) > 3 or total_duration > 60:  # Threshold parameters
                # For larger operations, save to disk to free memory
                temp_path = temp_dir / f"concat_{len(clips)}_{id(final_clip)}.mp4"
                logger.info(f"Saving large concatenated result ({total_duration:.1f}s) to disk")
                
                # Write to file - this is a blocking operation
                self._write_video_file(final_clip, str(temp_path), is_intermediate=False, is_high_quality=True)
                
                # Register the temp file for cleanup
                self.resource_manager.register_temp_file(temp_path)
                
                # Close the concatenated clip to free memory
                final_clip.close()
                
                # Create a new clip from the saved file
                result_clip = VideoFileClip(str(temp_path))
                return self.resource_manager.register_clip(result_clip)
            else:
                # For smaller operations, just return the concatenated clip directly
                logger.info(f"Keeping concatenated result ({total_duration:.1f}s) in memory")
                return self.resource_manager.register_clip(final_clip)
                
        except Exception as e:
            logger.error(f"Error during clip concatenation: {str(e)}")
            raise MediaGenerationError(f"Failed to concatenate clips: {str(e)}")

    def assemble_scenes(
        self,
        scenes: List[Dict],
        output_path: str,
        transition_duration: float = 0.5,
        use_two_pass: bool = False,
        memory_threshold: int = 80,  # Default memory threshold percentage
        optimize_for_youtube: bool = True,  # New parameter for YouTube optimization
        **kwargs
    ) -> str:
        """
        Assemble multiple scenes into a final video using a hybrid memory/disk approach.
        
        Args:
            scenes: List of scene dictionaries containing video and audio paths
            output_path: Path where the final video should be saved
            transition_duration: Duration for transitions between scenes
            use_two_pass: Whether to use two-pass encoding for the final video (better quality)
            memory_threshold: Memory usage percentage threshold before using disk (default: 80%)
            optimize_for_youtube: Whether to use high quality settings for YouTube (default: True)
            **kwargs: Additional arguments for scene processing
            
        Returns:
            Path to the assembled video file
        """
        if not scenes:
            raise MediaGenerationError("No scenes provided for assembly")
            
        output_path = self._prepare_output_path(output_path)
        logger.info(f"Assembling {len(scenes)} scenes into {output_path}")
        
        if use_two_pass:
            logger.info("Two-pass encoding enabled for final output (higher quality, slower)")
        
        if optimize_for_youtube:
            logger.info("Using high quality settings optimized for YouTube")
        else:
            logger.info("Using standard quality settings (faster processing)")
        
        # Create temporary directory in our managed temp location
        temp_base = os.path.join(self.output_dir, 'temp', 'processing')
        with tempfile.TemporaryDirectory(dir=temp_base) as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Initial memory check
            memory_usage = self.memory_monitor.get_memory_usage()
            initial_memory = memory_usage.get('percent', 0)
            logger.info(f"Initial memory usage: {initial_memory:.1f}%")
            
            # Prepare for hybrid approach
            all_scene_clips = []
            batch_size = min(5, len(scenes))  # Start with small batches
            force_disk_mode = False
            
            # Process scenes in adaptive batches
            for i in range(0, len(scenes), batch_size):
                # Determine current batch size
                current_batch_end = min(i + batch_size, len(scenes))
                logger.info(f"Processing scenes {i+1} to {current_batch_end} (batch of {current_batch_end-i})")
                
                # Check memory before processing batch
                memory_usage = self.memory_monitor.get_memory_usage()
                memory_percent = memory_usage.get('percent', 0)
                logger.info(f"Memory before batch: {memory_percent:.1f}%")
                
                # Decide whether to use memory or disk based on current memory usage
                use_disk = force_disk_mode or (memory_percent > memory_threshold)
                
                if use_disk:
                    logger.info(f"Using disk mode (memory at {memory_percent:.1f}%)")
                    # Process this batch with disk storage
                    batch_clips = self._process_batch_to_disk(
                        scenes[i:current_batch_end], 
                        temp_dir,
                        transition_duration,
                        **kwargs
                    )
                else:
                    logger.info(f"Using memory mode (memory at {memory_percent:.1f}%)")
                    # Process this batch in memory
                    batch_clips = self._process_batch_in_memory(
                        scenes[i:current_batch_end],
                        **kwargs
                    )
                
                # Add clips to our result list
                all_scene_clips.extend(batch_clips)
                
                # Check memory after batch
                memory_usage = self.memory_monitor.get_memory_usage()
                memory_percent = memory_usage.get('percent', 0)
                logger.info(f"Memory after batch: {memory_percent:.1f}%")
                
                # Adjust strategy based on memory usage
                if memory_percent > 90:
                    # Memory is critically high, force disk mode and reduce batch size
                    force_disk_mode = True
                    batch_size = max(1, batch_size // 2)
                    logger.warning(f"Memory usage critical ({memory_percent:.1f}%). Reducing batch size to {batch_size} and forcing disk mode.")
                    
                    # Force garbage collection
                    self.memory_monitor.force_cleanup()
                elif memory_percent > memory_threshold:
                    # Memory is high, use disk mode but keep batch size
                    force_disk_mode = True
                    logger.info(f"Memory usage high ({memory_percent:.1f}%). Switching to disk mode.")
                elif memory_percent < memory_threshold * 0.7 and batch_size < len(scenes) // 2:
                    # Memory is fine, potentially increase batch size
                    batch_size = min(batch_size * 2, 10)  # Cap at 10 for safety
                    logger.info(f"Memory usage acceptable ({memory_percent:.1f}%). Increasing batch size to {batch_size}.")
            
            # Concatenate all scene clips
            logger.info(f"Concatenating {len(all_scene_clips)} scene clips")
            
            # Check memory before final concatenation
            memory_usage = self.memory_monitor.get_memory_usage()
            memory_percent = memory_usage.get('percent', 0)
            
            # Use direct concatenation if memory allows, otherwise use disk
            if memory_percent > memory_threshold or len(all_scene_clips) > 10:
                logger.info(f"Using disk-based concatenation (memory: {memory_percent:.1f}%)")
                final_video = self._concatenate_to_disk(
                    all_scene_clips, 
                    temp_dir, 
                    transition_duration
                )
            else:
                logger.info(f"Using in-memory concatenation (memory: {memory_percent:.1f}%)")
                final_video = self._concatenate_in_memory(
                    all_scene_clips,
                    transition_duration
                )
            
            # Write final video
            logger.info("Writing final video")
            self._write_video_file(
                final_video, 
                output_path, 
                is_intermediate=False,  # This is the final output
                is_high_quality=optimize_for_youtube,  # Use the YouTube optimization flag
                use_two_pass=use_two_pass
            )
            
            # Clean up
            final_video.close()
            for clip in all_scene_clips:
                try:
                    clip.close()
                except Exception:
                    pass
            
            logger.info(f"Video assembly complete: {output_path}")
            return output_path

    def _process_batch_in_memory(
        self,
        scenes: List[Dict],
        **kwargs
    ) -> List[VideoFileClip]:
        """
        Process a batch of scenes entirely in memory.
        
        Args:
            scenes: List of scene dictionaries
            **kwargs: Additional arguments for scene processing
            
        Returns:
            List of processed scene clips
        """
        logger.info(f"Processing {len(scenes)} scenes in memory")
        batch_clips = []
        
        for i, scene in enumerate(scenes):
            logger.info(f"Creating scene {i + 1}/{len(scenes)} in memory")
            clip = self._create_scene_clip(scene, i)
            batch_clips.append(self.resource_manager.register_clip(clip))
        
        return batch_clips

    def _process_batch_to_disk(
        self,
        scenes: List[Dict],
        temp_dir: Path,
        transition_duration: float = 0.5,
        **kwargs
    ) -> List[VideoFileClip]:
        """
        Process a batch of scenes writing each to disk.
        
        Args:
            scenes: List of scene dictionaries
            temp_dir: Directory for temporary files
            transition_duration: Duration for transitions
            **kwargs: Additional arguments for scene processing
            
        Returns:
            List of processed scene clips (loaded from disk)
        """
        logger.info(f"Processing {len(scenes)} scenes to disk")
        batch_clips = []
        
        for i, scene in enumerate(scenes):
            logger.info(f"Creating scene {i + 1}/{len(scenes)} to disk")
            
            # Create the scene clip
            clip = self._create_scene_clip(scene, i)
            
            # Save to disk immediately
            temp_path = temp_dir / f"scene_{len(batch_clips)}.mp4"
            logger.info(f"Saving scene to {temp_path}")
            self._write_video_file(clip, str(temp_path), is_intermediate=True, is_high_quality=False)
            
            # Close the memory clip
            clip.close()
            
            # Create new clip from disk
            disk_clip = VideoFileClip(str(temp_path))
            self.resource_manager.register_temp_file(temp_path)
            batch_clips.append(self.resource_manager.register_clip(disk_clip))
            
            # Force cleanup after each scene
            self.memory_monitor.force_cleanup()
        
        return batch_clips

    def _concatenate_in_memory(
        self,
        clips: List[VideoFileClip],
        transition_duration: float = 0.5
    ) -> VideoFileClip:
        """
        Concatenate clips entirely in memory.
        
        Args:
            clips: List of clips to concatenate
            transition_duration: Duration for transitions
            
        Returns:
            Concatenated video clip
        """
        logger.info(f"Concatenating {len(clips)} clips in memory")
        
        if len(clips) == 1:
            return clips[0]
            
        # Apply transitions if needed
        if transition_duration > 0:
            clips_with_transitions = []
            for i, clip in enumerate(clips):
                if i == 0:
                    clips_with_transitions.append(clip)
                else:
                    with_transition = clip.crossfadein(transition_duration)
                    clips_with_transitions.append(with_transition)
                    
            logger.info(f"Using compose method with {transition_duration}s transitions")
            return concatenate_videoclips(
                clips_with_transitions,
                method="compose", 
                padding=-transition_duration
            )
        else:
            logger.info("Using chain method without transitions")
            return concatenate_videoclips(clips, method="chain")

    def _concatenate_to_disk(
        self,
        clips: List[VideoFileClip],
        temp_dir: Path,
        transition_duration: float = 0.5
    ) -> VideoFileClip:
        """
        Concatenate clips using disk for intermediate results.
        
        Args:
            clips: List of clips to concatenate
            temp_dir: Directory for temporary files
            transition_duration: Duration for transitions
            
        Returns:
            Concatenated video clip
        """
        logger.info(f"Concatenating {len(clips)} clips using disk")
        
        if len(clips) == 1:
            return clips[0]
            
        # First create in-memory concatenation
        concat_clip = self._concatenate_in_memory(clips, transition_duration)
        
        # Then save to disk
        temp_path = temp_dir / f"concat_result.mp4"
        logger.info(f"Saving concatenated result to {temp_path}")
        self._write_video_file(concat_clip, str(temp_path), is_intermediate=True, is_high_quality=False)
        
        # Close the memory clip
        concat_clip.close()
        
        # Create new clip from disk
        disk_clip = VideoFileClip(str(temp_path))
        self.resource_manager.register_temp_file(temp_path)
        
        return disk_clip
        
    def combine_video_and_audio_only(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        use_two_pass: bool = False,
        optimize_for_youtube: bool = True  # New parameter for YouTube optimization
    ) -> str:
        """
        Combines a video with an audio track without modifying the video content.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path where to save the final video
            use_two_pass: Whether to use two-pass encoding for better quality
            optimize_for_youtube: Whether to use high quality settings for YouTube
            
        Returns:
            str: Path to the assembled video
        """
        try:
            logger.info("Combining video and audio...")
            if use_two_pass:
                logger.info("Two-pass encoding enabled (higher quality, slower)")
            
            if optimize_for_youtube:
                logger.info("Using high quality settings optimized for YouTube")
            else:
                logger.info("Using standard quality settings (faster processing)")
            
            # Prepare output path
            output_path = self._prepare_output_path(output_path)
                
            # Load video and audio
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Set audio to video
            final_clip = video.set_audio(audio)
            
            # Write the final clip using high quality settings
            logger.info(f"Writing final video to {output_path}...")
            self._write_video_file(
                final_clip, 
                output_path, 
                is_intermediate=False,  # This is the final output
                is_high_quality=optimize_for_youtube,  # Use YouTube optimization setting
                use_two_pass=use_two_pass
            )
            
            # Clean up
            video.close()
            audio.close()
            final_clip.close()
            
            logger.info("Video assembly completed successfully!")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during video assembly: {str(e)}")
            raise 

    def _detect_available_encoders(self) -> dict:
        """
        Detect actually available hardware encoders on the system.
        Uses a test encoding to verify each encoder works correctly.
        
        Returns:
            Dictionary of working encoder names mapped to their codec names
        """
        # Potential hardware encoders to test
        potential_encoders = {
            'nvidia': 'h264_nvenc',
            'intel': 'h264_qsv',
            'amd': 'h264_amf',
            'vaapi': 'h264_vaapi'
        }
        
        available_encoders = {}
        
        # Use a very short command to test each encoder
        import subprocess
        import os
        import shutil
        
        # Only test if ffmpeg is available
        if not shutil.which('ffmpeg'):
            logger.info("FFmpeg not found in PATH, skipping hardware encoder detection")
            return available_encoders
            
        # Create test directory if needed
        test_dir = os.path.join(self.output_dir, 'temp', 'encoder_test')
        os.makedirs(test_dir, exist_ok=True)
        
        for name, codec in potential_encoders.items():
            # Skip test for encoders we've already checked
            if name in self.tested_encoders:
                continue
                
            # Mark this encoder as tested so we don't check it again
            self.tested_encoders.add(name)
            
            test_output = os.path.join(test_dir, f"test_{name}.mp4")
            if os.path.exists(test_output):
                try:
                    os.remove(test_output)
                except:
                    pass
                    
            # Very small, quick encoding test
            cmd = [
                'ffmpeg',
                '-loglevel', 'error',  # Reduce output noise
                '-f', 'lavfi',         # Use libavfilter
                '-i', 'color=black:s=64x64:r=1:d=1',  # Generate tiny black frame
                '-c:v', codec,         # Test this encoder
                '-y',                  # Overwrite output
                '-frames:v', '1',      # Just one frame
                test_output            # Output path
            ]
            
            try:
                # Run with timeout to avoid hanging
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
                
                # Check if file was created successfully
                if result.returncode == 0 and os.path.exists(test_output) and os.path.getsize(test_output) > 0:
                    logger.info(f"Hardware encoder '{name}' ({codec}) is available and working")
                    available_encoders[name] = codec
                else:
                    logger.debug(f"Hardware encoder '{name}' ({codec}) test failed with code {result.returncode}")
            except subprocess.TimeoutExpired:
                logger.debug(f"Hardware encoder test for '{name}' ({codec}) timed out")
            except Exception as e:
                logger.debug(f"Error testing encoder '{name}' ({codec}): {str(e)}")
                
            # Clean up test file
            if os.path.exists(test_output):
                try:
                    os.remove(test_output)
                except:
                    pass
                    
        # Clean up test directory if it's empty
        try:
            os.rmdir(test_dir)
        except:
            pass
            
        return available_encoders 