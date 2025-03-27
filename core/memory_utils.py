"""
Memory monitoring and management utilities.
Provides tools for tracking memory usage and triggering cleanup when needed.
"""

import os
import psutil
import logging
from typing import Dict, Optional, Generator, Tuple, List, Union
from pathlib import Path
import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

class FrameGenerator:
    """
    Generator-based approach for memory-efficient frame processing.
    Yields frames one at a time to avoid holding entire videos in memory.
    """
    
    def __init__(self, 
                 image_path: Union[str, Path] = None, 
                 video_path: Union[str, Path] = None,
                 duration: float = None,
                 fps: int = 24,
                 animation_style: str = "ken_burns",
                 zoom_range: Tuple[float, float] = (1.0, 1.3),
                 pan_range: Tuple[float, float] = (-0.1, 0.1)):
        """
        Initialize frame generator for either image animation or video processing.
        
        Args:
            image_path: Path to image for animation-based generation
            video_path: Path to video for frame-by-frame processing
            duration: Duration for animated image clips
            fps: Frames per second
            animation_style: Style for image animation ("ken_burns", "zoom", "pan")
            zoom_range: Range of zoom factors (start, end)
            pan_range: Range of pan offsets (start, end)
        """
        self.image_path = Path(image_path) if image_path else None
        self.video_path = Path(video_path) if video_path else None
        self.duration = duration
        self.fps = fps
        self.animation_style = animation_style
        self.zoom_range = zoom_range
        self.pan_range = pan_range
        self.total_frames = int(duration * fps) if duration else 0
        
        # Load image if provided
        if self.image_path:
            self._initialize_image()
        elif self.video_path:
            self._initialize_video()
        else:
            raise ValueError("Either image_path or video_path must be provided")
            
        logger.debug(f"Created frame generator with {self.total_frames} frames")
    
    def _initialize_image(self):
        """Load and prepare the image for animation."""
        # Load the image using PIL
        self.pil_img = Image.open(str(self.image_path))
        # Convert to RGB mode if necessary
        if self.pil_img.mode != 'RGB':
            self.pil_img = self.pil_img.convert('RGB')
        
        # Get dimensions
        self.width, self.height = self.pil_img.size
        
        # Resize to target dimensions if needed
        target_w, target_h = 1280, 720
        if self.width != target_w or self.height != target_h:
            logger.debug(f"Resizing image from {self.width}x{self.height} to {target_w}x{target_h}")
            self.pil_img = self.pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            self.width, self.height = target_w, target_h
            
        # Pre-calculate frame parameters for optimization
        self.frame_params = []
        for frame_idx in range(self.total_frames):
            # Calculate progress (0 to 1)
            t = frame_idx / self.fps
            progress = t / self.duration
            
            # Smooth easing function
            ease = lambda x: 0.5 - np.cos(x * np.pi) / 2
            smooth_progress = ease(progress)
            
            # Calculate parameters based on animation style
            if self.animation_style == "ken_burns":
                zoom = self.zoom_range[0] + (self.zoom_range[1] - self.zoom_range[0]) * smooth_progress
                pan_x = self.pan_range[0] + (self.pan_range[1] - self.pan_range[0]) * smooth_progress
                pan_y = self.pan_range[0] + (self.pan_range[1] - self.pan_range[0]) * smooth_progress
                
                # Calculate dimensions and positions
                zoomed_w = int(self.width * zoom)
                zoomed_h = int(self.height * zoom)
                pos_x = int((zoomed_w - self.width) * (0.5 + pan_x))
                pos_y = int((zoomed_h - self.height) * (0.5 + pan_y))
                
                # Ensure we don't go out of bounds
                pos_x = max(0, min(pos_x, zoomed_w - self.width))
                pos_y = max(0, min(pos_y, zoomed_h - self.height))
                
                self.frame_params.append({
                    'zoom': zoom,
                    'zoomed_w': zoomed_w,
                    'zoomed_h': zoomed_h,
                    'pos_x': pos_x,
                    'pos_y': pos_y,
                })
    
    def _initialize_video(self):
        """Prepare for video frame-by-frame processing."""
        # Import here to avoid circular imports
        from moviepy.editor import VideoFileClip
        
        # Just get video info without loading all frames
        clip = VideoFileClip(str(self.video_path))
        self.total_frames = int(clip.duration * clip.fps)
        self.width, self.height = clip.size
        self.duration = clip.duration
        self.fps = clip.fps
        # Close clip to free memory
        clip.close()
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generate frames one at a time without storing all in memory.
        
        Yields:
            numpy array representing a single frame
        """
        if self.image_path:
            # Generate animated frames from a single image
            for frame_idx, params in enumerate(self.frame_params):
                # Apply the Ken Burns effect
                if self.animation_style == "ken_burns":
                    # Create zoomed version
                    zoomed_pil = self.pil_img.resize(
                        (params['zoomed_w'], params['zoomed_h']), 
                        Image.Resampling.LANCZOS
                    )
                    
                    # Extract the visible portion
                    frame = zoomed_pil.crop((
                        params['pos_x'], 
                        params['pos_y'], 
                        params['pos_x'] + self.width, 
                        params['pos_y'] + self.height
                    ))
                    
                    # Apply fade-in/out if needed
                    if frame_idx < self.fps / 2:  # First 0.5 seconds
                        alpha = frame_idx / (self.fps / 2)
                        # Apply fade in (simplified for example)
                        frame = Image.blend(
                            Image.new('RGB', frame.size, (0, 0, 0)),
                            frame,
                            alpha
                        )
                    elif frame_idx > self.total_frames - self.fps / 2:  # Last 0.5 seconds
                        alpha = (self.total_frames - frame_idx) / (self.fps / 2)
                        # Apply fade out (simplified for example)
                        frame = Image.blend(
                            Image.new('RGB', frame.size, (0, 0, 0)),
                            frame,
                            alpha
                        )
                
                # Convert PIL image to numpy array
                yield np.array(frame)
        
        elif self.video_path:
            # Process video frames one by one
            from moviepy.editor import VideoFileClip
            
            # Open the video but don't load all frames
            with VideoFileClip(str(self.video_path)) as clip:
                # Loop through frames without loading entire video
                for t in np.arange(0, clip.duration, 1/self.fps):
                    frame = clip.get_frame(t)
                    yield frame

    def get_frame_at_time(self, t: float) -> np.ndarray:
        """
        Get a single frame at a specific time point.
        
        Args:
            t: Time point in seconds
            
        Returns:
            Frame as numpy array
        """
        if self.image_path:
            frame_idx = min(int(t * self.fps), self.total_frames - 1)
            params = self.frame_params[frame_idx]
            
            # Create zoomed version
            zoomed_pil = self.pil_img.resize(
                (params['zoomed_w'], params['zoomed_h']), 
                Image.Resampling.LANCZOS
            )
            
            # Extract the visible portion
            frame = zoomed_pil.crop((
                params['pos_x'], 
                params['pos_y'], 
                params['pos_x'] + self.width, 
                params['pos_y'] + self.height
            ))
            
            return np.array(frame)
        
        elif self.video_path:
            # Use MoviePy to get the frame at the specified time
            from moviepy.editor import VideoFileClip
            
            with VideoFileClip(str(self.video_path)) as clip:
                return clip.get_frame(t)

class MemoryMonitor:
    """Monitors memory usage and provides utilities for memory management."""
    
    def __init__(self, max_memory_mb: int = 1024, cleanup_threshold: float = 0.8):
        """
        Initialize the memory monitor.
        
        Args:
            max_memory_mb: Maximum allowed memory usage in MB
            cleanup_threshold: Fraction of max memory that triggers cleanup
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.cleanup_threshold = cleanup_threshold
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict containing memory usage information in MB
        """
        memory_info = self.process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            'percent': self.process.memory_percent(),
            'system_percent': psutil.virtual_memory().percent
        }
        
    def needs_cleanup(self) -> bool:
        """
        Check if memory usage exceeds threshold and cleanup is needed.
        
        Returns:
            True if cleanup is needed, False otherwise
        """
        memory_info = self.process.memory_info()
        threshold = self.max_memory * self.cleanup_threshold
        
        if memory_info.rss > threshold:
            logger.warning(
                f"Memory usage ({memory_info.rss / 1024 / 1024:.1f}MB) "
                f"exceeds threshold ({threshold / 1024 / 1024:.1f}MB)"
            )
            return True
        return False
        
    def force_cleanup(self):
        """Force memory cleanup using garbage collection."""
        import gc
        
        # Get memory usage before cleanup
        before = self.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        
        # Get memory usage after cleanup
        after = self.get_memory_usage()
        
        # Log the cleanup results
        logger.info(
            f"Memory cleanup completed:\n"
            f"  Before: {before['rss']:.1f}MB\n"
            f"  After:  {after['rss']:.1f}MB\n"
            f"  Freed:  {(before['rss'] - after['rss']):.1f}MB"
        )
        
    def log_memory_status(self, context: str = ""):
        """
        Log current memory usage status.
        
        Args:
            context: Optional context string for the log message
        """
        usage = self.get_memory_usage()
        
        logger.info(
            f"Memory status{f' ({context})' if context else ''}:\n"
            f"  RSS:     {usage['rss']:.1f}MB\n"
            f"  VMS:     {usage['vms']:.1f}MB\n"
            f"  Process: {usage['percent']:.1f}%\n"
            f"  System:  {usage['system_percent']:.1f}%"
        )

def get_available_memory() -> int:
    """
    Get available system memory in bytes.
    
    Returns:
        Available memory in bytes
    """
    return psutil.virtual_memory().available

def estimate_video_memory(
    width: int,
    height: int,
    duration: float,
    fps: int = 24,
    bits_per_pixel: int = 24
) -> int:
    """
    Estimate memory requirements for a video clip.
    
    Args:
        width: Video width in pixels
        height: Video height in pixels
        duration: Video duration in seconds
        fps: Frames per second
        bits_per_pixel: Bits per pixel (typically 24 for RGB)
        
    Returns:
        Estimated memory requirement in bytes
    """
    bytes_per_frame = (width * height * bits_per_pixel) // 8
    total_frames = int(duration * fps)
    return bytes_per_frame * total_frames

def create_memory_efficient_temp_dir() -> Optional[Path]:
    """
    Create a temporary directory in the most memory-efficient location available.
    Prefers RAM disk if available, falls back to fastest available disk.
    
    Returns:
        Path to the created temporary directory, or None if creation failed
    """
    try:
        # Check if RAM disk is available (common locations)
        ram_disk_paths = ['/dev/shm', '/run/shm', '/tmp']
        
        for path in ram_disk_paths:
            if os.path.exists(path) and os.access(path, os.W_OK):
                temp_dir = Path(path) / 'video_pipeline_temp'
                temp_dir.mkdir(exist_ok=True)
                logger.info(f"Created temporary directory on RAM disk: {temp_dir}")
                return temp_dir
        
        # Fall back to system temp directory
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / 'video_pipeline_temp'
        temp_dir.mkdir(exist_ok=True)
        logger.info(f"Created temporary directory on disk: {temp_dir}")
        return temp_dir
        
    except Exception as e:
        logger.error(f"Failed to create temporary directory: {e}")
        return None 