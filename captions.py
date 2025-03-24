"""
Handles the generation and overlay of captions on videos.
Features:
- Creating caption overlays with customizable styles
- Timing caption display with voiceover
- Supporting multiple caption formats and positions
- TikTok-style karaoke captions with word-level highlighting
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from PIL import Image, ImageDraw

from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    ImageClip
)

# Configure logging
logger = logging.getLogger(__name__)

def calculate_dynamic_timing_adjustment(word: str, index: int, total_words: int, 
                                       sentence_duration: float, base_adjustment: float,
                                       tts_provider: str = "unrealspeech") -> float:
    """
    Calculate a dynamic timing adjustment for a word based on various factors.
    
    Args:
        word: The word to calculate adjustment for
        index: Position of word in the sentence (0-based)
        total_words: Total number of words in sentence
        sentence_duration: Duration of the complete sentence in seconds
        base_adjustment: Base timing adjustment value (e.g., 0.6)
        tts_provider: TTS provider name
        
    Returns:
        float: Dynamic timing adjustment in seconds
    """
    # Start with the base adjustment
    adjustment = base_adjustment
    
    # Factor 1: Word length adjustment (longer words need more lead time)
    length_factor = len(word) / 5  # Normalize to a typical word length
    length_adjustment = base_adjustment * (length_factor - 1) * 0.2
    adjustment += max(-0.1, min(0.3, length_adjustment))  # Constrain the impact
    
    # Factor 2: Position-based scaling (later words need more adjustment)
    position_ratio = index / max(1, total_words - 1)  # 0 to 1
    position_scaling = 0.2 * position_ratio  # Up to 0.2s additional for last word
    adjustment += position_scaling
    
    # Factor 3: Speaking rate consideration
    avg_word_duration = sentence_duration / total_words if total_words > 0 else 0.3
    if avg_word_duration < 0.2:  # Fast speaking rate
        adjustment += 0.1  # Add more lead time for fast speech
    elif avg_word_duration > 0.4:  # Slow speaking rate
        adjustment -= 0.1  # Reduce lead time for slow speech
    
    # Factor 4: Special word handling
    if any(char.isdigit() for char in word):  # Contains numbers
        adjustment += 0.1  # Numbers often take longer to pronounce
    
    if word.endswith((',', '.', '!', '?', ':', ';')):
        adjustment += 0.05  # Words before punctuation often have longer durations
    
    # Provider-specific adjustments
    if tts_provider.lower() == "elevenlabs":
        adjustment += 0.1  # Add provider-specific offset if needed
    
    # Ensure reasonable bounds
    return max(0.3, min(1.5, adjustment))

def convert_color_to_rgb(color_value):
    """
    Convert string colors (names or hex) to RGB tuples.
    
    Args:
        color_value: Color value as string (name or hex) or RGB tuple
        
    Returns:
        tuple: RGB tuple (r, g, b)
    """
    if isinstance(color_value, tuple):
        if len(color_value) >= 3:
            return color_value[:3]  # Return just RGB components
        else:
            return (255, 255, 255)  # Default to white for invalid tuples
    
    if isinstance(color_value, str):
        color_value = color_value.lower()
        # Common color names
        if color_value == "white":
            return (255, 255, 255)
        elif color_value == "black":
            return (0, 0, 0)
        elif color_value == "red":
            return (255, 0, 0)
        elif color_value == "green":
            return (0, 255, 0)
        elif color_value == "blue":
            return (0, 0, 255)
        elif color_value == "yellow":
            return (255, 255, 0)
        # Hex color
        elif color_value.startswith("#"):
            try:
                h = color_value.lstrip('#')
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            except Exception:
                return (255, 255, 255)  # Default to white on error
                
    # Default to white for any other cases
    return (255, 255, 255)

def create_text_clip(
    text: str,
    font: str, 
    font_size: int, 
    color: str, 
    position: tuple = None,
    start_time: float = None,
    end_time: float = None,
    stroke_color: str = None,
    stroke_width: int = 0,
    method: str = None,
    align: str = None,
    size: tuple = None
) -> TextClip:
    """
    Create a text clip with common parameters and optionally position and time it.
    
    Args:
        text: Text content
        font: Font name
        font_size: Font size
        color: Text color
        position: Optional (x,y) position tuple
        start_time: Optional start time
        end_time: Optional end time
        stroke_color: Optional outline color
        stroke_width: Optional outline width
        method: Optional TextClip method ('caption', etc)
        align: Optional text alignment
        size: Optional size constraint tuple (width, height)
        
    Returns:
        TextClip: The created and configured text clip
    """
    # Create the text clip with all provided parameters
    params = {
        'txt': text,
        'fontsize': font_size,
        'font': font,
        'color': color
    }
    
    # Add optional parameters if provided
    if stroke_color and stroke_width:
        params['stroke_color'] = stroke_color
        params['stroke_width'] = stroke_width
    
    if method:
        params['method'] = method
        
    if align:
        params['align'] = align
        
    if size:
        params['size'] = size
    
    # Create the clip
    clip = TextClip(**params)
    
    # Apply position if provided
    if position:
        clip = clip.set_position(position)
    
    # Apply timing if provided
    if start_time is not None and end_time is not None:
        clip = clip.set_start(start_time).set_end(end_time)
    
    return clip

def create_shadow_text_clip(
    text: str,
    font: str, 
    font_size: int, 
    position: tuple,
    shadow_offset: int = 2,
    start_time: float = None,
    end_time: float = None
) -> TextClip:
    """
    Create a shadow effect for text by putting a black offset version behind it.
    
    Args:
        text: The text content
        font: Font name
        font_size: Font size
        position: Position tuple (x, y)
        shadow_offset: Offset for shadow in pixels
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        TextClip: The shadow text clip
    """
    shadow = TextClip(
        text,
        font=font,
        fontsize=font_size,
        color="black"
    )
    
    # Set exact integer position with offset
    shadow_x, shadow_y = position
    shadow = shadow.set_position((int(shadow_x + shadow_offset), int(shadow_y + shadow_offset)))
    
    # Set duration if provided
    if start_time is not None and end_time is not None:
        shadow = shadow.set_start(start_time).set_end(end_time)
    
    return shadow

def apply_smooth_transitions(
    clip,
    word_duration: float,
    crossfade_duration: float = 0.12,
    min_word_duration_for_fade: float = 0.2,
    max_fade_duration: float = 0.1
) -> TextClip:
    """
    Apply smooth fade in/out transitions to a clip if the word duration is long enough.
    
    Args:
        clip: The clip to apply transitions to
        word_duration: Duration of the word in seconds
        crossfade_duration: Desired crossfade duration
        min_word_duration_for_fade: Minimum word duration to apply fades
        max_fade_duration: Maximum fade duration in seconds
        
    Returns:
        TextClip: Clip with transitions applied
    """
    # Only apply fades if word is long enough
    if word_duration > min_word_duration_for_fade:
        # Use either the crossfade value or max duration, whichever is smaller
        fade_duration = min(crossfade_duration, max_fade_duration)
        # Apply crossfades
        clip = clip.crossfadein(fade_duration).crossfadeout(fade_duration)
    
    return clip

def overlay_caption_on_video(
    video: VideoFileClip,
    caption_clip: TextClip,
) -> CompositeVideoClip:
    """
    Core utility function to overlay a caption clip on a video.
    Used by both the scene builder pipeline and legacy caption generator.
    
    Args:
        video: The base video clip
        caption_clip: The prepared caption clip
        
    Returns:
        CompositeVideoClip: Video with caption overlaid
    """
    return CompositeVideoClip(
        [video, caption_clip],
        size=video.size
    )

def create_caption_clip(
    text: str,
    duration: float,
    video_height: int,
    video_width: int,
    font: str = "Arial",
    font_size: int = 18,
    color: str = "white",
    position: str = "bottom",
    debug: bool = False
) -> TextClip:
    """
    Creates a static text overlay clip using MoviePy.
    This is the core caption generation function used by SceneBuilder and VideoAssembler.
    Font size is set to 18px for good readability while maintaining clean appearance.
    
    Args:
        text: The narration text to display as caption (should match TTS audio)
        duration: Duration of the caption in seconds (should match audio duration)
        video_height: Height of the video in pixels
        video_width: Width of the video in pixels
        font: Font name to use
        font_size: Font size in pixels (default: 18)
        color: Text color (default: white)
        position: Either "bottom" or "top" (default: bottom)
        debug: If True, saves a test frame to verify caption appearance (default: False)
    
    Returns:
        TextClip: The generated caption clip
        
    Raises:
        ValueError: If text is empty or duration is not positive
    """
    # Remove any <SPLIT> markers from the text
    text = text.replace("<SPLIT>", " ").strip()
    
    if not text.strip():
        raise ValueError("Caption text cannot be empty")
    if duration <= 0:
        raise ValueError("Duration must be positive")

    logger.info(f"Creating caption with text: {text}")
    logger.info(f"Using font: {font} at size: {font_size}")
    
    try:
        test_clip = TextClip("Test", font=font)
        test_clip.close()
        logger.info("Font test successful")
    except Exception as e:
        logger.error(f"Font test failed: {e}")
        logger.info("Falling back to default font")
        font = None  # Let MoviePy use default font

    txt_clip = TextClip(
        text,
        fontsize=font_size,
        color=color,
        font=font,
        stroke_color='black',
        stroke_width=1,
        method='caption',
        size=(video_width, None),
        align='center',
    )
    
    # Get the height of the text
    txt_height = txt_clip.size[1]
    
    # Create a semi-transparent background for better readability
    # Calculate width with some padding
    padding = 20
    bg_width = min(txt_clip.size[0] + padding * 2, video_width)
    
    # Create background clip
    bg_clip = ColorClip(
        size=(bg_width, txt_height + padding), 
        color=(0, 0, 0)
    ).set_opacity(0.5).set_duration(duration)
    
    # Set duration for text clip
    txt_clip = txt_clip.set_duration(duration)
    
    # Composite text over background
    txt_clip = CompositeVideoClip(
        [
            bg_clip.set_position('center'),
            txt_clip.set_position('center')
        ],
        size=(video_width, txt_height + padding)
    )
    
    logger.info(f"Created text clip with size: {txt_clip.size}")
    
    # Increase bottom margin for better placement
    margin = 50  # Increased margin for better spacing from bottom
    
    if position == "bottom":
        y_position = video_height - txt_clip.size[1] - margin  # Place above bottom margin
    else:  # top
        y_position = margin  # Place below top margin
    
    logger.info(f"Calculated y_position: {y_position}")
    
    # Set final position and duration
    txt_clip = (txt_clip
                .set_position(('center', y_position))
                .set_duration(duration))
    
    if debug:
        # Create a test frame to verify caption appearance
        test_frame = ColorClip(size=(1920, video_height), color=(0,0,0), duration=0.1)
        test_composite = CompositeVideoClip([test_frame, txt_clip])
        
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)
        
        # Save test frame
        test_frame_path = debug_dir / f"caption_test_{text[:20].replace(' ', '_')}.png"
        test_composite.save_frame(str(test_frame_path), t=0)
        logger.info(f"Saved caption test frame to: {test_frame_path}")
        
        # Clean up
        test_frame.close()
        test_composite.close()
    
    return txt_clip

def create_karaoke_captions(
    word_timings: List[Tuple[str, float, float]],
    video_width: int,
    video_height: int,
    style: Dict[str, Any]
) -> List[TextClip]:
    """
    Create TikTok-style karaoke captions with word-level highlighting and page-based display.
    
    Args:
        word_timings: List of (word, start_time, end_time) tuples
        video_width: Width of the video
        video_height: Height of the video
        style: Dictionary of style parameters
            - font: Font name
            - font_size: Font size
            - color: Regular text color
            - highlight_color: Color for highlighted words
            - highlight_bg_color: Background color for highlighted words (for boxed style)
            - stroke_color: Outline color
            - stroke_width: Outline width
            - bg_color: Background color (with alpha)
            - use_background: Whether to use a background (default: False)
            - highlight_use_box: Whether to use box highlighting style (default: False)
            - visible_lines: Number of lines to show at once (default: 2)
            - bottom_padding: Padding from bottom of screen (default: 80)
            - timing_adjustment: Seconds to adjust timing (default: 0.4)
            - use_timing_adjustment: Whether to apply timing adjustment (default: True)
            - frame_buffer: Small time gap between word highlights (default: 1/24 second)
            - word_spacing: Extra space to add between words (default: 10)
            
    Returns:
        List[TextClip]: List of text clips for each word and background
    """
    if not word_timings:
        return []
    
    # Check for spacing and positioning preferences
    preserve_spacing = style.get("preserve_spacing", True)
    consistent_positioning = style.get("consistent_positioning", True)
    
    # Get buffer time - small gap between word highlights to prevent frame-perfect timing issues
    # Default to 1/24 of a second (one frame at 24fps)
    frame_buffer = style.get("frame_buffer", 1/24)
    
    # IMPROVEMENT 1: Content-aware sizing
    # Calculate text complexity to adjust font size and visible lines
    full_text = ' '.join([w[0] for w in word_timings])
    text_complexity = len(full_text)
    
    # Apply content-aware adjustments for complex content
    if text_complexity > 100:  # Threshold for complex content
        style["font_size"] = int(style.get("font_size", 60) * 0.85)
        style["visible_lines"] = max(style.get("visible_lines", 2), 3)
    
    # Apply timing adjustment to compensate for perception delay, if enabled
    use_timing_adjustment = style.get("use_timing_adjustment", True)
    if use_timing_adjustment:
        # Get base timing adjustment
        base_adjustment = style.get("timing_adjustment", 0.6)
        
        # Get TTS provider if available
        tts_provider = style.get("tts_provider", "unrealspeech")
        
        # Calculate total duration for rate analysis
        if word_timings:
            sentence_duration = word_timings[-1][2] - word_timings[0][1]
            total_words = len(word_timings)
        else:
            sentence_duration = 0
            total_words = 0
        
        adjusted_word_timings = []
        
        for i, (word, start_time, end_time) in enumerate(word_timings):
            # Calculate dynamic adjustment for this specific word
            dynamic_adjustment = calculate_dynamic_timing_adjustment(
                word=word, 
                index=i, 
                total_words=total_words,
                sentence_duration=sentence_duration, 
                base_adjustment=base_adjustment,
                tts_provider=tts_provider
            )
            
            # Adjust start and end times using the dynamic adjustment
            adjusted_start = max(0, start_time - dynamic_adjustment)
            # Add a tiny buffer at the end to prevent perfect frame alignment issues
            adjusted_end = max(adjusted_start + 0.05, end_time - dynamic_adjustment)
            adjusted_word_timings.append((word, adjusted_start, adjusted_end))
        
        # Use adjusted timings for the rest of the function
        word_timings = adjusted_word_timings
    
    # Clean word timings - just remove <SPLIT> markers and trim whitespace
    cleaned_word_timings = []
    full_text = ""
    
    for word, start_time, end_time in word_timings:
        # Remove <SPLIT> markers and clean
        cleaned_word = word.replace("<SPLIT>", "").strip()
        
        # Skip empty words
        if not cleaned_word:
            continue
        
        # Add to full text and word timings
        full_text += cleaned_word + " "
        cleaned_word_timings.append((cleaned_word, start_time, end_time))
    
    word_timings = cleaned_word_timings
    
    # Extract style parameters with defaults
    font = style.get("font", "Arial-Bold")
    font_size = style.get("font_size", 60)
    
    # Ensure text colors are in proper format
    color_value = style.get("color", "white")
    if color_value == "white":
        color = "white"  # Keep as string for TextClip
    elif color_value == "black":
        color = "black"  # Keep as string for TextClip
    else:
        color = color_value
    
    highlight_color_value = style.get("highlight_color", "#ff5c5c")
    if highlight_color_value == "white":
        highlight_color = "white"  # Keep as string for TextClip
    elif highlight_color_value == "black":
        highlight_color = "black"  # Keep as string for TextClip
    else:
        highlight_color = highlight_color_value
    
    # Get highlight background color using utility function
    highlight_bg_color = convert_color_to_rgb(style.get("highlight_bg_color", "white"))
    
    stroke_color_value = style.get("stroke_color", "black")
    if stroke_color_value == "white":
        stroke_color = "white"  # Keep as string for TextClip
    elif stroke_color_value == "black":
        stroke_color = "black"  # Keep as string for TextClip
    else:
        stroke_color = stroke_color_value
    
    stroke_width = style.get("stroke_width", 2)
    
    # Ensure bg_color is in RGB format (r,g,b)
    bg_color_value = style.get("bg_color", (0, 0, 0))
    # Ensure it's a tuple with 3 elements (RGB)
    if isinstance(bg_color_value, tuple):
        if len(bg_color_value) == 3:
            bg_color = bg_color_value
        elif len(bg_color_value) == 4:
            bg_color = bg_color_value[:3]  # Use only RGB components
        else:
            bg_color = (0, 0, 0)  # Default to black
    else:
        # Default to black for non-tuple values
        bg_color = (0, 0, 0)
    
    # Fixed opacity value
    bg_opacity = 0.5
    
    use_background = style.get("use_background", False)
    highlight_use_box = style.get("highlight_use_box", False)
    visible_lines = style.get("visible_lines", 2)
    bottom_padding = style.get("bottom_padding", 80)
    
    line_height = int(font_size * 1.2)
    
    # IMPROVEMENT 2: Dynamic line width adaptation based on content length
    base_ratio = style.get("max_line_width_ratio", 0.9)
    content_factor = min(1.0, 100 / max(1, text_complexity))
    adjusted_ratio = base_ratio * (0.8 + (0.2 * content_factor))
    max_line_width = int(video_width * adjusted_ratio)
    
    bg_margin = 20
    word_spacing = style.get("word_spacing", 10)  # Get word spacing from style, default to 10
    
    # Calculate total duration from word timings
    if word_timings:
        total_duration = word_timings[-1][2]  # End time of last word
    else:
        total_duration = 0
    
    # First pass: organize words into lines
    lines = []
    current_line = []
    current_line_width = 0
    
    for word, start, end in word_timings:
        # Create a temporary TextClip to measure the width of the word
        # Add a space after each word to ensure proper spacing during rendering
        temp_clip = TextClip(word, font=font, fontsize=font_size, color=color)
        word_width = temp_clip.w
        temp_clip.close()
        
        # Check if adding this word would exceed the max line width
        if current_line_width + word_width > max_line_width and current_line:
            # Add the current line to lines and start a new line
            lines.append(current_line)
            current_line = [(word, start, end, word_width)]
            current_line_width = word_width + word_spacing
        else:
            # Add the word to the current line
            current_line.append((word, start, end, word_width))
            current_line_width += word_width + word_spacing
    
    # Add the last line if not empty
    if current_line:
        lines.append(current_line)
    
    # IMPROVEMENT 4: Implement vertical overflow protection
    if len(lines) > visible_lines:
        # Either increase visible lines or reduce font
        if len(lines) <= 4:  # Can handle with more lines
            visible_lines = len(lines)
            style["visible_lines"] = visible_lines
        else:  # Need font reduction
            reduction_factor = 0.85
            font_size = max(40, int(font_size * reduction_factor))
            # Recalculate line height with new font size
            line_height = int(font_size * 1.2)
    
    # Second pass: organize lines into pages
    pages = []
    current_page = []
    
    for line in lines:
        current_page.append(line)
        if len(current_page) >= visible_lines:
            pages.append(current_page)
            current_page = []
    
    # Add the last page if not empty
    if current_page:
        pages.append(current_page)
    
    # Calculate page transition times
    page_transitions = []
    
    if pages:
        for i, page in enumerate(pages):
            # Start time is the start time of the first word in the first line of the page
            page_start = page[0][0][1]
            
            # End time is the end time of the last word in the last line of the page
            page_end = page[-1][-1][2]
            
            # For the last page, use the total duration as the end time
            if i == len(pages) - 1:
                page_end = total_duration
                
            page_transitions.append((page_start, page_end))
    
    # Calculate base position for caption window (at bottom of screen)
    # For consistent positioning, use a fixed visible_lines value if enabled
    if style.get("consistent_positioning", True):
        # Use a consistent baseline for positioning regardless of actual line count
        base_visible_lines = style.get("base_visible_lines", 3)
        base_y = int(video_height - bottom_padding - (base_visible_lines * line_height))
    else:
        # Use dynamic positioning based on actual visible lines
        base_y = int(video_height - bottom_padding - (visible_lines * line_height))
    
    # Create clips for all words
    clips = []
    
    # For each page
    for page_idx, (page, (page_start, page_end)) in enumerate(zip(pages, page_transitions)):
        # For each line in the page
        for line_idx, line in enumerate(page):
            # Calculate starting x position for this line (center-aligned)
            line_width = sum(word[3] for word in line) + word_spacing * (len(line) - 1)
            start_x = int((video_width - line_width) / 2)  # Ensure integer positioning
            
            current_x = start_x
            
            # Calculate y position (fixed position based on line index within page)
            y_position = int(base_y + (line_idx * line_height))
            
            # Add background for this line if enabled
            if use_background and not highlight_use_box:
                # Convert dimensions to integers for ColorClip
                bg_width = int(line_width + bg_margin*2)
                bg_height = int(line_height)
                bg = ColorClip(
                    size=(bg_width, bg_height), 
                    color=bg_color
                )
                # Set opacity separately
                bg = bg.set_opacity(bg_opacity)
                bg = bg.set_position((int((video_width - line_width) / 2 - bg_margin), int(y_position - bg_margin/2)))
                
                # Set duration
                bg = bg.set_start(page_start).set_end(page_end)
                
                clips.append(bg)
            
            # For each word in this line
            for word, start_time, end_time, word_width in line:
                # Only show text for this page if the word's timing overlaps with the page
                if start_time <= page_end and end_time >= page_start:
                    # Before highlight (if applicable)
                    if start_time > page_start:
                        # Create shadow with slight offset for cleaner look
                        shadow_before = TextClip(
                            word,  # Space is already in the word if preserve_spacing is enabled
                            font=font,
                            fontsize=font_size,
                            color="black"
                        )
                        # Use exact integer values for consistent rendering
                        exact_x_before = int(current_x + 2)
                        exact_y_before = int(y_position + 2)
                        shadow_before = shadow_before.set_position((exact_x_before, exact_y_before))
                        # Add buffer to timing for smoother transitions
                        shadow_before = shadow_before.set_start(page_start).set_end(start_time - frame_buffer)
                        clips.append(shadow_before)
                        
                        # Then create white text on top
                        before_txt = TextClip(
                            word,  # Space is already in the word if preserve_spacing is enabled
                            font=font,
                            fontsize=font_size,
                            color=color
                        )
                        # Use exact integer values for consistent rendering
                        exact_x_text = int(current_x)
                        exact_y_text = int(y_position)
                        before_txt = before_txt.set_position((exact_x_text, exact_y_text))
                        # Add buffer to timing for smoother transitions
                        before_txt = before_txt.set_start(page_start).set_end(start_time - frame_buffer)
                        clips.append(before_txt)
                    
                    # After highlight (if applicable)
                    if end_time < page_end:
                        # Create shadow with slight offset for cleaner look
                        shadow_after = TextClip(
                            word,  # Space is already in the word if preserve_spacing is enabled
                            font=font,
                            fontsize=font_size,
                            color="black"
                        )
                        # Use exact integer values for consistent rendering
                        exact_x_after = int(current_x + 2)
                        exact_y_after = int(y_position + 2)
                        shadow_after = shadow_after.set_position((exact_x_after, exact_y_after))
                        # Add buffer to timing for smoother transitions
                        shadow_after = shadow_after.set_start(end_time + frame_buffer).set_end(page_end)
                        clips.append(shadow_after)
                        
                        # Then create white text on top
                        after_txt = TextClip(
                            word,  # Space is already in the word if preserve_spacing is enabled
                            font=font,
                            fontsize=font_size,
                            color=color
                        )
                        # Use exact integer values for consistent rendering
                        exact_x_text = int(current_x)
                        exact_y_text = int(y_position)
                        after_txt = after_txt.set_position((exact_x_text, exact_y_text))
                        # Add buffer to timing for smoother transitions
                        after_txt = after_txt.set_start(end_time + frame_buffer).set_end(page_end)
                        clips.append(after_txt)
                    
                    # Highlighted word clip - only visible during its timing
                    # Calculate overlap with page
                    overlap_start = max(start_time, page_start)
                    overlap_end = min(end_time, page_end)
                    
                    # Add buffer time for cleaner transitions
                    overlap_start_buffered = overlap_start
                    overlap_end_buffered = overlap_end - frame_buffer
                    
                    # TikTok boxed highlighting style
                    if highlight_use_box:
                        try:
                            if isinstance(highlight_bg_color, tuple) and len(highlight_bg_color) >= 3:
                                box_color = highlight_bg_color[:3]
                            else:
                                box_color = (239, 68, 68)
                            
                            margin_x = 15
                            margin_y = 10
                            
                            # IMPROVEMENT 3: Improve box highlight sizing to prevent overflow
                            box_width = int(word_width + min(margin_x * 1.5, max(10, word_width * 0.15)))
                            box_height = int(font_size * 1.25)
                            
                            box = create_rounded_rectangle_clip(
                                width=box_width,
                                height=box_height,
                                color=box_color,
                                corner_radius=box_height // 2
                            )
                            
                            # Ensure exact integer positioning for consistent placement
                            exact_x = int(current_x)
                            exact_y = int(y_position)
                            
                            # Apply exact position to the box with proper margin
                            box = box.set_position((exact_x - margin_x, exact_y + font_size/4.0 - (box_height - int(font_size * 1.0))/2))
                            
                            # Set timing with buffer
                            box = box.set_start(overlap_start_buffered).set_end(overlap_end_buffered)
                            
                            # Add space after word for consistent spacing during highlight
                            shadow_text = TextClip(
                                word,  # Space is already in the word if preserve_spacing is enabled
                                fontsize=font_size, 
                                font=font, 
                                color="black"
                            )
                            
                            shadow_text = shadow_text.set_position((exact_x + 2, exact_y + 2))
                            shadow_text = shadow_text.set_start(overlap_start).set_end(overlap_end)
                            
                            # Add space after word for consistent spacing during highlight
                            highlighted_text = TextClip(
                                word,  # Space is already in the word if preserve_spacing is enabled
                                fontsize=font_size, 
                                font=font, 
                                color="white"
                            )
                            
                            highlighted_text = highlighted_text.set_position((exact_x, exact_y))
                            highlighted_text = highlighted_text.set_start(overlap_start).set_end(overlap_end)
                            
                            clips.append(box)
                            clips.append(shadow_text)
                            clips.append(highlighted_text)
                        except Exception as e:
                            # Ensure exact integer positioning for consistent placement
                            exact_x = int(current_x)
                            exact_y = int(y_position)
                            
                            # Add space in fallback case as well
                            shadow_fallback = TextClip(
                                word,  # Space is already in the word if preserve_spacing is enabled
                                fontsize=font_size, 
                                font=font, 
                                color="black"
                            )
                            
                            shadow_fallback = shadow_fallback.set_position((exact_x + 2, exact_y + 2))
                            shadow_fallback = shadow_fallback.set_start(overlap_start).set_end(overlap_end)
                            clips.append(shadow_fallback)
                            
                            # Add space in fallback case as well
                            highlighted_text = TextClip(
                                word,  # Space is already in the word if preserve_spacing is enabled
                                fontsize=font_size, 
                                font=font, 
                                color=highlight_color
                            )
                            
                            highlighted_text = highlighted_text.set_position((exact_x, exact_y))
                            highlighted_text = highlighted_text.set_start(overlap_start).set_end(overlap_end)
                            clips.append(highlighted_text)
                    else:
                        # Standard highlighting (just change text color)
                        # Create shadow with slight offset for cleaner look
                        shadow_highlight = TextClip(
                            word,  # Space is already in the word if preserve_spacing is enabled
                            font=font,
                            fontsize=font_size,
                            color="black"
                        )
                        shadow_highlight = shadow_highlight.set_position((int(current_x + 2), int(y_position + 2)))
                        shadow_highlight = shadow_highlight.set_start(overlap_start).set_end(overlap_end)
                        clips.append(shadow_highlight)
                        
                        # Then create highlighted text on top
                        highlight_txt = TextClip(
                            word,  # Space is already in the word if preserve_spacing is enabled
                            font=font,
                            fontsize=font_size,
                            color=highlight_color
                        )
                        highlight_txt = highlight_txt.set_position((int(current_x), int(y_position)))
                        highlight_txt = highlight_txt.set_start(overlap_start).set_end(overlap_end)
                        clips.append(highlight_txt)
                
                # Move x position for next word
                current_x += word_width + word_spacing  # Add the extra spacing
    
    return clips

def create_unified_karaoke_captions(
    word_timings: List[Tuple[str, float, float]],
    video_width: int,
    video_height: int,
    style: Dict[str, Any]
) -> List[TextClip]:
    """
    Create TikTok-style karaoke captions with a unified rendering approach.
    Instead of creating individual TextClips for each word, this creates entire lines
    as single clips and uses composition for highlighting.
    
    Args:
        word_timings: List of (word, start_time, end_time) tuples
        video_width: Width of the video
        video_height: Height of the video
        style: Dictionary of style parameters
            
    Returns:
        List[TextClip]: List of text clips for each line and transition
    """
    if not word_timings:
        return []
    
    # Extract core style parameters
    font = style.get("font", "Arial-Bold")
    font_size = style.get("font_size", 60)
    color = style.get("color", "white")
    highlight_color = style.get("highlight_color", "#ff5c5c")
    highlight_bg_color = convert_color_to_rgb(style.get("highlight_bg_color", (255, 191, 0)))
    
    # Layout parameters
    bottom_padding = style.get("bottom_padding", 80)
    visible_lines = style.get("visible_lines", 2)
    word_spacing = style.get("word_spacing", 10)
    line_height = int(font_size * 1.2)
    use_box_highlighting = style.get("highlight_use_box", False)
    
    # Apply timing adjustment to compensate for perception delay, if enabled
    use_timing_adjustment = style.get("use_timing_adjustment", True)
    if use_timing_adjustment:
        # Get base timing adjustment
        base_adjustment = style.get("timing_adjustment", 0.6)
        
        # Get TTS provider if available
        tts_provider = style.get("tts_provider", "unrealspeech")
        
        # Calculate total duration for rate analysis
        if word_timings:
            sentence_duration = word_timings[-1][2] - word_timings[0][1] 
            total_words = len(word_timings)
        else:
            sentence_duration = 0
            total_words = 0
        
        adjusted_word_timings = []
        
        for i, (word, start_time, end_time) in enumerate(word_timings):
            # Calculate dynamic adjustment for this specific word
            dynamic_adjustment = calculate_dynamic_timing_adjustment(
                word=word, 
                index=i, 
                total_words=total_words,
                sentence_duration=sentence_duration, 
                base_adjustment=base_adjustment,
                tts_provider=tts_provider
            )
            
            # Adjust start and end times using the dynamic adjustment
            adjusted_start = max(0, start_time - dynamic_adjustment)
            # Add a tiny buffer at the end to prevent perfect frame alignment issues
            adjusted_end = max(adjusted_start + 0.05, end_time - dynamic_adjustment)
            adjusted_word_timings.append((word, adjusted_start, adjusted_end))
            
            # Log the adjustment used for debugging (every 10th word)
            if i % 10 == 0:
                logger.debug(f"Unified: Word '{word}': base_adj={base_adjustment:.2f}, dynamic_adj={dynamic_adjustment:.2f}")
        
        # Use adjusted timings for the rest of the function
        word_timings = adjusted_word_timings
    
    # Clean word timings - just remove <SPLIT> markers and trim whitespace
    cleaned_word_timings = []
    full_text = ""
    
    for word, start_time, end_time in word_timings:
        # Remove <SPLIT> markers and clean
        cleaned_word = word.replace("<SPLIT>", "").strip()
        
        # Skip empty words
        if not cleaned_word:
            continue
        
        # Add to full text and word timings
        full_text += cleaned_word + " "
        cleaned_word_timings.append((cleaned_word, start_time, end_time))
    
    word_timings = cleaned_word_timings
    
    # Calculate max line width
    max_line_width = int(video_width * style.get("max_line_width_ratio", 0.85))
    
    # Calculate total duration from word timings
    if word_timings:
        total_duration = word_timings[-1][2]  # End time of last word
    else:
        total_duration = 0
    
    # Organize words into lines
    lines = []
    current_line = []
    current_line_text = ""
    current_line_width = 0
    
    for word, start, end in word_timings:
        # Test if adding this word would exceed max width
        test_text = current_line_text + word + " "
        
        # Use the utility function to create a temporary clip for measurement
        temp_clip = create_text_clip(test_text, font, font_size, color)
        test_width = temp_clip.w
        temp_clip.close()
        
        if test_width > max_line_width and current_line:
            # Add current line and start a new one
            lines.append({
                "words": current_line,
                "text": current_line_text.strip(),
                "start_time": current_line[0][1],
                "end_time": current_line[-1][2]
            })
            current_line = [(word, start, end)]
            current_line_text = word + " "
            current_line_width = 0
        else:
            # Add to current line
            current_line.append((word, start, end))
            current_line_text = test_text
    
    # Add the last line if not empty
    if current_line:
        lines.append({
            "words": current_line,
            "text": current_line_text.strip(),
            "start_time": current_line[0][1],
            "end_time": current_line[-1][2]
        })
    
    # Calculate appropriate "pages" of lines
    pages = []
    current_page = []
    
    for line in lines:
        current_page.append(line)
        if len(current_page) >= visible_lines:
            pages.append(current_page)
            current_page = []
    
    # Add last page if not empty
    if current_page:
        pages.append(current_page)
    
    # Calculate position for the caption block
    if style.get("consistent_positioning", True):
        # Get consistent base position
        base_visible_lines = style.get("base_visible_lines", 3)
        base_y = int(video_height - bottom_padding - (base_visible_lines * line_height))
    else:
        base_y = int(video_height - bottom_padding - (visible_lines * line_height))
    
    # Generate the clips
    all_clips = []
    
    # For each page
    for page_idx, page in enumerate(pages):
        page_start_time = min(line["start_time"] for line in page)
        page_end_time = max(line["end_time"] for line in page)
        
        # For each line in the page
        for line_idx, line in enumerate(page):
            # Calculate y position for this line
            y_position = int(base_y + (line_idx * line_height))
            
            # Create a clip for the full line to serve as the background/unhighlighted version
            line_text = line["text"]
            
            # Use the utility function to create line clips
            line_clip = create_text_clip(
                text=line_text,
                font=font,
                font_size=font_size,
                color=color,
                stroke_color="black",
                stroke_width=1
            )
            
            # Center the line
            line_width = line_clip.w
            line_x = int((video_width - line_width) / 2)
            
            # Create shadow effect using utility function
            shadow_clip = create_shadow_text_clip(
                text=line_text,
                font=font,
                font_size=font_size,
                position=(line_x, y_position),
                start_time=page_start_time,
                end_time=page_end_time
            )
            all_clips.append(shadow_clip)
            
            # Add the base line using utility function
            positioned_line_clip = create_text_clip(
                text=line_text,
                font=font,
                font_size=font_size,
                color=color,
                position=(line_x, y_position),
                start_time=page_start_time,
                end_time=page_end_time
            )
            all_clips.append(positioned_line_clip)
            
            # Process each word for highlighting
            line_words = line["words"]
            
            # Calculate total width of characters up to each word
            # to determine highlighting positions
            word_positions = []
            current_position = 0
            
            for i, (word, _, _) in enumerate(line_words):
                # Get text up to current word
                if i == 0:
                    prefix = ""
                else:
                    prefix = " ".join([w[0] for w in line_words[:i]])
                    if not prefix.endswith(" "):
                        prefix += " "
                
                # Calculate position of this word
                if prefix:
                    prefix_clip = create_text_clip(prefix, font, font_size, color)
                    current_position = prefix_clip.w
                    prefix_clip.close()
                
                word_positions.append(current_position)
                
                # Add word length for next iteration
                word_clip = create_text_clip(word, font, font_size, color)
                word_width = word_clip.w
                word_clip.close()
                
                current_position += word_width + word_spacing
            
            # Create highlight clips for each word
            for i, ((word, start_time, end_time), word_pos) in enumerate(zip(line_words, word_positions)):
                # Add timing buffer to prevent overlaps
                buffer_time = style.get("timing_buffer", 0.05)  # 50ms default buffer
                
                # Adjust timing to prevent overlapping highlights
                if i > 0:
                    prev_end = line_words[i-1][2]
                    if start_time < prev_end + buffer_time:
                        start_time = prev_end + buffer_time
                
                # Create crossfade effect for smoother transitions
                crossfade_duration = style.get("crossfade_duration", 0.12)  # 120ms default crossfade
                
                # Calculate actual crossfade time (don't exceed word duration)
                word_duration = end_time - start_time
                actual_crossfade = min(crossfade_duration, word_duration * 0.4)
                
                # Adjust timings for crossfade effect
                crossfade_start = max(0, start_time - actual_crossfade/2)
                crossfade_end = min(total_duration, end_time + actual_crossfade/2)
                
                if use_box_highlighting:
                    # Create a boxed highlight
                    word_clip = create_text_clip(word, font, font_size, color)
                    word_width = word_clip.w
                    word_clip.close()
                    
                    # Create rounded box
                    margin_x = 15
                    margin_y = 10
                    box_width = int(word_width + min(margin_x * 1.5, max(10, word_width * 0.15)))
                    box_height = int(font_size * 1.25)
                    
                    box = create_rounded_rectangle_clip(
                        width=box_width,
                        height=box_height,
                        color=highlight_bg_color,
                        corner_radius=box_height // 2
                    )
                    
                    # Position box relative to line
                    box_x = line_x + word_pos - margin_x
                    box_y = y_position + font_size/4.0 - (box_height - int(font_size * 1.0))/2
                    
                    # Apply crossfade effect
                    box = box.set_position((int(box_x), int(box_y)))
                    box = box.set_start(crossfade_start).set_end(crossfade_end)
                    
                    # Add fade in/out for smoother appearance
                    box = apply_smooth_transitions(
                        clip=box,
                        word_duration=word_duration,
                        crossfade_duration=actual_crossfade,
                        min_word_duration_for_fade=0.2,
                        max_fade_duration=0.1
                    )
                    
                    all_clips.append(box)
                    
                    # Add highlighted word on top of box (shadow and text)
                    shadow_highlight = create_shadow_text_clip(
                        text=word,
                        font=font,
                        font_size=font_size,
                        position=(line_x + word_pos, y_position),
                        start_time=crossfade_start,
                        end_time=crossfade_end
                    )
                    all_clips.append(shadow_highlight)
                    
                    # Main highlighted text
                    highlight_word = create_text_clip(
                        text=word,
                        font=font,
                        font_size=font_size,
                        color="white",  # White text on colored box
                        position=(int(line_x + word_pos), int(y_position)),
                        start_time=crossfade_start,
                        end_time=crossfade_end
                    )
                    all_clips.append(highlight_word)
                else:
                    # Simple color highlighting (shadow and text)
                    shadow_highlight = create_shadow_text_clip(
                        text=word,
                        font=font,
                        font_size=font_size,
                        position=(line_x + word_pos, y_position),
                        start_time=crossfade_start,
                        end_time=crossfade_end
                    )
                    all_clips.append(shadow_highlight)
                    
                    # Main highlighted text
                    highlight_word = create_text_clip(
                        text=word,
                        font=font,
                        font_size=font_size,
                        color=highlight_color,
                        position=(int(line_x + word_pos), int(y_position)),
                        start_time=crossfade_start,
                        end_time=crossfade_end
                    )
                    
                    # Add fade in/out for smoother color transitions
                    highlight_word = apply_smooth_transitions(
                        clip=highlight_word,
                        word_duration=word_duration,
                        crossfade_duration=actual_crossfade,
                        min_word_duration_for_fade=0.2,
                        max_fade_duration=0.08
                    )
                    
                    all_clips.append(highlight_word)
    
    return all_clips

def add_karaoke_captions_to_video(
    video: VideoFileClip,
    word_timings: List[Tuple[str, float, float]],
    style: Dict[str, Any]
) -> VideoFileClip:
    """
    Add TikTok-style karaoke captions to a video.
    
    Args:
        video: The video clip to add captions to
        word_timings: List of (word, start_time, end_time) tuples
        style: Dictionary of style parameters
            
    Returns:
        VideoFileClip: Video with karaoke captions
    """
    if not word_timings:
        return video
        
    # Get video dimensions
    width, height = video.size
    
    # Adjust styling based on video format
    if width < height:  # Portrait format (TikTok-style)
        # For portrait format, use larger font and higher bottom padding
        if style.get("font_size") is None:
            # Default font size for portrait is larger
            style["font_size"] = 70
            
        # Default bottom padding for portrait
        if style.get("bottom_padding") is None:
            style["bottom_padding"] = 150
    elif width > height:  # Landscape format
        current_font_size = style.get("font_size", 60)
        
        # Apply a consistent scaling factor for landscape
        landscape_scale_factor = 0.7
            
        # Apply the scaling
        style["font_size"] = max(40, int(current_font_size * landscape_scale_factor))
            
        # Adjust bottom padding for landscape format
        if style.get("bottom_padding") in [80, 150]:
            # Use a smaller bottom padding for landscape format
            style["bottom_padding"] = 60
            
        # Adjust max line width to use more horizontal space
        if "max_line_width_ratio" not in style:
            # Use more of the horizontal space in landscape mode
            style["max_line_width_ratio"] = 0.95  # 95% of video width
    
    # Use the new unified caption method
    caption_clips = create_unified_karaoke_captions(
        word_timings=word_timings,
        video_width=width,
        video_height=height,
        style=style
    )
    
    # Combine with the video
    if caption_clips:
        return CompositeVideoClip(
            [video] + caption_clips,
            size=video.size
        )
    else:
        return video

def add_captions_to_video(
    video_path: str,
    captions: List[Dict],
    output_path: str,
    position: str = "bottom",
    y_offset: int = 0,
    font: str = None,
    font_size: int = 40,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 2,
    bg_color: str = None,
    bg_opacity: float = 0.6,
    highlight_words: bool = False,
    highlight_color: str = "#00ff00",
    highlight_bg_color: str = None,
    highlight_use_box: bool = False,
    box_color: str = "#ffffff",
    box_opacity: float = 0.8,
    corner_radius: int = 10,
    padding: int = 20,
    align: str = "center",
    size_multiplier: float = 1.0,
) -> None:
    """Add captions to a video file."""
    # Load the video
    video = VideoFileClip(video_path)
    
    # Create caption clips
    caption_clips = []
    for cap in captions:
        caption = create_caption_clip(
            text=cap["text"],
            duration=cap["duration"],
            video_height=video.size[1],
            video_width=video.size[0],
            font=font,
            font_size=font_size,
            color=color,
            position=position,
            debug=False
        )
        caption_clips.append(caption)

    # Create final composite
    final_video = CompositeVideoClip([video] + caption_clips)

    # Write the output file
    final_video.write_videofile(output_path)

def create_rounded_rectangle_clip(width, height, color, corner_radius=15):
    """
    Create a clip with a rounded rectangle shape.
    
    Args:
        width: Width of the rectangle
        height: Height of the rectangle
        color: RGB color tuple (r,g,b)
        corner_radius: Radius of the rounded corners
        
    Returns:
        ImageClip: A clip with the rounded rectangle shape
    """
    # Ensure corner radius isn't too large for the rectangle
    corner_radius = min(corner_radius, height // 2, width // 2)
    
    # Create a transparent RGBA image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a rounded rectangle
    draw.rounded_rectangle([(0, 0), (width-1, height-1)], 
                          fill=(*color, 255),  # Add alpha channel
                          radius=corner_radius)
    
    # Convert to numpy array and create ImageClip
    img_array = np.array(img)
    return ImageClip(img_array) 