"""
Handles the generation and overlay of captions on videos.
Features:
- Creating caption overlays with customizable styles
- Timing caption display with voiceover
- Supporting multiple caption formats and positions
- TikTok-style karaoke captions with word-level highlighting
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
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

    print(f"\nCreating caption with text: {text}")
    print(f"Using font: {font} at size: {font_size}")
    
    try:
        test_clip = TextClip("Test", font=font)
        test_clip.close()
        print("Font test successful")
    except Exception as e:
        print(f"Font test failed: {e}")
        print("Falling back to default font")
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
    
    print(f"Created text clip with size: {txt_clip.size}")
    
    # Increase bottom margin for better placement
    margin = 50  # Increased margin for better spacing from bottom
    
    if position == "bottom":
        y_position = video_height - txt_clip.size[1] - margin  # Place above bottom margin
    else:  # top
        y_position = margin  # Place below top margin
    
    print(f"Calculated y_position: {y_position}")
    
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
        print(f"Saved caption test frame to: {test_frame_path}")
        
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
            
    Returns:
        List[TextClip]: List of text clips for each word and background
    """
    if not word_timings:
        return []
    
    # Clean any <SPLIT> markers from word timings
    cleaned_word_timings = []
    for word, start_time, end_time in word_timings:
        # Remove <SPLIT> markers from words
        cleaned_word = word.replace("<SPLIT>", "").strip()
        if cleaned_word:  # Only add if the word is not empty after cleaning
            cleaned_word_timings.append((cleaned_word, start_time, end_time))
    
    # Use cleaned word timings for the rest of the function
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
    
    # Ensure highlight_bg_color is in RGB format
    highlight_bg_color_value = style.get("highlight_bg_color", "white")
    # Convert string color names to RGB tuples
    if isinstance(highlight_bg_color_value, str):
        if highlight_bg_color_value.lower() == "white":
            highlight_bg_color = (255, 255, 255)
        elif highlight_bg_color_value.lower() == "black":
            highlight_bg_color = (0, 0, 0)
        elif highlight_bg_color_value.lower() == "red":
            highlight_bg_color = (255, 0, 0)
        elif highlight_bg_color_value.lower() == "green":
            highlight_bg_color = (0, 255, 0)
        elif highlight_bg_color_value.lower() == "blue":
            highlight_bg_color = (0, 0, 255)
        elif highlight_bg_color_value.lower() == "yellow":
            highlight_bg_color = (255, 255, 0)
        elif highlight_bg_color_value.startswith("#"):
            # Convert hex color to RGB
            h = highlight_bg_color_value.lstrip('#')
            highlight_bg_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        else:
            # Default to white for unknown colors
            highlight_bg_color = (255, 255, 255)
    else:
        # Keep as is (assuming it's already in RGB format)
        highlight_bg_color = highlight_bg_color_value
    
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
    max_line_width = int(video_width * 0.9)
    bg_margin = 20
    word_spacing = 10
    
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
        temp_clip = TextClip(word + " ", font=font, fontsize=font_size, color=color)
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
    base_y = int(video_height - bottom_padding - (visible_lines * line_height))
    
    # Create clips for all words
    clips = []
    
    # For each page
    for page_idx, (page, (page_start, page_end)) in enumerate(zip(pages, page_transitions)):
        # For each line in the page
        for line_idx, line in enumerate(page):
            # Calculate starting x position for this line (center-aligned)
            line_width = sum(word[3] for word in line) + word_spacing * (len(line) - 1)
            start_x = (video_width - line_width) / 2
            
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
                            word + " ",
                            font=font,
                            fontsize=font_size,
                            color="black"
                        )
                        shadow_before = shadow_before.set_position((int(current_x + 2), int(y_position + 2)))
                        shadow_before = shadow_before.set_start(page_start).set_end(start_time)
                        clips.append(shadow_before)
                        
                        # Then create white text on top
                        before_txt = TextClip(
                            word + " ",
                            font=font,
                            fontsize=font_size,
                            color=color
                        )
                        before_txt = before_txt.set_position((int(current_x), int(y_position)))
                        before_txt = before_txt.set_start(page_start).set_end(start_time)
                        clips.append(before_txt)
                    
                    # After highlight (if applicable)
                    if end_time < page_end:
                        # Create shadow with slight offset for cleaner look
                        shadow_after = TextClip(
                            word + " ",
                            font=font,
                            fontsize=font_size,
                            color="black"
                        )
                        shadow_after = shadow_after.set_position((int(current_x + 2), int(y_position + 2)))
                        shadow_after = shadow_after.set_start(end_time).set_end(page_end)
                        clips.append(shadow_after)
                        
                        # Then create white text on top
                        after_txt = TextClip(
                            word + " ",
                            font=font,
                            fontsize=font_size,
                            color=color
                        )
                        after_txt = after_txt.set_position((int(current_x), int(y_position)))
                        after_txt = after_txt.set_start(end_time).set_end(page_end)
                        clips.append(after_txt)
                    
                    # Highlighted word clip - only visible during its timing
                    # Calculate overlap with page
                    overlap_start = max(start_time, page_start)
                    overlap_end = min(end_time, page_end)
                    
                    # TikTok boxed highlighting style
                    if highlight_use_box:
                        try:
                            if isinstance(highlight_bg_color, tuple) and len(highlight_bg_color) >= 3:
                                box_color = highlight_bg_color[:3]
                            else:
                                box_color = (239, 68, 68)
                            
                            margin_x = 15
                            margin_y = 10
                            box_width = int(word_width + 1.8 * margin_x)
                            box_height = int(font_size * 1.25)
                            
                            box = create_rounded_rectangle_clip(
                                width=box_width,
                                height=box_height,
                                color=box_color,
                                corner_radius=box_height // 2
                            )
                            
                            box = box.set_position((current_x - margin_x, y_position + font_size/4.0 - (box_height - int(font_size * 1.0))/2))
                            box = box.set_start(overlap_start).set_end(overlap_end)
                            
                            shadow_text = TextClip(word, 
                                                 fontsize=font_size, 
                                                 font=font, 
                                                 color="black")
                            
                            shadow_text = shadow_text.set_position((current_x + 2, y_position + 2))
                            shadow_text = shadow_text.set_start(overlap_start).set_end(overlap_end)
                            
                            highlighted_text = TextClip(word, 
                                                       fontsize=font_size, 
                                                       font=font, 
                                                       color="white")
                            
                            highlighted_text = highlighted_text.set_position((current_x, y_position))
                            highlighted_text = highlighted_text.set_start(overlap_start).set_end(overlap_end)
                            
                            clips.append(box)
                            clips.append(shadow_text)
                            clips.append(highlighted_text)
                        except Exception as e:
                            shadow_fallback = TextClip(word, 
                                                     fontsize=font_size, 
                                                     font=font, 
                                                     color="black")
                            
                            shadow_fallback = shadow_fallback.set_position((current_x + 2, y_position + 2))
                            shadow_fallback = shadow_fallback.set_start(overlap_start).set_end(overlap_end)
                            clips.append(shadow_fallback)
                            
                            highlighted_text = TextClip(word, 
                                                       fontsize=font_size, 
                                                       font=font, 
                                                       color=highlight_color)
                            
                            highlighted_text = highlighted_text.set_position((current_x, y_position))
                            highlighted_text = highlighted_text.set_start(overlap_start).set_end(overlap_end)
                            clips.append(highlighted_text)
                    else:
                        # Standard highlighting (just change text color)
                        # Create shadow with slight offset for cleaner look
                        shadow_highlight = TextClip(
                            word + " ",
                            font=font,
                            fontsize=font_size,
                            color="black"
                        )
                        shadow_highlight = shadow_highlight.set_position((int(current_x + 2), int(y_position + 2)))
                        shadow_highlight = shadow_highlight.set_start(overlap_start).set_end(overlap_end)
                        clips.append(shadow_highlight)
                        
                        # Then create highlighted text on top
                        highlight_txt = TextClip(
                            word + " ",
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

def add_karaoke_captions_to_video(
    video: VideoFileClip,
    word_timings: List[Tuple[str, float, float]],
    style: Dict[str, Any]
) -> VideoFileClip:
    """
    Add TikTok-style karaoke captions to a video.
    
    Args:
        video: The base video clip
        word_timings: List of (word, start_time, end_time) tuples
        style: Dictionary of style parameters
            
    Returns:
        VideoFileClip: Video with karaoke captions
    """
    if not word_timings:
        return video
    
    # Get video dimensions
    width, height = video.size
    
    # Create karaoke caption clips
    caption_clips = create_karaoke_captions(
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