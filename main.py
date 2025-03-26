"""
Main entry point for the Video Pipeline application.
A multi-scene media generation pipeline supporting both video and image generation.
"""

import argparse
import os
import shutil
from pathlib import Path
from core.config import Config, ConfigError
from media.replicate_client import ReplicateRayClient
from media.sdxl_client import SDXLClient
from media.gemini_client import GeminiClient
from audio.tts import TextToSpeech
from video.captions import create_caption_clip, add_captions_to_video
from video.assembler import VideoAssembler
from video.scene_builder import SceneBuilder
from video.parse_script import parse_two_part_script
from moviepy.editor import ColorClip
from core.logging_config import configure_logging
import logging

def clear_output_directories(config, preserve_videos=True):
    """
    Clear all output directories except for videos directory if preserve_videos is True.
    
    Args:
        config: Config instance containing output directory paths
        preserve_videos: Whether to preserve the videos directory
    """
    directories_to_clear = [
        config.image_output_dir,
        config.audio_output_dir,
        config.captions_output_dir,
        config.temp_output_dir
    ]
    
    # Add replicate_segments directory if it exists
    replicate_segments_dir = config.base_output_dir / "replicate_segments"
    if replicate_segments_dir.exists():
        directories_to_clear.append(replicate_segments_dir)
    
    # Add videos directory if not preserving
    if not preserve_videos:
        directories_to_clear.append(config.video_output_dir)
    
    print("\n=== Clearing Output Directories ===")
    for directory in directories_to_clear:
        if directory.exists():
            print(f"Clearing: {directory}")
            try:
                # Remove all files in directory
                for item in directory.glob("*"):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print(f"✓ Cleared {directory}")
            except Exception as e:
                print(f"! Error clearing {directory}: {e}")
    
    if preserve_videos:
        print(f"✓ Preserved videos directory: {config.video_output_dir}")
    print("=== Cleanup Complete ===\n")

def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate AI videos/images from text prompts using Replicate models and ElevenLabs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate video using Ray model:
  python main.py --media-type video --script-file script.txt --output-file output.mp4

  # Generate animated images using SDXL model:
  python main.py --media-type image --script-file script.txt --output-file output.mp4 --image-model sdxl

  # Generate animated images using Google Gemini (free tier):
  python main.py --media-type image --script-file script.txt --output-file output.mp4 --image-model gemini

  # Customize animation style for images:
  python main.py --media-type image --script-file script.txt --animation-style ken_burns

  # Generate TikTok-style video with karaoke captions:
  python main.py --media-type image --script-file script.txt --video-format short --style tiktok_neon
  
  # Generate YouTube-style video with karaoke captions:
  python main.py --media-type image --script-file script.txt --video-format landscape --style tiktok_neon
  
  # Generate YouTube-optimized video with high resolution:
  python main.py --media-type image --script-file script.txt --video-format landscape --youtube-optimized
  
  # Generate faster low-resolution video for testing:
  python main.py --media-type image --script-file script.txt --no-youtube-optimized
  
  # Available karaoke caption presets:
  # - tiktok_neon: Cyan highlighting with bold text (best for most videos)
  # - tiktok_bold: Bold text with highlighted words in yellow boxes (authentic TikTok style)
  # - tiktok_yellow: Bold text with highlighted words in vibrant yellow flag style
  # - tiktok_minimal: Clean minimal style with pink highlighting (subtle)
  # - tiktok_boxed: Bold text with highlighted words in red boxes (authentic TikTok style)
  
  # Custom styling for karaoke captions:
  python main.py --media-type image --script-file script.txt --video-format short --highlight-color "#00ffff" --stroke-width 3 --font-size 70
  
  # Configure caption paging system for longer narrations:
  python main.py --media-type image --script-file script.txt --video-format short --style tiktok_neon --visible-lines 3 --bottom-padding 100
  
  # Control karaoke caption timing adjustments:
  python main.py --media-type image --script-file script.txt --video-format short --style tiktok_neon --use-timing-adjustment true
  
  # Use UnrealSpeech for TTS instead of ElevenLabs (always uses 'Daniel' voice):
  python main.py --media-type video --script-file script.txt --tts-provider unrealspeech
  
  # Clear temporary files before generating new video:
  python main.py --media-type image --script-file script.txt --clear
"""
    )
    
    parser.add_argument(
        "--script-file",
        type=str,
        required=True,
        help="Path to script file (two lines per scene: visual prompt and narration)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="final_video.mp4",
        help="Output path for the final video"
    )
    
    parser.add_argument(
        "--media-type",
        type=str,
        choices=["video", "image"],
        default="video",
        help="Type of media to generate (video or image)"
    )
    
    parser.add_argument(
        "--image-model",
        type=str,
        choices=["sdxl", "gemini"],
        default="sdxl",
        help="Model to use for image generation (sdxl from Replicate or gemini from Google)"
    )
    
    parser.add_argument(
        "--animation-style",
        type=str,
        choices=["ken_burns", "zoom", "pan"],
        default="ken_burns",
        help="Animation style for image mode"
    )
    
    parser.add_argument(
        "--video-format",
        type=str,
        choices=["landscape", "short", "square"],
        default="landscape",
        help="Video format/aspect ratio (landscape=16:9, short=9:16, square=1:1)"
    )
    
    parser.add_argument(
        "--use-timing-adjustment",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to apply timing adjustments for karaoke captions (0.2s for portrait, 0.25s for landscape)"
    )
    
    parser.add_argument(
        "--youtube-optimized",
        action="store_true",
        dest="youtube_optimized",
        default=True,
        help="Optimize video resolution and bitrate for YouTube standards (default: enabled)"
    )
    
    parser.add_argument(
        "--no-youtube-optimized",
        action="store_false",
        dest="youtube_optimized",
        help="Disable YouTube optimization for faster generation and testing"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all output directories (except videos) before processing"
    )
    
    # Style options
    style_group = parser.add_argument_group("Style options")
    style_group.add_argument(
        "--style",
        type=str,
        choices=["default", "tiktok_neon", "tiktok_bold", "tiktok_minimal", "tiktok_boxed", "tiktok_yellow"],
        default="default",
        help="Preset style for captions (overrides individual style settings)"
    )
    
    style_group.add_argument(
        "--font",
        type=str,
        default="Arial-Bold",
        help="Font to use for captions"
    )
    
    style_group.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="Font size for captions"
    )
    
    style_group.add_argument(
        "--color",
        type=str,
        default="white",
        help="Text color for captions"
    )
    
    # Add karaoke caption style options
    style_group.add_argument(
        "--highlight-color",
        type=str,
        default="#ff5c5c",
        help="Color for highlighted words in karaoke captions (used in all video formats)"
    )
    
    style_group.add_argument(
        "--stroke-color",
        type=str,
        default="black",
        help="Outline color for karaoke captions"
    )
    
    style_group.add_argument(
        "--stroke-width",
        type=int,
        default=2,
        help="Outline width for karaoke captions (default: 2)"
    )
    
    style_group.add_argument(
        "--use-background",
        action="store_true",
        help="Add semi-transparent background behind captions"
    )
    
    style_group.add_argument(
        "--highlight-bg-color",
        type=str,
        default="white",
        help="Background color for highlighted words in boxed style (default: white)"
    )
    
    style_group.add_argument(
        "--highlight-use-box",
        action="store_true",
        default=False,
        help="Use box highlighting style for words (like authentic TikTok captions)"
    )
    
    # Add rolling caption window parameters
    style_group.add_argument(
        "--visible-lines",
        type=int,
        default=2,
        help="Number of caption lines to show at once (default: 2)"
    )
    
    style_group.add_argument(
        "--bottom-padding",
        type=int,
        default=80,
        help="Padding from bottom of screen for captions (in pixels)"
    )
    
    # TTS options
    tts_group = parser.add_argument_group("Text-to-Speech options")
    tts_group.add_argument(
        "--tts-provider",
        type=str,
        choices=["elevenlabs", "unrealspeech"],
        default="elevenlabs",
        help="TTS provider to use (elevenlabs or unrealspeech). Always uses 'Daniel' voice."
    )
    
    tts_group.add_argument(
        "--disable-smart-voice",
        action="store_true",
        help="Disable smart voice parameter optimization (voice is always 'Daniel')"
    )
    
    # Add verbose flag for debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and debugging output"
    )
    
    return parser

def get_style_preset(preset_name: str) -> dict:
    """Get a predefined style preset configuration.
    
    Args:
        preset_name: Name of the preset style
        
    Returns:
        dict: Style configuration dictionary
    """
    presets = {
        "default": {
            "font": "Arial-Bold",
            "font_size": 24,
            "color": "white",
            "highlight_color": "#ff5c5c",
            "stroke_color": "black",
            "stroke_width": 2,
            "use_background": False,
            "bg_color": (0, 0, 0, 128),  # Semi-transparent black
            "visible_lines": 2,
            "bottom_padding": 80
        },
        "tiktok_neon": {
            "font": "Arial-Bold",
            "font_size": 70,
            "color": "white",
            "highlight_color": "#00ffff",  # Cyan neon
            "stroke_color": "black",
            "stroke_width": 3,
            "use_background": False,
            "bg_color": (0, 0, 0, 128),  # Semi-transparent black
            "visible_lines": 2,
            "bottom_padding": 80
        },
        "tiktok_bold": {
            "font": "Arial-Black",  # Bolder font for TikTok style
            "font_size": 80,  # Larger font size for authentic TikTok look
            "color": "white",  # Regular text color
            "highlight_color": "white",  # Text color for highlighted words
            "highlight_bg_color": (255, 191, 0),  # Golden yellow color for highlighted words
            "stroke_color": "black",  # Used for shadow effect
            "stroke_width": 3,  # Add visible border around text
            "use_background": False,  # No background for non-highlighted words
            "bg_color": (0, 0, 0),  # Black (RGB tuple)
            "highlight_use_box": True,  # Use box highlighting style
            "visible_lines": 3,  # Show more lines for complex content
            "bottom_padding": 120,  # Increased bottom padding for better positioning
            "max_line_width_ratio": 0.85,  # Keep good margins
            "preserve_spacing": True,  # Ensure word spacing is preserved
            "consistent_positioning": True,  # Ensure caption positioning is consistent
            "base_visible_lines": 3,  # Fixed number of lines for baseline positioning
            "word_spacing": 15,  # Extra space between words for clarity
            "timing_buffer": 0.08,  # 80ms buffer between highlighted words
            "crossfade_duration": 0.15  # 150ms crossfade for smooth transitions
        },
        "tiktok_yellow": {
            "font": "Arial-Black",  # Bolder font for TikTok style
            "font_size": 80,  # Larger font size for authentic TikTok look
            "color": "white",  # Regular text color
            "highlight_color": "white",  # Text color for highlighted words
            "highlight_bg_color": (255, 215, 0),  # Vibrant yellow color for highlighted words - matches image
            "stroke_color": "black",  # Used for shadow effect
            "stroke_width": 3,  # Add visible border around text
            "use_background": False,  # No background for non-highlighted words
            "bg_color": (0, 0, 0),  # Black (RGB tuple)
            "highlight_use_box": True,  # Use box highlighting style
            "visible_lines": 3,  # Show more lines for complex content
            "bottom_padding": 120,  # Increased bottom padding for better positioning
            "max_line_width_ratio": 0.85,  # Keep good margins
            "preserve_spacing": True,  # Ensure word spacing is preserved
            "consistent_positioning": True,  # Ensure caption positioning is consistent
            "base_visible_lines": 3,  # Fixed number of lines for baseline positioning
            "word_spacing": 15,  # Extra space between words for clarity
            "timing_buffer": 0.08,  # 80ms buffer between highlighted words
            "crossfade_duration": 0.15  # 150ms crossfade for smooth transitions
        },
        "tiktok_minimal": {
            "font": "Arial",
            "font_size": 60,
            "color": "white",
            "highlight_color": "#ff5c5c",  # Pink/red
            "stroke_color": "black",
            "stroke_width": 3,  # Add visible border around text
            "use_background": False,
            "bg_color": (0, 0, 0, 128),  # Semi-transparent black
            "visible_lines": 2,
            "bottom_padding": 80,
            "crossfade_duration": 0.12  # 120ms crossfade for smooth transitions
        },
        "tiktok_boxed": {
            "font": "Arial-Black",  # Bolder font for TikTok style
            "font_size": 100,  # Larger font size for authentic TikTok look
            "color": "white",  # Regular text color
            "highlight_color": "white",  # Text color for highlighted words
            "highlight_bg_color": (255, 48, 64),  # More precise TikTok red
            "stroke_color": "black",  # Used for shadow effect
            "stroke_width": 3,  # Add visible border around text
            "use_background": False,  # No background for non-highlighted words
            "bg_color": (0, 0, 0),  # Black (RGB tuple)
            "highlight_use_box": True,  # Use box highlighting style
            "visible_lines": 2,
            "bottom_padding": 150,  # Increased bottom padding for better positioning
            "timing_adjustment": 1.0,  # 1 second lead time for highlighting words
            "crossfade_duration": 0.15  # 150ms crossfade for smooth transitions
        }
    }
    
    return presets.get(preset_name, presets["default"])

def process_script(args: argparse.Namespace, scene_builder: SceneBuilder, assembler: VideoAssembler, style_config: dict) -> None:
    """Process the script file and generate the media using the configured client.
    
    Args:
        args: Command line arguments containing script_file, output_file, and style options
        scene_builder: SceneBuilder instance
        assembler: VideoAssembler instance
        style_config: Style configuration for captions
    """
    try:
        # Parse the two-part script file
        try:
            scene_pairs = parse_two_part_script(args.script_file)
            print(f"\nFound {len(scene_pairs)} scenes to process")
        except FileNotFoundError:
            raise FileNotFoundError(f"Script file not found: {args.script_file}")
        except ValueError as e:
            raise ValueError(f"Script file error: {str(e)}")
        
        scenes = []
        scene_lookup = {}  # Store scenes by index for continuation
        
        # Process each scene pair
        for i, (visual_prompt, narration_text, metadata) in enumerate(scene_pairs, 1):
            print(f"\n=== Processing Scene {i}/{len(scene_pairs)} ===")
            print(f"Visual prompt: {visual_prompt}")
            print(f"Narration: {narration_text}")
            
            if metadata.get("continue_from") is not None:
                prev_idx = metadata["continue_from"]
                print(f"Continuing from scene {prev_idx}")
            
            # Generate scene (media + audio + captions)
            scene = scene_builder.build_scene(
                visual_text=visual_prompt,
                narration_text=narration_text,
                style_config=style_config,
                continue_from_scene=metadata.get('continue_from'),
                video_format=args.video_format,
                animation_style=args.animation_style,
                highlight_color=args.highlight_color,
                stroke_color=args.stroke_color,
                stroke_width=args.stroke_width,
                use_background=args.use_background,
                highlight_use_box=args.highlight_use_box,
                highlight_bg_color=args.highlight_bg_color,
                visible_lines=args.visible_lines,
                bottom_padding=args.bottom_padding
            )
            
            print(f"Scene {i} generated successfully:")
            print(f"- Media: {scene['media_path']}")
            if len(scene['media_paths']) > 1:
                print(f"- Additional media: {scene['media_paths'][1:]}")
            print(f"- Audio: {scene['audio_path']}")
            print(f"- Duration: {scene['duration']:.2f}s")
            if metadata.get("continue_from") is not None:
                print(f"- Continued from: Scene {metadata['continue_from']}")
            
            # Verify scene components
            if scene_builder.verify_output(scene):
                scenes.append(scene)
                scene_lookup[i] = scene  # Store for potential continuation
            else:
                raise ValueError(f"Scene {i} verification failed")
        
        # Assemble final video
        print("\nAssembling final video...")
        final_path = assembler.assemble_scenes(
            scenes=scenes,
            output_path=args.output_file,
            fps=24,
            optimize_for_youtube=args.youtube_optimized
        )
        
        print("\n=== Pipeline Complete ===")
        print(f"All segments generated and final video compiled at: {final_path}")
        print("\nVideo details:")
        print(f"- Total scenes: {len(scenes)}")
        print(f"- Output path: {final_path}")
        
        # Note: To upload this video to YouTube, use the integrations/youtube/uploader.py script
        print("\nTo upload this video to YouTube, use the integrations/youtube/uploader.py script:")
        if args.video_format == "short":
            print(f"python integrations/youtube/uploader.py --video-path {final_path} --metadata youtube_metadata.txt --shorts")
        else:
            print(f"python integrations/youtube/uploader.py --video-path {final_path} --metadata youtube_metadata.txt")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("- script.txt exists with valid two-part format (two lines per scene)")
        print("- REPLICATE_API_TOKEN is set in .env")
        print("- All required packages are installed")
        print("\nExample script.txt format:")
        print("# Scene 1 (Visual)")
        print("A futuristic city with flying cars and neon lights")
        print("# Scene 1 (Narration)")
        print("Welcome to the future, where innovation never sleeps")
        print("# Scene 2 (Visual) [continue=1]")
        print("The same city transforming as day turns to night")
        raise

def main():
    """Main entry point for the media generation pipeline."""
    try:
        # Configure logging with debug mode based on verbosity
        args = create_parser().parse_args()
        configure_logging(debug_mode=args.verbose)
        
        # Load and validate configuration
        config = Config()
        
        # Clear output directories if requested
        if args.clear:
            clear_output_directories(config, preserve_videos=True)
        
        # Initialize media client based on type
        if args.media_type == "video":
            media_client = ReplicateRayClient(
                api_token=config.replicate_token,
                dev_mode=config.is_dev_mode
            )
            print("\nUsing Ray model for video generation")
        else:  # image
            if args.image_model == "gemini":
                media_client = GeminiClient(
                    api_key=config.gemini_api_key,
                    dev_mode=config.is_dev_mode
                )
                print("\nUsing Google Gemini model for image generation (free tier)")
            else:  # sdxl
                media_client = SDXLClient(
                    api_token=config.replicate_token,
                    dev_mode=config.is_dev_mode
                )
                print("\nUsing SDXL model for image generation")
        
        # Initialize other components
        tts = TextToSpeech(
            api_key=config.elevenlabs_token if args.tts_provider == "elevenlabs" else config.unrealspeech_token,
            use_smart_voice=not args.disable_smart_voice,
            provider=args.tts_provider
        )
        print(f"\nUsing {args.tts_provider.capitalize()} TTS provider with 'Daniel' voice")
        print(f"Smart voice parameter optimization: {'Disabled' if args.disable_smart_voice else 'Enabled'}")
        
        scene_builder = SceneBuilder(
            media_client=media_client,
            tts_client=tts,
            media_type=args.media_type
        )
        assembler = VideoAssembler()
        
        # Apply style preset if specified
        if args.style != "default":
            preset = get_style_preset(args.style)
            # Override command line arguments with preset values
            args.font = preset["font"]
            args.font_size = preset["font_size"]
            args.color = preset["color"]
            args.highlight_color = preset["highlight_color"]
            if "highlight_bg_color" in preset:
                args.highlight_bg_color = preset["highlight_bg_color"]
            args.stroke_color = preset["stroke_color"]
            args.stroke_width = preset["stroke_width"]
            args.use_background = preset["use_background"]
            if "highlight_use_box" in preset:
                args.highlight_use_box = preset["highlight_use_box"]
            args.visible_lines = preset["visible_lines"]
            args.bottom_padding = preset["bottom_padding"]
        
        # Set up style config from args
        style_config = {
            "aspect_ratio": "16:9" if args.video_format == "landscape" else "9:16" if args.video_format == "short" else "1:1",
            "font": args.font,
            "font_size": args.font_size,
            "color": args.color,
            "animation_style": args.animation_style,
            "caption_style": "karaoke",
            "highlight_color": args.highlight_color,
            "highlight_bg_color": args.highlight_bg_color,
            "stroke_color": args.stroke_color,
            "stroke_width": args.stroke_width,
            "use_background": args.use_background,
            "highlight_use_box": args.highlight_use_box,
            "visible_lines": args.visible_lines,
            "bottom_padding": args.bottom_padding,
            "youtube_optimized": args.youtube_optimized
        }
        
        print("\n=== Starting Media Generation Pipeline ===")
        if config.is_dev_mode and args.media_type == "video":
            print("Running in DEVELOPMENT MODE - Using cached videos only")
        print(f"Media type: {args.media_type}")
        print(f"TTS provider: {args.tts_provider}")
        print(f"Smart voice optimization: {'disabled' if args.disable_smart_voice else 'enabled'}")
        print(f"Video format: {args.video_format} ({style_config['aspect_ratio']})")
        if args.media_type == "image":
            print(f"Animation style: {args.animation_style}")
        print(f"Script file: {args.script_file}")
        print(f"Output file: {args.output_file}")
        if args.style != "default":
            print(f"Caption style preset: {args.style}")
        print(f"Style config: {style_config}")
        
        # Process script and generate video
        process_script(args, scene_builder, assembler, style_config)
        
    except ConfigError as e:
        print(f"\nConfiguration error: {str(e)}")
        print("\nPlease ensure:")
        print("- REPLICATE_API_TOKEN is set in .env")
        print("- ELEVENLABS_API_KEY is set in .env (for TTS)")
        exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 