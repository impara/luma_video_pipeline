"""
Main entry point for the Luma Video Pipeline application.
A multi-scene media generation pipeline supporting both video and image generation.
"""

import argparse
from pathlib import Path
from config import Config, ConfigError
from replicate_client import ReplicateRayClient
from sdxl_client import SDXLClient
from tts import TextToSpeech
from captions import create_caption_clip, add_captions_to_video
from assembler import VideoAssembler
from scene_builder import SceneBuilder
from parse_script import parse_two_part_script
from moviepy.editor import ColorClip

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
  python main.py --media-type image --script-file script.txt --output-file output.mp4

  # Customize animation style for images:
  python main.py --media-type image --script-file script.txt --animation-style ken_burns

  # Generate TikTok-style video with karaoke captions:
  python main.py --media-type image --script-file script.txt --video-format short --style tiktok_neon
  
  # Available TikTok-style caption presets:
  # - tiktok_neon: Cyan highlighting with bold text (best for most videos)
  # - tiktok_bold: Bold text with highlighted words in yellow boxes (authentic TikTok style)
  # - tiktok_minimal: Clean minimal style with pink highlighting (subtle)
  # - tiktok_boxed: Bold text with highlighted words in red boxes (authentic TikTok style)
  
  # Custom styling for karaoke captions:
  python main.py --media-type image --script-file script.txt --video-format short --highlight-color "#00ffff" --stroke-width 3 --font-size 70
  
  # Configure caption paging system for longer narrations:
  python main.py --media-type image --script-file script.txt --video-format short --style tiktok_neon --visible-lines 3 --bottom-padding 100
  
  # Use UnrealSpeech for TTS instead of ElevenLabs (always uses 'Daniel' voice):
  python main.py --media-type video --script-file script.txt --tts-provider unrealspeech
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
    
    # Style options
    style_group = parser.add_argument_group("Style options")
    style_group.add_argument(
        "--style",
        type=str,
        choices=["default", "tiktok_neon", "tiktok_bold", "tiktok_minimal", "tiktok_boxed"],
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
        help="Color for highlighted words in karaoke captions (used in short format)"
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
            "font_size": 100,  # Larger font size for authentic TikTok look
            "color": "white",  # Regular text color
            "highlight_color": "white",  # Text color for highlighted words
            "highlight_bg_color": (255, 191, 0),  # Golden yellow color for highlighted words
            "stroke_color": "black",  # Used for shadow effect
            "stroke_width": 0,  # Not used with our new shadow implementation
            "use_background": False,  # No background for non-highlighted words
            "bg_color": (0, 0, 0),  # Black (RGB tuple)
            "highlight_use_box": True,  # Use box highlighting style
            "visible_lines": 2,
            "bottom_padding": 150  # Increased bottom padding for better positioning
        },
        "tiktok_minimal": {
            "font": "Arial",
            "font_size": 60,
            "color": "white",
            "highlight_color": "#ff5c5c",  # Pink/red
            "stroke_color": "black",
            "stroke_width": 2,
            "use_background": False,
            "bg_color": (0, 0, 0, 128),  # Semi-transparent black
            "visible_lines": 2,
            "bottom_padding": 80
        },
        "tiktok_boxed": {
            "font": "Arial-Black",  # Bolder font for TikTok style
            "font_size": 100,  # Larger font size for authentic TikTok look
            "color": "white",  # Regular text color
            "highlight_color": "white",  # Text color for highlighted words
            "highlight_bg_color": (255, 48, 64),  # More precise TikTok red
            "stroke_color": "black",  # Used for shadow effect
            "stroke_width": 0,  # Not used with our new shadow implementation
            "use_background": False,  # No background for non-highlighted words
            "bg_color": (0, 0, 0),  # Black (RGB tuple)
            "highlight_use_box": True,  # Use box highlighting style
            "visible_lines": 2,
            "bottom_padding": 150  # Increased bottom padding for better positioning
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
            
            # Check if this scene continues from a previous one
            continue_from = None
            if metadata.get('continue_from') and args.media_type == "video":
                prev_idx = metadata['continue_from']
                if prev_idx in scene_lookup:
                    continue_from = scene_lookup[prev_idx]
                    print(f"Continuing from scene {prev_idx}")
            
            # Generate scene (media + audio + captions)
            scene = scene_builder.build_scene(
                visual_text=visual_prompt,
                narration_text=narration_text,
                style_config=style_config,
                continue_from_scene=continue_from,
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
            if continue_from:
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
            fps=24
        )
        
        print("\n=== Pipeline Complete ===")
        print(f"All segments generated and final video compiled at: {final_path}")
        print("\nVideo details:")
        print(f"- Total scenes: {len(scenes)}")
        print(f"- Output path: {final_path}")
        print("\nScene breakdown:")
        for i, (visual, narration, metadata) in enumerate(scene_pairs, 1):
            print(f"\nScene {i}:")
            print(f"- Visual: {visual}")
            print(f"- Narration: {narration}")
            if metadata.get('continue_from'):
                print(f"- Continues from: Scene {metadata['continue_from']}")
        
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
        # Load and validate configuration
        config = Config()
        
        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Initialize media client based on type
        if args.media_type == "video":
            media_client = ReplicateRayClient(
                api_token=config.replicate_token,
                dev_mode=config.is_dev_mode
            )
            print("\nUsing Ray model for video generation")
        else:  # image
            media_client = SDXLClient(
                api_token=config.replicate_token,
                dev_mode=config.is_dev_mode  # Use config.is_dev_mode for consistency
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
                print(f"DEBUG - Setting highlight_use_box from preset: {args.highlight_use_box}")
            args.visible_lines = preset["visible_lines"]
            args.bottom_padding = preset["bottom_padding"]
        
        # Set up style config from args
        style_config = {
            "aspect_ratio": "16:9" if args.video_format == "landscape" else "9:16" if args.video_format == "short" else "1:1",
            "font": args.font,
            "font_size": args.font_size,
            "color": args.color,
            "animation_style": args.animation_style,
            "caption_style": "karaoke" if args.video_format == "short" else "standard",
            "highlight_color": args.highlight_color,
            "highlight_bg_color": args.highlight_bg_color,
            "stroke_color": args.stroke_color,
            "stroke_width": args.stroke_width,
            "use_background": args.use_background,
            "highlight_use_box": args.highlight_use_box,
            "visible_lines": args.visible_lines,
            "bottom_padding": args.bottom_padding
        }
        
        print(f"DEBUG - Final highlight_use_box value: {style_config['highlight_use_box']}")
        
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