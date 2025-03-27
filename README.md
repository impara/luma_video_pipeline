# Media Generation Pipeline

A Python-based pipeline that generates narrated videos using either:

- Replicate's Ray model for video generation
- SDXL for high-quality images with Ken Burns animation
- Google Gemini API for image generation (free tier available)

Combined with ElevenLabs for text-to-speech voiceovers and automatic caption overlays.

## Features

- Multi-provider media generation:
  - Video mode: Generate AI videos using Replicate Ray model
  - Image mode: Generate images with dynamic animations
    - SDXL via Replicate for high-quality images
    - Google Gemini for free-tier image generation
- Text-to-speech voiceovers using ElevenLabs or UnrealSpeech
- Automatic caption generation and overlay
- TikTok-style karaoke captions for short-form videos
- Multi-scene video assembly with transitions
- Support for various aspect ratios and styles
- Local caching of generated content
- Ken Burns and other animation effects for images
- Modular architecture with centralized caching and error handling
- Consistent "Daniel" voice with smart parameter optimization
- Improved caption synchronization for numeric references
- Clear output directories option for fresh runs

## Architecture

The pipeline is built with a modular architecture:

- **Core**: Contains configuration, logging, error handling, and utilities

  - `config.py`: Centralized configuration management
  - `logging_config.py`: Unified logging setup
  - `error_handling.py`: Error types and retry mechanisms
  - `cache_handler.py`: Caching system
  - `utils.py`: Common utility functions

- **Media**: Media generation clients with a common interface

  - `client_base.py`: Abstract base class defining the client interface
  - `replicate_client.py`: Client for Replicate's Ray model (video)
  - `sdxl_client.py`: Client for SDXL image generation
  - `gemini_client.py`: Client for Google Gemini API

- **Audio**: Audio generation and processing

  - `tts.py`: Text-to-speech synthesis
  - `unrealspeech_provider.py`: UnrealSpeech implementation
  - `voice_optimizer.py`: Voice parameter optimization

- **Video**: Video generation, processing, and assembly

  - `scene_builder.py`: Orchestrates scene generation
  - `assembler.py`: Combines scenes into final video
  - `captions.py`: Caption generation and overlay
  - `parse_script.py`: Script parsing utilities

- **Integrations**: External service integrations

  - YouTube: For video upload and management

- **Output**: Generated files (not in version control)
  - `videos/`: Generated video files
  - `images/`: Generated image files
  - `audio/`: TTS audio files
  - `captions/`: Caption files
  - `temp/`: Temporary processing files
  - `replicate_segments/`: Cached video segments

## Prerequisites

- Python 3.10 or higher
- Replicate API token (only if using SDXL or Ray models) - get one at https://replicate.com
- ElevenLabs API key (for TTS functionality) - get one at https://elevenlabs.io
- Google Gemini API key (only if using Gemini for images) - get one at https://ai.google.dev/

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd luma_video_pipeline
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your API keys:
# - REPLICATE_API_TOKEN (for SDXL/Ray)
# - ELEVENLABS_API_KEY or API_KEY (for TTS)
# - GEMINI_API_KEY (for Google Gemini)
```

## Usage

The pipeline supports multiple modes of operation:

### Video Mode (using Ray)

```bash
python main.py \
    --media-type video \
    --script-file script.txt \
    --output-file output.mp4
```

### Image Mode (using SDXL or Gemini)

Using SDXL (default):

```bash
python main.py \
    --media-type image \
    --image-model sdxl \
    --script-file script.txt \
    --output-file output.mp4 \
    --animation-style ken_burns
```

Using Google Gemini (free tier):

```bash
python main.py \
    --media-type image \
    --image-model gemini \
    --script-file script.txt \
    --output-file output.mp4 \
    --animation-style ken_burns
```

### Clean Output Directories

To start with fresh output directories (preserving videos):

```bash
python main.py \
    --media-type image \
    --script-file script.txt \
    --output-file output.mp4 \
    --clear
```

The `--clear` flag cleans all output directories (images, audio, captions, temp, replicate_segments) while preserving the videos directory.

### Animation Styles

When using image mode, you can choose from several animation styles:

- `ken_burns`: Smooth zoom and pan effect (default)
- `zoom`: Simple zoom in/out
- `pan`: Horizontal/vertical panning
- More styles coming soon!

### Social Media Short Format

For creating vertical 9:16 videos for platforms like TikTok, Instagram Reels, or YouTube Shorts:

```bash
python main.py \
    --media-type video \
    --script-file script.txt \
    --output-file short_video.mp4 \
    --video-format short
```

### YouTube Optimization

The pipeline includes YouTube optimization features that adjust video quality settings to match YouTube's recommended specifications:

- **High Resolution**:

  - Landscape: 1920x1080 (1080p)
  - Short: 1080x1920
  - Square: 1080x1080

- **Optimized Bitrates**:
  - Landscape: 10Mbps
  - Short: 8Mbps
  - Square: 9Mbps

By default, the pipeline runs in fast generation mode for quicker development and testing. When you're ready to produce a production-quality video, add the `--youtube-optimized` flag:

```bash
# Generate faster, lower-resolution video for testing (default)
python main.py \
    --media-type video \
    --script-file script.txt \
    --video-format landscape

# Generate high-quality video for YouTube
python main.py \
    --media-type video \
    --script-file script.txt \
    --video-format landscape \
    --youtube-optimized
```

The `--youtube-optimized` flag affects both:

- **Output Resolution**: Increases to higher resolution (1080p) for optimal YouTube quality
- **Encoding Parameters**: Uses higher quality, more computationally intensive encoding settings

The default fast generation mode:

- **Uses Lower Resolution**: 720p instead of 1080p
- **Uses Faster Encoding**: Applies faster encoding presets with lower quality settings
- **Speeds Up Processing**: Can be 40-60% faster than YouTube-optimized mode

Note: Videos will still be accepted by YouTube even without optimization. The optimization settings follow YouTube's recommended specifications for best quality, not their minimum requirements.

You can also combine with image mode:

```bash
python main.py \
    --media-type image \
    --script-file script.txt \
    --output-file short_video.mp4 \
    --video-format short \
    --animation-style ken_burns
```

### TikTok-Style Karaoke Captions

When using `--video-format short`, the pipeline automatically enables TikTok-style karaoke captions that highlight each word as it's spoken. You can customize the appearance:

```bash
python main.py \
    --media-type video \
    --script-file script.txt \
    --output-file tiktok_video.mp4 \
    --video-format short \
    --highlight-color "#00ffff" \
    --stroke-color "black" \
    --stroke-width 2 \
    --font-size 60 \
    --use-timing-adjustment true
```

The `--use-timing-adjustment` flag controls whether to apply timing adjustments to better synchronize the karaoke highlighting with the audio. By default:

- Portrait videos use a 0.2 second adjustment
- Landscape videos use a 0.25 second adjustment
  Set to `false` to disable timing adjustments if you prefer the raw timings.

For authentic TikTok-style boxed captions (where highlighted words appear in a box):

```bash
python main.py \
    --media-type video \
    --script-file script.txt \
    --output-file tiktok_video.mp4 \
    --video-format short \
    --style tiktok_boxed
```

### Caption Style Presets

The pipeline includes several ready-to-use caption style presets:

- `tiktok_neon`: Cyan highlighting with bold text (best for most videos)
- `tiktok_bold`: Bold text with highlighted words in yellow boxes (authentic TikTok style)
- `tiktok_yellow`: Bold text with highlighted words in vibrant yellow flag style (matches VON style in image)
- `tiktok_minimal`: Clean minimal style with pink highlighting (subtle)
- `tiktok_boxed`: Bold text with highlighted words in red boxes (authentic TikTok style)

```bash
# Use a preset style
python main.py \
    --media-type video \
    --script-file script.txt \
    --output-file tiktok_video.mp4 \
    --video-format short \
    --style tiktok_boxed
```

### For Karaoke Captions

- Keep narration text natural and well-paced
- Use high-contrast colors for better readability
- Adjust font size based on video dimensions
- Test with different highlight colors for best effect
- Consider using stroke outlines for better visibility
- Use the paging system for longer narrations
- Adjust visible lines (2-3 recommended) for optimal viewing
- Numeric references (like "99:7-8") are handled with special timing to ensure proper synchronization

#### Caption Paging System

The TikTok-style karaoke captions now feature a paging system that:

- Shows a fixed number of lines (2-3) at a time in a consistent position
- Transitions cleanly between pages when narration progresses to a new set of lines
- Maintains word highlighting while eliminating distracting scrolling
- Provides better readability and a cleaner viewing experience
- Prevents captions from being cut off at screen edges

Configure the paging system with these options:

```bash
# Example with custom paging system settings
python main.py \
    --media-type image \
    --script-file script.txt \
    --output-file tiktok_video.mp4 \
    --video-format short \
    --style tiktok_boxed \
    --visible-lines 2 \
    --bottom-padding 80
```

### Script Format

The pipeline uses a labeled format for each scene:

```text
# Scene 1 (Visual)
A futuristic cityscape with hovercars and neon lights
# Scene 1 (Narration)
Welcome to the future, where innovation never sleeps

# Scene 2 (Visual) [continue=1]
The same city transforming as day turns to night
# Scene 2 (Narration)
As darkness falls, the city awakens with a new energy
```

For each scene:

- **Visual Section**

  - Starts with `# Scene N (Visual)` header
  - Optional continuation flag: `[continue=N]` (video mode only)
  - Used by Ray/SDXL to generate media
  - Focus on vivid, descriptive visual elements

- **Narration Section**
  - Starts with `# Scene N (Narration)` header
  - Used for both text-to-speech and captions
  - Write in a natural, engaging style

### Command Line Options

```bash
python main.py --help

options:
  -h, --help            show this help message and exit
  --script-file SCRIPT_FILE
                        Path to script file (two lines per scene)
  --output-file OUTPUT_FILE
                        Output path for the final video (default: final_video.mp4)
  --media-type {video,image}
                        Type of media to generate (default: video)
  --image-model {sdxl,gemini}
                        Model to use for image generation (default: sdxl)
  --animation-style {ken_burns,zoom,pan}
                        Animation style for image mode (default: ken_burns)
  --video-format {landscape,short,square}
                        Video format/aspect ratio (default: landscape)
                        'landscape' is 16:9, 'short' is 9:16 for social media, 'square' is 1:1
  --use-timing-adjustment {true,false}
                        Whether to apply timing adjustments for karaoke captions (default: true)
  --youtube-optimized   Optimize video resolution and bitrate for YouTube standards (higher quality, slower processing)
  --clear               Clear all output directories (except videos) before processing

Style options:
  --style {default,tiktok_neon,tiktok_bold,tiktok_minimal,tiktok_boxed,tiktok_yellow}
                        Preset style for captions (default: default)
  --font FONT          Font for captions (default: Arial-Bold)
  --font-size FONT_SIZE
                        Font size for captions (default: 24)
  --color COLOR         Text color for captions (default: white)
  --highlight-color COLOR
                        Color for highlighted words in karaoke captions (default: #ff5c5c)
  --highlight-bg-color COLOR
                        Background color for highlighted words in boxed style (default: white)
  --stroke-color COLOR
                        Outline color for karaoke captions (default: black)
  --stroke-width WIDTH
                        Outline width for karaoke captions (default: 2)
  --highlight-use-box   Use box highlighting style for words (like authentic TikTok captions)
  --use-background      Add semi-transparent background behind captions
  --visible-lines LINES
                        Number of caption lines to show at once (default: 2)
  --bottom-padding PADDING
                        Padding from bottom of screen in pixels (default: 80)

Text-to-Speech options:
  --tts-provider {elevenlabs,unrealspeech}
                        TTS provider to use (default: elevenlabs). Always uses 'Daniel' voice.
  --disable-smart-voice
                        Disable smart voice parameter optimization
```

## Development Mode

The pipeline supports a development mode (`VIDEO_PIPELINE_DEV_MODE=true` in .env) that:

- Uses cached media instead of making API calls
- Reduces iteration time during development
- Works with both video and image generation

## Core Components

### Media Client Interface

The `MediaClient` interface allows easy switching between different media generation backends:

- **ReplicateRayClient**: For video generation
- **SDXLClient**: For image generation
- **GeminiClient**: For Google Gemini image generation

All clients share a common interface, making it easy to add new providers.

### Cache Handler

The `CacheHandler` provides a centralized caching mechanism:

- Avoids redundant API calls
- Persists generated media across sessions
- Handles cache keys, file paths, and storage
- Supports development mode for faster iteration

### Error Handling

The `error_handling` module provides:

- Standardized exception hierarchy
- Retry decorators with exponential backoff
- Consistent error messages and recovery strategies

### Utilities

Common functionality is centralized in the `utils` module:

- File download helpers
- Path handling utilities
- URL processing

## Best Practices

### For Video Generation (Ray)

- Use clear, descriptive prompts
- Keep scenes between 4-10 seconds
- Use continuation flag for smooth transitions
- Test in dev mode first using cached videos

### For Image Generation (SDXL)

- Optimal dimensions: 1024x576 (16:9)
- Use guidance_scale 7-9 for best results
- Adjust animation parameters for smooth motion
- Generate multiple variations when needed

### For Image Generation (Gemini)

- Clear descriptive prompts work best
- Use the free tier API key to generate images
- Set a consistent aspect ratio for best results
- Consider using the Ken Burns animation style

### For Karaoke Captions

- Keep narration text natural and well-paced
- Use high-contrast colors for better readability
- Adjust font size based on video dimensions
- Test different highlight colors for best effect
- Consider using stroke outlines for better visibility
- Use the paging system for longer narrations
- Adjust visible lines (2-3 recommended) for optimal viewing

### General Tips

- Write natural, conversational narration
- Balance visual and audio durations
- Use high-contrast caption colors
- Test different animation styles
- Cache generated content for faster iteration
- Use the `--clear` flag for fresh runs to avoid using old cached content
- Use fast generation mode (default) during development for quick iteration
- Add `--youtube-optimized` only for final production videos
- For very large projects, process in smaller batches to manage memory usage
- Consider testing on a small script before processing lengthy content

## Technical Details

- Videos are cached in `output/replicate_segments/`
- Images are stored in `output/images/`
- Audio files are stored in `output/audio/`
- Temporary files are stored in `output/temp/`
- Final videos saved in `output/videos/`
- Caption data stored in `output/captions/`
- Supports aspect ratios: 16:9, 9:16, 1:1
- Default video FPS: 24
- Audio format: WAV (44.1kHz)

## Advanced Performance Features

### Memory-Efficient Video Handling

The pipeline uses sophisticated memory management to process videos efficiently:

- **Adaptive Batch Processing**: Dynamically adjusts batch sizes based on current memory usage
- **Memory Monitoring**: Continuously tracks memory consumption to prevent out-of-memory errors
- **Hybrid Processing Strategy**:
  - Uses in-memory processing when memory is available for faster processing
  - Automatically switches to disk-based processing when memory usage is high
  - Dynamically adjusts batch size in response to memory pressure
- **Frame-by-Frame Processing**: Uses generator-based frame processing to avoid loading entire videos into memory
- **Hardware Acceleration Detection**: Automatically detects and uses available hardware encoding (NVENC, QSV, VAAPI) with fallback to CPU encoding when needed

These features work together to enable processing of large videos even on systems with limited resources, while still taking full advantage of available memory and hardware when possible.

### Performance-Optimized Encoding

The pipeline uses an intelligent two-tier encoding system for optimal performance:

**Intermediate Files**:

- Uses ultrafast/veryfast encoding presets for temporary files
- Optimized for speed rather than quality during processing
- Automatically uses lower quality settings to minimize processing time
- Hardware-accelerated when available (NVIDIA, Intel QSV, AMD)

**Final Output**:

- By default, uses faster encoding presets for development and testing
- When `--youtube-optimized` flag is added:
  - Uses high-quality encoding settings for production-ready video
  - Applies YouTube-recommended parameters for optimal streaming quality
  - Uses two-pass encoding for superior quality/size ratio
  - Uses medium/slower presets for better compression efficiency
  - Hardware acceleration is quality-optimized for final output

This system significantly improves overall processing speed during development while providing the option for high-quality output when needed.

### Video Resolutions

**Fast Generation Mode (Default)**:

- Landscape (16:9): 1280x720 (720p)
- Short (9:16): 720x1280
- Square (1:1): 720x720
- Video bitrate: 4Mbps
- Audio bitrate: 128kbps

**YouTube-Optimized Mode** (with `--youtube-optimized` flag):

- Landscape (16:9): 1920x1080 (1080p)
- Short (9:16): 1080x1920
- Square (1:1): 1080x1080
- Video bitrates: 8-10Mbps (format-specific)
- Audio bitrate: 192kbps

Both modes produce YouTube-compatible videos. The optimized mode follows YouTube's recommended specifications for best quality (1080p), while the default mode ensures at least 720p quality on YouTube with faster processing.

## Production Readiness

The pipeline is production-ready with:

- Robust error handling
- Clear separation of concerns
- Comprehensive test coverage
- Graceful fallbacks
- Detailed logging
- Resource cleanup

For production deployment:

1. Ensure API keys are securely stored
2. Monitor API usage and costs
3. Implement rate limiting
4. Use version control for scripts
5. Regular backups of generated content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Text-to-Speech Features

The pipeline includes advanced TTS capabilities:

- **Consistent Voice**: Always uses the "Daniel" voice for a unified experience
- **Smart Parameter Optimization**: Automatically adjusts voice parameters based on content type
- **Multiple TTS Providers**: Supports both ElevenLabs and UnrealSpeech
- **Improved Word Timing**: Special handling for numeric references ensures captions stay in sync with audio
- **Caching System**: Avoids redundant TTS API calls for identical text
- **Word-Level Timing**: Provides precise timing data for karaoke captions

To use a specific TTS provider:

```bash
# Use ElevenLabs (default)
python main.py --media-type video --script-file script.txt --tts-provider elevenlabs

# Use UnrealSpeech
python main.py --media-type video --script-file script.txt --tts-provider unrealspeech
```

To disable smart voice parameter optimization:

```bash
python main.py --media-type video --script-file script.txt --disable-smart-voice
```
