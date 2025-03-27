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
cd video_pipeline
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

The pipeline supports multiple modes of operation, customizable with command line options:

### Basic Usage Patterns

```bash
# Basic video mode (using Ray)
python main.py --media-type video --script-file script.txt --output-file output.mp4

# Basic image mode with Gemini (free tier)
python main.py --media-type image --image-model gemini --script-file script.txt --output-file output.mp4

# Clear cache directories before running
python main.py --media-type image --script-file script.txt --output-file output.mp4 --clear

# Social media vertical format (9:16 for TikTok, Instagram, YouTube Shorts)
python main.py --media-type video --script-file script.txt --output-file short_video.mp4 --video-format short
```

### Quality & Performance Settings

The pipeline offers two quality modes controlled by the `--youtube-optimized` flag:

```bash
# Fast generation mode (default) - 720p resolution, faster encoding
python main.py --media-type video --script-file script.txt

# YouTube-optimized mode - 1080p resolution, higher quality encoding
python main.py --media-type video --script-file script.txt --youtube-optimized
```

#### Resolution & Encoding Comparison

| Feature        | Default Mode          | YouTube-Optimized Mode  |
| -------------- | --------------------- | ----------------------- |
| Landscape      | 1280x720 (720p)       | 1920x1080 (1080p)       |
| Vertical       | 720x1280              | 1080x1920               |
| Square         | 720x720               | 1080x1080               |
| Video Bitrate  | ~4Mbps                | 8-10Mbps                |
| Audio Bitrate  | 128kbps               | 192kbps                 |
| Encoding Speed | 40-60% faster         | Higher quality, slower  |
| Best For       | Development & Testing | Final Production Videos |

### Text-to-Speech Options

The pipeline always uses the "Daniel" voice with smart parameter optimization:

```bash
# Use ElevenLabs TTS provider (default)
python main.py --media-type video --script-file script.txt --tts-provider elevenlabs

# Use UnrealSpeech TTS provider
python main.py --media-type video --script-file script.txt --tts-provider unrealspeech

# Disable smart parameter optimization (not recommended)
python main.py --media-type video --script-file script.txt --disable-smart-voice
```

### Caption Styling Options

The pipeline includes several ready-to-use caption style presets:

```bash
# Use a preset style (options: tiktok_neon, tiktok_bold, tiktok_yellow, tiktok_minimal, tiktok_boxed)
python main.py --media-type video --script-file script.txt --style tiktok_neon

# Custom caption styling
python main.py --media-type video --script-file script.txt \
  --highlight-color "#00ffff" \
  --stroke-color "black" \
  --stroke-width 2 \
  --font-size 60 \
  --visible-lines 2 \
  --bottom-padding 80
```

## Caption System

The pipeline features a sophisticated caption system with different modes:

### TikTok-Style Karaoke Captions

When using `--video-format short` or any style preset, the pipeline enables word-level "karaoke" captions that highlight each word as it's spoken.

#### Caption Style Presets

| Preset         | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| tiktok_neon    | Cyan highlighting with bold text (best for most videos)                   |
| tiktok_bold    | Bold text with highlighted words in yellow boxes (authentic TikTok style) |
| tiktok_yellow  | Bold text with highlighted words in vibrant yellow flag style             |
| tiktok_minimal | Clean minimal style with pink highlighting (subtle)                       |
| tiktok_boxed   | Bold text with highlighted words in red boxes (authentic TikTok style)    |

#### Caption Paging System

The TikTok-style karaoke captions feature a paging system that:

- Shows a fixed number of lines (2-3) at a time in a consistent position
- Transitions cleanly between pages when narration progresses
- Maintains word highlighting while eliminating distracting scrolling
- Provides better readability and a cleaner viewing experience

## Pipeline Components

### Media Generation

The pipeline offers three media generation options:

1. **Video Mode**: Generate AI videos using Replicate Ray model

   - Best for scenes that need actual motion and camera moves

2. **Image Mode with SDXL**: Generate high-quality images with Ken Burns animation

   - Best for detailed, artistic quality images
   - Uses your Replicate API token

3. **Image Mode with Gemini**: Generate images using Google's free tier
   - Best for faster, free image generation
   - Lower detail but entirely free with Google API key

### Text-to-Speech System

The pipeline includes advanced TTS capabilities:

- **Consistent Voice**: Always uses the "Daniel" voice for a unified experience
- **Smart Parameter Optimization**: Automatically adjusts voice parameters based on content type
- **Multiple TTS Providers**: Supports both ElevenLabs and UnrealSpeech
- **Word-Level Timing**: Provides precise timing data for karaoke captions
- **Special Handling**: Numeric references and verse citations are timed correctly

### Performance Optimizations

The pipeline includes several performance optimizations:

#### Memory-Efficient Video Handling

- **Adaptive Batch Processing**: Dynamically adjusts batch sizes based on memory usage
- **Hybrid Processing Strategy**: Uses memory when available, disk when necessary
- **Frame-by-Frame Processing**: Avoids loading entire videos into memory

#### Hardware Acceleration

- **Automatic Detection**: Identifies available hardware encoders (NVENC, QSV, VAAPI)
- **Quality-Optimized**: Uses different hardware settings for intermediate vs. final files
- **Graceful Fallback**: Falls back to CPU encoding when hardware acceleration fails

#### Two-Tier Encoding System

- **Intermediate Files**: Uses fast encoding presets for temporary files
- **Final Output**: Uses quality-optimized settings for the final video
- **YouTube Optimization**: Applies recommended parameters for YouTube uploads

## Best Practices

### Media Generation Tips

- **Video Mode**:

  - Use clear, descriptive prompts
  - Keep scenes between 4-10 seconds
  - Use continuation flag for smooth transitions
  - Test in dev mode first using cached videos

- **Image Mode**:
  - Provide detailed visual descriptions
  - Set a consistent aspect ratio for best results
  - Use the Ken Burns animation style for elegant movement
  - SDXL offers higher quality while Gemini is free tier

### Caption Best Practices

- Keep narration text natural and well-paced
- Use high-contrast colors for better readability
- Test different highlight colors for best effect
- Use 2-3 visible lines for optimal viewing
- Adjust bottom padding based on video format

### General Workflow Tips

- Write natural, conversational narration
- Test with smaller scripts before processing lengthy content
- Use fast generation mode during development
- Add `--youtube-optimized` only for final production videos
- Use the `--clear` flag for fresh runs to avoid using old cached content

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
