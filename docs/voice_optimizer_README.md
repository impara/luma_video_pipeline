# Voice Optimizer for TTS

The Voice Optimizer is a streamlined system that automatically enhances voice parameters based on script content analysis. It integrates seamlessly with the existing TextToSpeech implementation to provide optimized voice characteristics for the Daniel voice without adding command-line complexity.

## Features

- **~~Content-Aware Voice Selection~~**: ~~Automatically analyzes script content to determine the most appropriate voice~~ (Completely removed - now always uses "Daniel" voice)
- **Parameter Optimization**: Dynamically adjusts stability, style, and other voice parameters based on content type
- **Emotion Detection**: Identifies emotional tone in text and adjusts voice parameters accordingly
- **Complexity Analysis**: Analyzes text complexity to select appropriate model and clarity settings
- **Seamless Integration**: Works with existing word timing data for karaoke captions
- **Improved Word Timing**: Special handling for numeric references and verse citations (like "99:7-8") to ensure proper caption synchronization

## How It Works

The Voice Optimizer uses a combination of pattern matching, keyword analysis, and text metrics to determine the optimal voice parameters for different types of content:

1. **Content Type Analysis**: Identifies whether the text is educational, storytelling, promotional, informational, or inspirational
2. **Emotion Detection**: Detects the primary emotion expressed in the text (positive, negative, calm, energetic, etc.)
3. **Complexity Calculation**: Measures text complexity based on sentence length, word length, and punctuation
4. **Parameter Optimization**: Adjusts voice parameters based on the analysis results
5. **Model Selection**: Chooses between high-quality and fast models based on content length and complexity

## Voice Parameters

The optimizer adjusts several key voice parameters for the Daniel voice:

- **Stability** (0.0-1.0): Controls consistency vs. expressiveness
- **Style** (0.0-1.0): Adjusts stylistic variation in the voice
- **Model**: Selects between `eleven_turbo_v2` (faster) and `eleven_multilingual_v2` (higher quality)

## Content Type Optimization

Different content types receive different voice parameter optimizations (all using the "Daniel" voice):

| Content Type  | Stability | Style | Notes                                            |
| ------------- | --------- | ----- | ------------------------------------------------ |
| Educational   | 0.7       | 0.3   | Clear, professional parameters for tutorials     |
| Storytelling  | 0.4       | 0.7   | Expressive, engaging parameters for narratives   |
| Promotional   | 0.4       | 0.8   | Energetic, enthusiastic parameters for marketing |
| Informational | 0.6       | 0.4   | Authoritative parameters for facts and data      |
| Inspirational | 0.3       | 0.8   | Dynamic, motivational parameters for inspiration |

## Emotion-Based Adjustments

The optimizer further adjusts parameters based on detected emotions:

- **Positive/Negative**: More expressive (lower stability, higher style)
- **Calm**: More stable, less stylistic variation
- **Energetic**: Highly expressive with more style variation
- **Serious**: More stable and clear
- **Humorous**: More expressive with style variation

## Word Timing Improvements

The system includes special handling for numeric references and verse citations to ensure proper caption synchronization:

- **Verse References**: References like "99:7-8" receive extended timing to match how they're spoken (e.g., "ninety-nine, verses seven to eight")
- **Numeric Values**: Standalone numbers receive appropriate timing based on how they're verbalized
- **Pattern Recognition**: Uses regex patterns to identify different types of numeric references
- **Adaptive Timing**: Adjusts timing dynamically based on the complexity of the reference

This ensures that captions remain properly synchronized with the audio, even when encountering numeric references that are spoken differently than they appear in text.

## Usage

The Voice Optimizer is enabled by default in the TextToSpeech class and always uses the "Daniel" voice:

```python
# Initialize TTS with voice parameter optimization (always uses Daniel voice)
tts = TextToSpeech(use_smart_voice=True)  # True is the default

# Generate speech with optimized parameters for Daniel voice
result = tts.generate_speech("Your text here")
```

You can also disable voice parameter optimization entirely:

```python
# Initialize TTS without voice parameter optimization
tts = TextToSpeech(use_smart_voice=False)
```

From the command line:

```bash
# Use with parameter optimization
python main.py --media-type video --script-file script.txt

# Disable parameter optimization
python main.py --media-type video --script-file script.txt --disable-smart-voice
```

## Testing

You can test the voice optimizer using the provided test script:

```bash
python test_daniel_voice.py
```

This will generate sample audio files with the "Daniel" voice using optimized parameters for different content types.

## Benefits

- **Consistent Voice Identity**: Always uses the "Daniel" voice for a unified experience
- **Optimized Expressiveness**: Adjusts parameters based on content type for appropriate delivery
- **Enhanced User Experience**: Natural-sounding narration with consistent voice identity
- **Simplified Workflow**: No need to manually select voice parameters
- **Multi-Provider Support**: Works with both ElevenLabs and UnrealSpeech providers
- **Improved Caption Synchronization**: Special handling for numeric references ensures captions stay in sync with audio

## Integration with Existing Pipeline

The Voice Optimizer integrates seamlessly with the existing pipeline:

1. It works with the existing caching system, adding voice parameters to cache keys
2. It preserves word timing data for karaoke captions
3. It can be enabled/disabled via the `--disable-smart-voice` command-line argument
4. It supports both ElevenLabs and UnrealSpeech providers via the `--tts-provider` option
