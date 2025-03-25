"""
Test script for the voice optimizer functionality.
This script demonstrates how the voice optimizer analyzes different types of content
and selects appropriate voice parameters.
"""

import os
import logging
from pathlib import Path
from tts import TextToSpeech
from voice_optimizer import VoiceOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_voice_optimizer():
    """Test the voice optimizer with different types of content."""
    optimizer = VoiceOptimizer()
    
    test_texts = {
        "educational": "Learn how to code in Python with this step-by-step tutorial...",
        "storytelling": "Once upon a time in a magical forest...",
        "promotional": "Don't miss this amazing limited-time offer...",
        "informational": "Recent studies have shown that...",
        "inspirational": "You have the power to achieve your dreams..."
    }
    
    for content_type, text in test_texts.items():
        # Analyze text
        analysis = optimizer.analyze_text(text)
        
        # Get optimized parameters
        params = optimizer.optimize_voice_parameters(text)
        
        # Verify analysis results
        assert isinstance(analysis, dict)
        assert "primary_content_type" in analysis
        assert "primary_emotion" in analysis
        assert "complexity_score" in analysis
        
        # Verify voice parameters
        assert isinstance(params, dict)
        assert "voice_name" in params
        assert "stability" in params
        assert "style" in params
        assert 0 <= params["stability"] <= 1
        assert 0 <= params["style"] <= 1

def test_tts_with_optimizer():
    """Test the TextToSpeech class with voice optimization."""
    # Check if API key is available
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("ELEVENLABS_API_KEY environment variable not set. Skipping TTS test.")
        return
    
    # Initialize the TTS client with voice optimization
    tts = TextToSpeech(use_smart_voice=True)
    
    # Test texts
    test_texts = {
        "educational": "In this tutorial, we'll learn how to build a neural network from scratch.",
        "storytelling": "Once upon a time in a distant galaxy, a brave explorer discovered a hidden planet.",
        "promotional": "Introducing our revolutionary new product! This amazing innovation will change your life!",
        "dialogue": "\"What are you doing here?\" she asked. \"I've been waiting for you,\" he replied."
    }
    
    # Create output directory for test audio
    output_dir = Path("test_audio")
    output_dir.mkdir(exist_ok=True)
    
    print("\n=== TTS WITH VOICE OPTIMIZATION ===\n")
    
    for content_type, text in test_texts.items():
        print(f"\n--- {content_type.upper()} TEXT ---")
        print(f"Text: {text}")
        
        # Generate speech with smart voice optimization
        try:
            result = tts.generate_speech(text)
            
            # Copy the file to our test directory with a descriptive name
            audio_path = Path(result["path"])
            new_path = output_dir / f"{content_type}_optimized.mp3"
            
            # Read the audio file and write to the new path
            with open(audio_path, "rb") as src, open(new_path, "wb") as dst:
                dst.write(src.read())
                
            print(f"Generated audio saved to: {new_path}")
            print(f"Duration: {result.get('duration', 0):.2f} seconds")
            print(f"Word timings available: {len(result.get('word_timings', []))}")
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    # Test the voice optimizer analysis
    test_voice_optimizer()
    
    # Test TTS with voice optimization
    test_tts_with_optimizer() 