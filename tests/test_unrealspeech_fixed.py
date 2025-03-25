"""
Test script for the fixed UnrealSpeech provider implementation.
This script tests only the UnrealSpeech provider to verify the fix.
"""

import os
import logging
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from unrealspeech_provider import UnrealSpeechProvider, UnrealSpeechError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_directly():
    """Test the UnrealSpeech API directly with a simple request to diagnose issues."""
    logger.info("Testing UnrealSpeech API directly...")
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.error("API_KEY environment variable not found")
        return False
    
    base_url = "https://api.v8.unrealspeech.com"
    
    # Create a session
    session = requests.Session()
    session.max_redirects = 5
    
    # Prepare headers and payload
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    
    payload = {
        "Text": "This is a direct API test.",
        "VoiceId": "Eleanor",
        "Bitrate": "192k",
        "Speed": 0,      # Normal speed (range: -1 to 1)
        "Pitch": 1.0,    # Normal pitch (range: 0.5 to 1.5)
        "TimestampType": "word"
    }
    
    try:
        # Make the request
        logger.info(f"Making direct API request to {base_url}/speech")
        response = session.post(
            f"{base_url}/speech",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        # Check if successful
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            
            # Handle JSON response with audio URL
            if "application/json" in content_type:
                try:
                    data = response.json()
                    logger.info(f"Received JSON response: {json.dumps(data)[:200]}...")
                    
                    # Check if the JSON contains an OutputUri field
                    if "OutputUri" in data:
                        audio_url = data["OutputUri"]
                        logger.info(f"Found audio URL: {audio_url}")
                        
                        # Download the audio file from the URL
                        logger.info(f"Downloading audio from URL: {audio_url}")
                        audio_response = session.get(audio_url, timeout=30)
                        
                        if audio_response.status_code == 200:
                            # Save the audio file for inspection
                            output_file = Path("generated_audio/direct_api_test.mp3")
                            with open(output_file, "wb") as f:
                                f.write(audio_response.content)
                            
                            logger.info(f"Direct API test successful. Audio saved to {output_file} ({len(audio_response.content)} bytes)")
                            return True
                        else:
                            logger.error(f"Failed to download audio from URL: {audio_url}, status code: {audio_response.status_code}")
                            return False
                    else:
                        logger.error(f"JSON response does not contain OutputUri: {json.dumps(data)}")
                        return False
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response.text[:200]}...")
                    return False
            # Handle direct audio response
            elif "audio" in content_type or "mpeg" in content_type:
                # Save the audio file for inspection
                output_file = Path("generated_audio/direct_api_test.mp3")
                with open(output_file, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Direct API test successful. Audio saved to {output_file} ({len(response.content)} bytes)")
                return True
            else:
                logger.error(f"Unexpected content type: {content_type}")
                logger.error(f"Response content (first 100 bytes): {response.content[:100]}")
                return False
        else:
            # Try to parse error response
            try:
                error_data = response.json()
                logger.error(f"API error: {json.dumps(error_data, indent=2)}")
            except:
                logger.error(f"API error (non-JSON): {response.text[:500]}")
            
            return False
    
    except Exception as e:
        logger.error(f"Direct API test failed: {str(e)}")
        return False

def test_unrealspeech_direct():
    """Test UnrealSpeech provider directly."""
    logger.info("Testing UnrealSpeech provider directly...")
    
    try:
        # Initialize UnrealSpeech provider
        provider = UnrealSpeechProvider()
        
        # Generate speech
        result = provider.generate_speech(
            text="This is a test of the fixed UnrealSpeech provider.",
            voice_name="Eleanor"  # Using a valid voice from the updated list
        )
        
        # Log result
        logger.info(f"Generated audio file: {result['path']}")
        logger.info(f"Duration: {result['duration']:.2f} seconds")
        logger.info(f"Word timings: {len(result['word_timings'])} words")
        
        # Verify the file exists and has content
        audio_file = Path(result['path'])
        if audio_file.exists():
            file_size = audio_file.stat().st_size
            logger.info(f"Audio file size: {file_size} bytes")
            if file_size < 100:
                logger.warning(f"Audio file is suspiciously small: {file_size} bytes")
        else:
            logger.error(f"Audio file does not exist: {result['path']}")
        
        logger.info("UnrealSpeech test completed successfully!")
        return True
    except UnrealSpeechError as e:
        logger.error(f"UnrealSpeech test failed: {e}")
        return False

def test_voice_mapping():
    """Test the voice mapping functionality."""
    logger.info("Testing voice mapping functionality...")
    
    try:
        # Initialize UnrealSpeech provider
        provider = UnrealSpeechProvider()
        
        # Test with an old voice name that should be mapped
        result = provider.generate_speech(
            text="This is a test of the voice mapping functionality.",
            voice_name="Scarlett"  # This should be mapped to "Eleanor"
        )
        
        # Log result
        logger.info(f"Generated audio file: {result['path']}")
        logger.info(f"Duration: {result['duration']:.2f} seconds")
        logger.info(f"Word timings: {len(result['word_timings'])} words")
        
        logger.info("Voice mapping test completed successfully!")
        return True
    except UnrealSpeechError as e:
        logger.error(f"Voice mapping test failed: {e}")
        return False

def test_multiple_voices():
    """Test multiple voices to ensure they all work."""
    logger.info("Testing multiple voices...")
    
    # List of voices to test
    voices_to_test = ["Eleanor", "Daniel", "Amelia", "Benjamin", "Charlotte"]
    
    success_count = 0
    
    try:
        # Initialize UnrealSpeech provider
        provider = UnrealSpeechProvider()
        
        for voice in voices_to_test:
            try:
                logger.info(f"Testing voice: {voice}")
                
                # Generate speech with this voice
                result = provider.generate_speech(
                    text=f"This is a test of the {voice} voice.",
                    voice_name=voice
                )
                
                # Log result
                logger.info(f"Generated audio file: {result['path']}")
                logger.info(f"Duration: {result['duration']:.2f} seconds")
                
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to generate speech with voice {voice}: {e}")
        
        logger.info(f"Multiple voices test completed. {success_count}/{len(voices_to_test)} voices worked successfully.")
        return success_count > 0
    except Exception as e:
        logger.error(f"Multiple voices test failed: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Create output directory
    Path("generated_audio").mkdir(exist_ok=True)
    
    # First test the API directly
    logger.info("=== STEP 1: Testing UnrealSpeech API directly ===")
    api_success = test_api_directly()
    
    if api_success:
        logger.info("✅ Direct API test passed!")
    else:
        logger.error("❌ Direct API test failed. There may be issues with the API or credentials.")
        logger.info("Continuing with provider tests anyway...")
    
    # Test UnrealSpeech provider
    logger.info("\n=== STEP 2: Testing UnrealSpeech provider ===")
    success = test_unrealspeech_direct()
    
    if success:
        logger.info("✅ Basic UnrealSpeech test passed!")
        
        # Test voice mapping
        logger.info("\n=== STEP 3: Testing voice mapping ===")
        mapping_success = test_voice_mapping()
        if mapping_success:
            logger.info("✅ Voice mapping test passed!")
        else:
            logger.error("❌ Voice mapping test failed.")
        
        # Test multiple voices
        logger.info("\n=== STEP 4: Testing multiple voices ===")
        voices_success = test_multiple_voices()
        if voices_success:
            logger.info("✅ Multiple voices test passed!")
        else:
            logger.error("❌ Multiple voices test failed.")
        
        if mapping_success and voices_success:
            logger.info("\n✅ All tests passed! UnrealSpeech provider is working correctly!")
        else:
            logger.warning("\n⚠️ Some tests failed. UnrealSpeech provider is partially working.")
    else:
        logger.error("\n❌ Basic UnrealSpeech test failed. Provider is not working correctly.") 