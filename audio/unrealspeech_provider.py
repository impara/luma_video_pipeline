"""
UnrealSpeech provider for text-to-speech synthesis.
Implements the same interface as the ElevenLabs provider but uses UnrealSpeech API.
"""

import os
import uuid
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from pydub import AudioSegment
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnrealSpeechError(Exception):
    """Base exception class for UnrealSpeech errors."""
    pass

class UnrealSpeechProvider:
    """
    UnrealSpeech provider for text-to-speech synthesis.
    Implements the same interface as the ElevenLabs provider.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the UnrealSpeech provider.
        
        Args:
            api_key: Optional API key. If not provided, will look for API_KEY env var.
            
        Raises:
            ValueError: If API key is missing or invalid
        """
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable or api_key parameter is required")
        if not self.api_key.strip():
            raise ValueError("API key cannot be empty")
        
        # Base URL for UnrealSpeech API - updated to v8 endpoint
        self.base_url = "https://api.v8.unrealspeech.com"
        
        # Create a session with retry logic and limited redirects
        self.session = self._create_session()
        
        # Available voices in UnrealSpeech - updated with current voice IDs
        # Mapping from friendly names to actual voice IDs
        self.available_voices = [
            # Female voices
            {"name": "Eleanor", "voice_id": "Eleanor", "category": "female", "age": "young"},
            {"name": "Melody", "voice_id": "Melody", "category": "female", "age": "young"},
            {"name": "Charlotte", "voice_id": "Charlotte", "category": "female", "age": "young"},
            {"name": "Luna", "voice_id": "Luna", "category": "female", "age": "young"},
            {"name": "Sierra", "voice_id": "Sierra", "category": "female", "age": "young"},
            {"name": "Emily", "voice_id": "Emily", "category": "female", "age": "young"},
            {"name": "Amelia", "voice_id": "Amelia", "category": "female", "age": "mature"},
            {"name": "Lauren", "voice_id": "Lauren", "category": "female", "age": "mature"},
            {"name": "Chloe", "voice_id": "Chloe", "category": "female", "age": "mature"},
            
            # Male voices
            {"name": "Daniel", "voice_id": "Daniel", "category": "male", "age": "young"},
            {"name": "Jasper", "voice_id": "Jasper", "category": "male", "age": "young"},
            {"name": "Caleb", "voice_id": "Caleb", "category": "male", "age": "young"},
            {"name": "Noah", "voice_id": "Noah", "category": "male", "age": "young"},
            {"name": "Edward", "voice_id": "Edward", "category": "male", "age": "mature"},
            {"name": "Benjamin", "voice_id": "Benjamin", "category": "male", "age": "mature"},
            {"name": "Ethan", "voice_id": "Ethan", "category": "male", "age": "mature"},
            {"name": "Oliver", "voice_id": "Oliver", "category": "male", "age": "mature"}
        ]
        
        # Default voice mapping for compatibility with previous code
        self.voice_mapping = {
            "Scarlett": "Eleanor",  # Map Scarlett to Eleanor
            "Liv": "Melody",        # Map Liv to Melody
            "Amy": "Amelia",        # Map Amy to Amelia
            "Dan": "Daniel",        # Map Dan to Daniel
            "Will": "Benjamin"      # Map Will to Benjamin
        }
        
        # Test the API key with a simple request
        try:
            self._test_api_key()
            logger.info(f"Successfully connected to UnrealSpeech API. Found {len(self.available_voices)} voices.")
        except Exception as e:
            raise UnrealSpeechError(f"Failed to initialize UnrealSpeech client: {str(e)}")
    
    def _create_session(self):
        """Create a requests session with retry logic and limited redirects."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        # Mount the adapter to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set a reasonable max redirects limit
        session.max_redirects = 5
        
        return session
    
    def _test_api_key(self):
        """Test the API key with a simple request."""
        # We'll use a minimal request to test the API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use a minimal payload for testing with a valid voice ID
        payload = {
            "Text": "Test",
            "VoiceId": "Eleanor",  # Using a valid voice ID from the current API
            "Bitrate": "16k",      # Use lowest bitrate for quick test
            "Speed": 0,            # Normal speed (range: -1 to 1)
            "Pitch": 1.0           # Normal pitch (range: 0.5 to 1.5)
            # No TimestampType for simple test to avoid potential errors
        }
        
        # Make a request to the speech endpoint
        try:
            response = self.session.post(
                f"{self.base_url}/speech",
                headers=headers,
                json=payload,
                timeout=10  # Add a timeout to prevent hanging
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_message = f"API key test failed with status code {response.status_code}"
                try:
                    error_data = response.json()
                    error_message += f": {error_data.get('message', 'Unknown error')}"
                except:
                    error_message += f": {response.text}"
                raise UnrealSpeechError(error_message)
        except requests.exceptions.RequestException as e:
            # Handle request exceptions specifically
            raise UnrealSpeechError(f"API connection error: {str(e)}")
    
    def get_available_voices(self) -> List[Dict]:
        """
        Get list of available voices.
        
        Returns:
            List of voice dictionaries with name and voice_id
        """
        return self.available_voices
    
    def map_voice_name(self, voice_name: str) -> str:
        """
        Map old voice names to new ones if needed.
        
        Args:
            voice_name: The voice name to map
            
        Returns:
            The mapped voice name or the original if no mapping exists
        """
        if voice_name in self.voice_mapping:
            mapped_name = self.voice_mapping[voice_name]
            logger.info(f"Mapped voice name '{voice_name}' to '{mapped_name}'")
            return mapped_name
        return voice_name
    
    def ensure_consistent_voice_id(self, voice_name: str) -> Dict:
        """
        Ensure consistent voice ID handling, especially for the Daniel voice.
        
        Args:
            voice_name: The voice name to use
            
        Returns:
            Dict containing voice information
        """
        # For Daniel voice, always use the exact same voice entry
        if voice_name.lower() == "daniel":
            logger.info("Using consistent voice ID for Daniel")
            # Find the Daniel voice in available voices
            daniel_voice = next((v for v in self.available_voices if v["name"] == "Daniel"), None)
            if daniel_voice:
                return daniel_voice
            else:
                # Fallback if Daniel voice not found (shouldn't happen)
                logger.warning("Daniel voice not found in available voices, using first male voice")
                return next((v for v in self.available_voices if v["category"] == "male"), self.available_voices[0])
        
        # For other voices, use the standard lookup
        voice = next((v for v in self.available_voices if v["name"] == voice_name), None)
        if not voice:
            # Try to find by voice_id if name doesn't match
            voice = next((v for v in self.available_voices if v["voice_id"] == voice_name), None)
            if not voice:
                # If still not found, raise an error
                available_voices = ", ".join([v["name"] for v in self.available_voices])
                raise UnrealSpeechError(f"Voice '{voice_name}' not found. Available voices: {available_voices}")
        
        return voice
        
    def generate_speech(self, 
                        text: str, 
                        voice_name: str = "Eleanor", 
                        output_dir: Path = Path("generated_audio"),
                        settings: Dict = None,
                        stability: float = None,
                        similarity_boost: float = None,
                        style: float = None,
                        use_speaker_boost: bool = None) -> Dict:
        """
        Generate speech using UnrealSpeech API.
        
        Args:
            text: Text to convert to speech
            voice_name: Name of the voice to use
            output_dir: Directory to save the audio file
            settings: Optional settings for speech generation
            stability: Not used in UnrealSpeech (ElevenLabs compatibility)
            similarity_boost: Not used in UnrealSpeech (ElevenLabs compatibility)
            style: Not used in UnrealSpeech (ElevenLabs compatibility)
            use_speaker_boost: Not used in UnrealSpeech (ElevenLabs compatibility)
                
        Returns:
            Dict containing:
                - path: Path to the generated audio file
                - word_timings: List of (word, start_time, end_time) tuples
                - duration: Duration of the audio in seconds
                
        Raises:
            UnrealSpeechError: If speech generation fails
        """
        try:
            # Ensure text is not empty
            if not text:
                raise UnrealSpeechError("Text cannot be empty")
            
            # Ensure output directory exists
            output_dir.mkdir(exist_ok=True)
            
            # Map voice name if needed
            voice_name = self.map_voice_name(voice_name)
            
            # Use consistent voice ID handling
            voice = self.ensure_consistent_voice_id(voice_name)
            
            # Generate unique filename
            filename = f"unrealspeech_{int(time.time())}_{voice['name']}.mp3"
            output_path = output_dir / filename
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"  # Explicitly request audio format
            }
            
            # For Daniel voice, use completely fixed parameters for maximum consistency
            if voice['name'].lower() == "daniel":
                logger.info("Using fixed parameters for Daniel voice to ensure maximum consistency")
                payload = {
                    "Text": text,
                    "VoiceId": "Daniel",
                    "Speed": 0,           # Fixed neutral speed
                    "Pitch": 1.0,         # Fixed neutral pitch
                    "Bitrate": "192k",    # Fixed high quality bitrate
                    "TimestampType": "word"
                }
                logger.info("Fixed Daniel voice parameters: Speed=0, Pitch=1.0, Bitrate=192k")
            else:
                # Default settings for other voices
                default_settings = {
                    "speed": 0,      # Normal speed (range: -1 to 1)
                    "pitch": 1.0,    # Normal pitch (range: 0.5 to 1.5)
                    "bitrate": "192k",
                    "timestamp_type": "word"  # Request word-level timestamps (singular, not plural)
                }
                
                # Update with provided settings
                if settings:
                    default_settings.update(settings)
                
                # Prepare payload
                payload = {
                    "Text": text,
                    "VoiceId": voice["voice_id"],
                    "Speed": default_settings["speed"],
                    "Pitch": default_settings["pitch"],
                    "Bitrate": default_settings["bitrate"],
                    "TimestampType": default_settings["timestamp_type"]
                }
            
            logger.info(f"Generating speech with UnrealSpeech for text: '{text[:50]}...' using voice: {voice['name']}")
            
            # Make API request with timeout and error handling
            try:
                response = self.session.post(
                    f"{self.base_url}/speech",
                    headers=headers,
                    json=payload,
                    timeout=30  # Increase timeout for longer texts
                )
                
                # Check if the request was successful
                if response.status_code != 200:
                    error_message = f"Speech generation failed with status code {response.status_code}"
                    try:
                        error_data = response.json()
                        error_message += f": {error_data.get('message', 'Unknown error')}"
                    except:
                        error_message += f": {response.text}"
                    raise UnrealSpeechError(error_message)
                
                # Check content type to determine how to handle the response
                content_type = response.headers.get("Content-Type", "")
                
                # If the response is JSON, it might contain a URL to the audio file
                if "application/json" in content_type:
                    logger.info("Received JSON response, checking for audio URL")
                    try:
                        data = response.json()
                        logger.info(f"JSON response: {json.dumps(data)[:200]}...")
                        
                        # Check if the JSON contains an OutputUri field
                        if "OutputUri" in data:
                            audio_url = data["OutputUri"]
                            logger.info(f"Found audio URL: {audio_url}")
                            
                            # Download the audio file from the URL
                            logger.info(f"Downloading audio from URL: {audio_url}")
                            audio_response = self.session.get(audio_url, timeout=30)
                            
                            if audio_response.status_code != 200:
                                raise UnrealSpeechError(f"Failed to download audio from URL: {audio_url}, status code: {audio_response.status_code}")
                            
                            # Use the content from the audio response
                            audio_content = audio_response.content
                            logger.info(f"Successfully downloaded audio: {len(audio_content)} bytes")
                        else:
                            # If no OutputUri, check if there's an error message
                            if "message" in data:
                                raise UnrealSpeechError(f"API returned error: {data['message']}")
                            else:
                                raise UnrealSpeechError(f"API returned JSON without OutputUri: {json.dumps(data)[:200]}...")
                    except json.JSONDecodeError:
                        raise UnrealSpeechError(f"Failed to parse JSON response: {response.text[:200]}...")
                elif "audio" in content_type or "mpeg" in content_type:
                    # If the response is audio, use it directly
                    logger.info(f"Received audio response: {content_type}")
                    audio_content = response.content
                else:
                    # If the response is neither JSON nor audio, raise an error
                    logger.error(f"Expected audio or JSON response but got content type: {content_type}")
                    logger.error(f"Response headers: {response.headers}")
                    logger.error(f"Response content (first 100 bytes): {response.content[:100]}")
                    raise UnrealSpeechError(f"API returned unexpected content type: {content_type}")
                
                # Check if we have content
                if len(audio_content) < 100:  # Arbitrary small size check
                    logger.error(f"Audio content too small: {len(audio_content)} bytes")
                    logger.error(f"Audio content: {audio_content}")
                    raise UnrealSpeechError(f"API returned too little audio data: {len(audio_content)} bytes")
                
            except requests.exceptions.RequestException as e:
                # Handle request exceptions specifically
                raise UnrealSpeechError(f"API request error: {str(e)}")
            
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(audio_content)
            
            logger.info(f"Saved audio file to {output_path} ({len(audio_content)} bytes)")
            
            # Extract word timings from response headers if available
            word_timings = []
            has_api_timings = False
            try:
                # Check if response has timestamps in headers
                if "X-Timestamps" in response.headers:
                    timestamps_str = response.headers["X-Timestamps"]
                    logger.info(f"Found timestamps in X-Timestamps header: {timestamps_str[:100]}...")
                    timestamps = json.loads(timestamps_str)
                    word_timings = timestamps
                    has_api_timings = True
                # Check if response has JSON content with timestamps
                elif "application/json" in response.headers.get("Content-Type", ""):
                    data = response.json()
                    if "Timestamps" in data:
                        # Process timestamps based on format
                        timestamps = data["Timestamps"]
                        for item in timestamps:
                            if "Word" in item and "Start" in item and "End" in item:
                                word = item["Word"]
                                start = item["Start"]
                                end = item["End"]
                                word_timings.append((word, start, end))
                        has_api_timings = True
            except Exception as e:
                logger.warning(f"Failed to extract word timings: {e}")
                has_api_timings = False
            
            # Calculate duration using a safer approach
            try:
                # Try to get audio duration using pydub
                audio = AudioSegment.from_file(output_path)
                duration = len(audio) / 1000.0  # Convert ms to seconds
                logger.info(f"Audio duration: {duration:.2f} seconds")
            except Exception as e:
                logger.warning(f"Failed to get audio duration with pydub: {e}")
                # Fallback: approximate duration based on word count
                word_count = len(text.split())
                duration = word_count / 3  # Assume 3 words per second
                logger.warning(f"Using approximate duration based on word count: {duration:.2f} seconds")
            
            # If no word timings were extracted, generate approximate ones
            if not word_timings:
                # Generate approximate timings only
                word_timings = self._generate_approximate_word_timings(text, duration)
                logger.info(f"Generated approximate word timings for {len(word_timings)} words")
            
            # For API-provided timings, use them directly without modification
            # Note: we don't apply numeric expansion to API timings
            if has_api_timings:
                logger.info(f"Using API-provided word timings for {len(word_timings)} words (no numeric expansion)")
            
            # Create result dictionary
            result = {
                "path": str(output_path),
                "word_timings": word_timings,
                "duration": duration
            }
            
            logger.info(f"Generated speech saved to {output_path} (duration: {duration:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"UnrealSpeech generation failed: {str(e)}")
            raise UnrealSpeechError(f"Failed to generate speech: {str(e)}")
    
    def _generate_approximate_word_timings(self, text: str, duration: float) -> List[Tuple[str, float, float]]:
        """
        Generate approximate word timings based on text length and audio duration.
        This is a fallback when API-provided timestamps are not available.
        
        Args:
            text: The text that was converted to speech
            duration: The duration of the audio in seconds
            
        Returns:
            List of (word, start_time, end_time) tuples
        """
        words = text.split()
        if not words:
            return []
            
        # Calculate average word duration
        avg_word_duration = duration / len(words)
        
        # Generate evenly spaced word timings
        word_timings = []
        current_time = 0.0
        
        for word in words:
            # Adjust duration based on word length (longer words take more time)
            word_length_factor = len(word) / 5  # Assuming average word length is 5 characters
            word_length_factor = max(0.5, min(2.0, word_length_factor))  # Limit between 0.5x and 2x
            
            word_duration = avg_word_duration * word_length_factor
            
            # Add word timing
            word_timings.append((word, current_time, current_time + word_duration))
            
            # Move to next word
            current_time += word_duration
        
        # Normalize to ensure the last word ends at the audio duration
        if word_timings:
            last_word_end = word_timings[-1][2]
            if last_word_end != duration:
                scale_factor = duration / last_word_end
                word_timings = [(word, start * scale_factor, end * scale_factor) 
                               for word, start, end in word_timings]
        
        return word_timings

    def map_voice_settings(self, elevenlabs_settings: Dict) -> Dict:
        """
        Map ElevenLabs voice settings to UnrealSpeech settings.
        
        Args:
            elevenlabs_settings: ElevenLabs voice settings
            
        Returns:
            Dict with UnrealSpeech settings
        """
        # Get voice name from settings
        voice_name = elevenlabs_settings.get("voice_name", "")
        
        # For Daniel voice, always use fixed parameters regardless of input
        if voice_name.lower() == "daniel":
            logger.info("Using fixed parameters for Daniel voice (ignoring input settings)")
            return {
                "speed": 0,       # Fixed neutral speed
                "pitch": 1.0,     # Fixed neutral pitch
                "bitrate": "192k" # Fixed high quality bitrate
            }
        
        # Default UnrealSpeech settings for other voices
        unrealspeech_settings = {
            "speed": 0,      # Normal speed (range: -1 to 1)
            "pitch": 1.0,    # Normal pitch (range: 0.5 to 1.5)
            "bitrate": "192k"
        }
        
        # For other voices, use the original mapping
        # Map stability to speed (inverse relationship)
        # ElevenLabs stability: 0-1 (higher = more stable/less expressive)
        # UnrealSpeech speed: -1 to 1 (higher = faster)
        if "stability" in elevenlabs_settings:
            # Convert stability 0-1 to speed -0.5 to 0.5
            # Lower stability (more expressive) maps to slower speed
            stability = elevenlabs_settings["stability"]
            unrealspeech_settings["speed"] = (stability - 0.5) * 0.5
        
        # Map style to pitch (direct relationship)
        # ElevenLabs style: 0-1 (higher = more stylistic variation)
        # UnrealSpeech pitch: 0.5-1.5 (higher = higher pitch)
        if "style" in elevenlabs_settings:
            # Convert style 0-1 to pitch 0.8-1.2
            style = elevenlabs_settings["style"]
            unrealspeech_settings["pitch"] = 0.8 + (style * 0.4)
        
        return unrealspeech_settings 