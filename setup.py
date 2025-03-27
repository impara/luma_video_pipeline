from setuptools import setup, find_packages

setup(
    name="video_pipeline",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "httpx>=0.28.1,<1.0.0",
        "moviepy==1.0.3",
        "python-dotenv==1.0.0",
        "pydub==0.25.1",
        "elevenlabs==1.54.0",
        "tenacity==8.2.3",
        "replicate==0.22.0",
        "requests==2.31.0",
        "urllib3==2.0.7",
        "playsound==1.3.0",  # UnrealSpeech dependency
        "google-api-python-client==2.108.0",  # YouTube API
        "google-auth-httplib2==0.1.1",  # YouTube API
        "google-auth-oauthlib==1.1.0",  # YouTube API
        "Pillow==10.0.0",  # Image processing
        "google-genai==1.5.0",  # Google Gemini
        "loguru==0.7.3",  # Logging
        "psutil>=5.9.0",  # Memory monitoring
    ],
    extras_require={
        "dev": [
            "pytest==7.4.4",
            "pytest-asyncio==0.23.3",
        ]
    }
) 