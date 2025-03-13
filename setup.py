from setuptools import setup, find_packages

setup(
    name="video_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx==0.26.0",
        "moviepy==1.0.3",
        "python-dotenv==1.0.0",
        "pydub==0.25.1",
        "elevenlabs==0.2.27",
        "tenacity==8.2.3",
        "pytest==7.4.4",
        "pytest-asyncio==0.23.3"
    ],
) 