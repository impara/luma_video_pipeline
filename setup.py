from setuptools import setup, find_packages

setup(
    name="luma_video_pipeline",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "replicate",
        "elevenlabs",
        "moviepy",
        "pydub",
        "python-dotenv",
        "pillow",
        "google-generativeai",
        "tenacity",
        "httpx",
    ]
) 