# YouTube Upload Functionality

The Video Pipeline now includes automated YouTube upload capabilities for both standard videos and YouTube Shorts.

## Standalone YouTube Uploader

The pipeline uses a standalone uploader approach, which gives you more control by separating video generation from uploading:

1. Generate your videos first using the main pipeline
2. Review the videos to ensure quality
3. Upload only when you're satisfied

## Setup

1. **Create YouTube API Credentials**:

   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the YouTube Data API v3
   - Create OAuth 2.0 credentials:
     - Go to "APIs & Services" > "Credentials"
     - Click "Create Credentials" > "OAuth client ID"
     - Select "Desktop app" as the application type
     - Download the credentials JSON file and save it as `client_secret.json` in your project directory

2. **Install Required Dependencies**:

   The YouTube uploader requires several Python packages:

   - `google-api-python-client`: For YouTube API access
   - `google-auth-httplib2` and `google-auth-oauthlib`: For authentication
   - `Pillow`: For image processing (thumbnail generation)

   Install them with pip:

   ```bash
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib Pillow
   ```

   Or update your existing environment:

   ```bash
   pip install -r requirements.txt
   ```

## Creating YouTube Metadata Files

Create a text file with the following sections to define your video's metadata:

```
# Title
Your Video Title Here

# Description
Your video description here.
You can include multiple lines.
Don't forget to add relevant keywords and links.

# Tags
tag1, tag2, tag3, AI generated, Video Pipeline

# Category
22

# Privacy
private

# Thumbnail
A description for generating a thumbnail image
```

### The Thumbnail Section

The `Thumbnail` section is used to provide a description for generating a custom thumbnail:

- When using `--generate-thumbnail`, this description is used to create a thumbnail image using SDXL
- The description should be detailed and specific about what you want in the thumbnail
- Example: "A beautiful sunset over mountains with vibrant orange and purple colors"

### YouTube Category IDs

Common category IDs:

- 1: Film & Animation
- 2: Autos & Vehicles
- 10: Music
- 15: Pets & Animals
- 17: Sports
- 20: Gaming
- 22: People & Blogs
- 23: Comedy
- 24: Entertainment
- 25: News & Politics
- 26: Howto & Style
- 27: Education
- 28: Science & Technology

### Privacy Settings

- `public`: Visible to everyone
- `unlisted`: Visible only to people with the link
- `private`: Visible only to you (default)

## Basic Usage

```bash
# Upload a standard landscape video
python integrations/youtube/uploader.py --video-path output/videos/kufr_concept.mp4 --metadata youtube_metadata.txt --generate-thumbnail

# Upload a YouTube Shorts video
python integrations/youtube/uploader.py --video-path output/videos/kufr_concept.mp4 --metadata youtube_metadata.txt --generate-thumbnail --shorts
```

## Thumbnail Options

You have two options for thumbnails:

1. **Generate a thumbnail** using the description in your metadata file:

```bash
python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt --generate-thumbnail
```

2. **Use an existing image** as the thumbnail:

```bash
python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt --thumbnail my_thumbnail.jpg
```

You can customize the generated thumbnail dimensions:

```bash
python integrations/youtube/uploader.py --video-path output/videos/my_video.mp4 --metadata youtube_metadata.txt --generate-thumbnail --thumbnail-width 1280 --thumbnail-height 720
```

## Benefits of the Standalone Approach

- Review videos before uploading
- Upload only when you're satisfied with the result
- Retry uploads without regenerating videos
- Upload previously generated videos at any time
- Cleaner separation of concerns

## Authentication Flow

The first time you run an upload, the script will:

1. Open a browser window for you to log in to your Google account
2. Ask for permission to upload videos to your YouTube channel
3. Save the authentication token locally for future uploads

## Best Practices for YouTube Shorts

1. **Keep Videos Short**: Aim for 15-60 seconds
2. **Use Vertical Format**: Always use `--video-format short` for 9:16 aspect ratio when generating videos
3. **Optimize Captions**: Use TikTok-style captions with `--style tiktok_neon` or similar
4. **Include Hashtags**: Always include #Shorts in your metadata
5. **Engaging Titles**: Keep titles short but descriptive
6. **Metadata Optimization**:
   - Include relevant keywords in your description
   - Use 3-5 relevant tags
   - Set appropriate category (Education, Entertainment, etc.)
7. **Start Private**: Upload as private first to verify everything looks correct
8. **Custom Thumbnails**: Create eye-catching thumbnails with the `--generate-thumbnail` option

## Thumbnail Best Practices

1. **Be Specific**: Provide detailed descriptions for generated thumbnails
2. **Include Key Elements**: Mention the main subject, colors, mood, and composition
3. **Optimal Dimensions**: YouTube recommends 1280×720 pixels (16:9 aspect ratio)
4. **Eye-Catching**: Use vibrant colors and clear composition
5. **Relevant**: Make sure the thumbnail represents the video content
6. **No Text**: The AI doesn't generate text well, so avoid requesting text in thumbnails
7. **Review Before Upload**: Generated thumbnails are saved in the `thumbnails` directory

## Troubleshooting

- **Authentication Issues**: Delete `youtube_token.json` and try again
- **Upload Failures**: Check your internet connection and video file integrity
- **Processing Issues**: Some videos may take time to process on YouTube's end
- **Quota Limits**: YouTube API has daily quota limits; spread uploads across days if needed
- **Thumbnail Generation Failures**: If thumbnail generation fails, try a simpler description or use an existing image

## Recommended Workflow

1. Create your script file (`script.txt`)
2. Create your YouTube metadata file (`youtube_metadata.txt`) with a good thumbnail description
3. Generate your video:

```bash
python main.py \
    --media-type image \
    --script-file script.txt \
    --video-format short \
    --style tiktok_neon
```

4. Review the generated video in the `final_videos` directory
5. Upload when satisfied, with a generated thumbnail:

```bash
python integrations/youtube/uploader.py \
    --video-path output/videos/your_video.mp4 \
    --metadata youtube_metadata.txt \
    --shorts \
    --generate-thumbnail
```

6. Check the console output for the video URL
7. Verify the video and thumbnail on YouTube and adjust privacy settings if needed
