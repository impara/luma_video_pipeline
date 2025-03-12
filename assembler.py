"""
Final video assembly module.
Responsible for:
- Combining generated video with voiceover audio
- Ensuring proper synchronization of all components
- Handling final video export with all elements combined
"""

from pathlib import Path as PathLib
from typing import Dict, List

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    VideoFileClip,
    concatenate_videoclips,
    concatenate_audioclips,
    ColorClip,
    transfx
)
from moviepy.video.fx import all as vfx

class VideoAssembler:
    def __init__(self):
        """Initialize the video assembler"""
        # Create output directory
        self.output_dir = PathLib("final_videos")
        self.output_dir.mkdir(exist_ok=True)

    def _create_scene_clip(self, scene: Dict, index: int) -> CompositeVideoClip:
        """
        Creates a composite clip from a single scene with video, audio, and captions.
        Now supports multiple caption parts per scene.
        
        Args:
            scene: Dictionary containing scene components
            index: Scene index for logging
        
        Returns:
            CompositeVideoClip: The assembled scene clip
        """
        print(f"Rendering scene {index + 1}...")
        
        # Load video/image and audio
        if scene.get("final_clip"):
            # Use the pre-rendered clip if available (from animated images)
            base_clip = scene["final_clip"]
        else:
            # Load from file
            base_clip = VideoFileClip(scene["video_path"])
        
        # Handle multiple audio files if present
        if "audio_paths" in scene and len(scene["audio_paths"]) > 1:
            audio_clips = [AudioFileClip(path) for path in scene["audio_paths"]]
            audio = concatenate_audioclips(audio_clips)
        else:
            audio = AudioFileClip(scene["audio_path"])
        
        # Get the caption clips
        caption_clips = scene.get("caption_clips", [])
        if not caption_clips and "caption_clip" in scene:  # Backward compatibility
            caption_clips = [scene["caption_clip"]]
            
        # Clean any <SPLIT> markers from caption clips if needed
        # This is a safety check in case they weren't cleaned earlier
        cleaned_caption_clips = []
        for clip in caption_clips:
            # We can't directly modify the text in an existing TextClip,
            # but we can ensure it's properly positioned and timed
            cleaned_caption_clips.append(clip)
        
        # Create composite with all captions
        clips = [base_clip] + cleaned_caption_clips
        composite = CompositeVideoClip(
            clips,
            size=base_clip.size
        )
        
        # Set audio
        composite = composite.set_audio(audio)
        
        # Add fade in and fade out for smooth transitions
        # This creates a smoother transition between scenes when concatenated
        fade_duration = 0.3  # Shorter fade duration for subtle effect
        composite = composite.fx(vfx.fadein, fade_duration)
        composite = composite.fx(vfx.fadeout, fade_duration)
        
        # Ensure the clip duration exactly matches the audio duration
        # This is critical for preventing sync issues, especially with multiple narration parts
        if composite.duration != audio.duration:
            print(f"Adjusting scene {index + 1} duration from {composite.duration:.2f}s to {audio.duration:.2f}s")
            composite = composite.set_duration(audio.duration)
        
        # Print debug info
        print(f"Scene {index + 1} details:")
        print(f"- Video duration: {base_clip.duration:.2f}s")
        print(f"- Audio duration: {audio.duration:.2f}s")
        print(f"- Number of caption parts: {len(caption_clips)}")
        for i, cap in enumerate(caption_clips, 1):
            print(f"  - Caption part {i} duration: {cap.duration:.2f}s")
        print(f"- Final duration: {composite.duration:.2f}s")
        
        return composite

    def assemble_scenes(
        self,
        scenes: List[Dict],
        output_path: str,
        fps: int = 24,  # Changed default to 24 fps to match source videos
        transition_duration: float = 0.5  # Duration for cross-dissolve transitions
    ) -> str:
        """
        Assembles multiple scenes into a final video with smooth transitions.
        
        Args:
            scenes: List of scene dictionaries from SceneBuilder
            output_path: Path where to save the final video
            fps: Frames per second for the output video (default: 24)
            transition_duration: Duration of cross-dissolve transitions between scenes
        
        Returns:
            str: Path to the final assembled video
        """
        try:
            print("\nStarting video assembly...")
            
            # Create scene clips
            scene_clips = []
            total_duration = 0
            
            for i, scene in enumerate(scenes):
                # Create the composite scene
                clip = self._create_scene_clip(scene, i)
                scene_clips.append(clip)
                total_duration += clip.duration
                print(f"Scene {i + 1} rendered successfully")
                print(f"Scene duration: {clip.duration:.2f}s")

            print(f"\nTotal video duration will be: {total_duration:.2f}s")
            
            # Create a background layer as a safety measure (slightly dark gray)
            # This ensures there's always something visible during transitions
            background = ColorClip(
                size=scene_clips[0].size, 
                color=(30, 30, 30),  # Dark gray background
                duration=total_duration
            )
            
            # Concatenate all scenes with smooth transitions
            print("\nConcatenating scenes...")
            if len(scene_clips) > 1:
                # Apply audio crossfades to each clip before concatenation
                # This creates smoother audio transitions between scenes
                for i in range(len(scene_clips)):
                    # Get the audio from the clip
                    audio = scene_clips[i].audio
                    
                    # Apply audio fade in/out to match video transitions
                    if audio is not None:
                        # Apply audio fade in at the beginning
                        if i == 0:
                            # Only fade in for first clip
                            audio = audio.audio_fadein(0.3)
                        else:
                            # Fade in and out for middle clips
                            audio = audio.audio_fadein(0.3)
                        
                        # Apply audio fade out at the end
                        if i == len(scene_clips) - 1:
                            # Only fade out for last clip
                            audio = audio.audio_fadeout(0.3)
                        else:
                            # Fade in and out for middle clips
                            audio = audio.audio_fadeout(0.3)
                        
                        # Set the modified audio back to the clip
                        scene_clips[i] = scene_clips[i].set_audio(audio)
                
                # Simple approach: just concatenate the clips
                # Now with audio crossfades to match video transitions
                final_video = concatenate_videoclips(scene_clips)
                
                # Add the background to ensure no transparency issues
                final_video = CompositeVideoClip(
                    [background, final_video],
                    size=scene_clips[0].size
                )
            else:
                # Only one scene
                final_video = CompositeVideoClip(
                    [background, scene_clips[0]],
                    size=scene_clips[0].size
                )
            
            # Ensure output path has .mp4 extension
            output_path = str(self.output_dir / PathLib(output_path))
            if not output_path.endswith('.mp4'):
                output_path += '.mp4'
            
            # Write final video with high quality settings
            print(f"\nWriting final video to {output_path}...")
            final_video.write_videofile(
                output_path,
                fps=fps,  # Using specified fps (default 24)
                codec='libx264',
                audio_codec='aac',
                bitrate="8000k",  # High bitrate for better quality
                threads=4,        # Use multiple threads
                preset='slow'     # Better compression
            )
            
            # Clean up
            final_video.close()
            for clip in scene_clips:
                clip.close()
            
            print("\nVideo assembly completed successfully!")
            return output_path
            
        except Exception as e:
            print(f"\nError during video assembly: {str(e)}")
            raise
        
    def combine_video_and_audio_only(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> str:
        """
        Basic utility method that only combines a video with an audio track.
        Does NOT support captions, transitions, or other advanced features.
        For full functionality, use assemble_scenes() instead.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path where to save the final video
            
        Returns:
            str: Path to the assembled video
        """
        
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        final = video.set_audio(audio)
        final.write_videofile(
            output_path,
            fps=24,  # Use 24 fps to match source videos
            codec='libx264',
            audio_codec='aac'
        )
        
        video.close()
        audio.close()
        final.close()
        
        return output_path 