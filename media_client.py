from abc import ABC, abstractmethod
from typing import List, Optional


class GenerativeMediaClient(ABC):
    """Abstract base class for generative media clients.
    
    This class defines a common interface for generating media (images, video, etc.)
    from text prompts. Concrete implementations should handle the specifics of
    interacting with different generative AI models or services.
    """
    
    @abstractmethod
    def generate_media(self, prompt: str, config: Optional[dict] = None) -> List[str]:
        """Generate media files based on the given prompt and optional configuration.
        
        Args:
            prompt (str): The text prompt describing the desired media output.
            config (Optional[dict]): Additional configuration parameters for the generation
                process. Defaults to None.
        
        Returns:
            List[str]: A list of paths to the generated media files. These can be either
                local file paths or remote URIs depending on the implementation.
                
        Raises:
            NotImplementedError: This is an abstract method that must be implemented
                by concrete subclasses.
        """
        raise NotImplementedError

# Example usage:
"""
# Example implementation for a specific model/service
class StableDiffusionClient(GenerativeMediaClient):
    def generate_media(self, prompt: str, config: dict = None) -> List[str]:
        # Implementation specific to Stable Diffusion
        config = config or {}
        # ... generate images ...
        return ['/path/to/generated/image1.png', '/path/to/generated/image2.png']

# Using in a pipeline
client = StableDiffusionClient()
media_paths = client.generate_media(
    prompt="A serene landscape at sunset",
    config={"num_images": 2, "guidance_scale": 7.5}
)
""" 