#!/usr/bin/env python3
"""
AI Image Generation Tool for GAICo

This tool provides AI-powered image generation capabilities, integrating with various
AI image generation APIs and models for creating realistic images for testing and comparison.
"""

import numpy as np
import requests
import json
import base64
import io
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GenerationConfig:
    """Configuration for AI image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    style_preset: Optional[str] = None


class BaseAIGenerator(ABC):
    """Abstract base class for AI image generators."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, config: GenerationConfig) -> List[np.ndarray]:
        """Generate images based on the configuration."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the generator is available/configured."""
        pass


class StableDiffusionGenerator(BaseAIGenerator):
    """Generator using Stable Diffusion API."""
    
    def __init__(self, api_key: str, api_url: str = "https://api.stability.ai"):
        super().__init__(api_key)
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def generate(self, config: GenerationConfig) -> List[np.ndarray]:
        """Generate images using Stability AI API."""
        if not self.is_available():
            raise ValueError("Stability AI API key not configured")
        
        url = f"{self.api_url}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        payload = {
            "text_prompts": [
                {
                    "text": config.prompt,
                    "weight": 1.0
                }
            ],
            "cfg_scale": config.guidance_scale,
            "height": config.height,
            "width": config.width,
            "samples": config.num_images,
            "steps": config.num_inference_steps,
        }
        
        if config.negative_prompt:
            payload["text_prompts"].append({
                "text": config.negative_prompt,
                "weight": -1.0
            })
        
        if config.seed:
            payload["seed"] = config.seed
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            images = []
            
            for artifact in result["artifacts"]:
                image_data = base64.b64decode(artifact["base64"])
                image = Image.open(io.BytesIO(image_data))
                images.append(np.array(image))
            
            return images
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise


class OpenAIGenerator(BaseAIGenerator):
    """Generator using OpenAI DALL-E API."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def generate(self, config: GenerationConfig) -> List[np.ndarray]:
        """Generate images using OpenAI DALL-E API."""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        url = "https://api.openai.com/v1/images/generations"
        
        payload = {
            "prompt": config.prompt,
            "n": config.num_images,
            "size": f"{config.width}x{config.height}",
            "response_format": "url"
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            images = []
            
            for data in result["data"]:
                image_url = data["url"]
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                
                image = Image.open(io.BytesIO(image_response.content))
                images.append(np.array(image))
            
            return images
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise


class HuggingFaceGenerator(BaseAIGenerator):
    """Generator using Hugging Face Inference API."""
    
    def __init__(self, api_key: str, model_id: str = "stabilityai/stable-diffusion-2-1"):
        super().__init__(api_key)
        self.model_id = model_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def generate(self, config: GenerationConfig) -> List[np.ndarray]:
        """Generate images using Hugging Face Inference API."""
        if not self.is_available():
            raise ValueError("Hugging Face API key not configured")
        
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        payload = {
            "inputs": config.prompt,
            "parameters": {
                "negative_prompt": config.negative_prompt,
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "num_images_per_prompt": config.num_images
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Hugging Face returns image data directly
            images = []
            if isinstance(response.content, bytes):
                image = Image.open(io.BytesIO(response.content))
                images.append(np.array(image))
            else:
                # Handle multiple images
                for image_data in response.json():
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    images.append(np.array(image))
            
            return images
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise


class LocalStableDiffusionGenerator(BaseAIGenerator):
    """Generator using local Stable Diffusion installation."""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the local Stable Diffusion model."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            if self.model_path:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.float16
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16
                )
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
        except ImportError:
            self.logger.warning("diffusers library not available. Install with: pip install diffusers transformers")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def is_available(self) -> bool:
        return self.pipeline is not None
    
    def generate(self, config: GenerationConfig) -> List[np.ndarray]:
        """Generate images using local Stable Diffusion."""
        if not self.is_available():
            raise ValueError("Local Stable Diffusion model not loaded")
        
        try:
            generator = None
            if config.seed:
                import torch
                generator = torch.Generator().manual_seed(config.seed)
            
            result = self.pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_images,
                generator=generator
            )
            
            images = []
            for image in result.images:
                images.append(np.array(image))
            
            return images
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise


class AIImageGenerator:
    """
    Main AI Image Generator tool that can use multiple backends.
    
    Features:
    - Multiple AI generation backends (Stability AI, OpenAI, Hugging Face, Local)
    - Batch generation capabilities
    - Prompt templates and variations
    - Integration with GAICo testing framework
    """
    
    def __init__(self, output_dir: str = "ai_generated_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.generators = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def add_generator(self, name: str, generator: BaseAIGenerator):
        """Add an AI generator backend."""
        self.generators[name] = generator
        self.logger.info(f"Added generator: {name}")
    
    def list_generators(self) -> List[str]:
        """List available generators."""
        return list(self.generators.keys())
    
    def generate_images(
        self, 
        prompt: str,
        generator_name: str = None,
        num_images: int = 1,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Generate images using the specified or available generator.
        
        Args:
            prompt: Text prompt for image generation
            generator_name: Name of the generator to use (if None, uses first available)
            num_images: Number of images to generate
            **kwargs: Additional configuration parameters
            
        Returns:
            List of generated images as numpy arrays
        """
        if not self.generators:
            raise ValueError("No AI generators configured")
        
        if generator_name is None:
            # Use first available generator
            for name, generator in self.generators.items():
                if generator.is_available():
                    generator_name = name
                    break
            else:
                raise ValueError("No available generators found")
        
        if generator_name not in self.generators:
            raise ValueError(f"Generator '{generator_name}' not found")
        
        generator = self.generators[generator_name]
        if not generator.is_available():
            raise ValueError(f"Generator '{generator_name}' is not available")
        
        # Create configuration
        config = GenerationConfig(
            prompt=prompt,
            num_images=num_images,
            **kwargs
        )
        
        self.logger.info(f"Generating {num_images} images with {generator_name}")
        return generator.generate(config)
    
    def generate_test_dataset(
        self,
        prompts: List[str],
        generator_name: str = None,
        images_per_prompt: int = 3,
        **kwargs
    ) -> Dict[str, List[np.ndarray]]:
        """
        Generate a test dataset with multiple prompts.
        
        Args:
            prompts: List of text prompts
            generator_name: Generator to use
            images_per_prompt: Number of images per prompt
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary mapping prompts to lists of generated images
        """
        dataset = {}
        
        for prompt in prompts:
            self.logger.info(f"Generating images for prompt: {prompt}")
            try:
                images = self.generate_images(
                    prompt, 
                    generator_name, 
                    images_per_prompt, 
                    **kwargs
                )
                dataset[prompt] = images
            except Exception as e:
                self.logger.error(f"Failed to generate images for '{prompt}': {e}")
                dataset[prompt] = []
        
        return dataset
    
    def generate_gaico_test_data(
        self,
        base_prompts: List[str],
        generator_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate test data specifically for GAICo image metrics testing.
        
        Args:
            base_prompts: List of base prompts to generate reference images
            generator_name: Generator to use
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with test data for GAICo
        """
        test_data = {
            "reference_images": [],
            "generated_images": [],
            "similar_images": [],
            "different_images": []
        }
        
        for prompt in base_prompts:
            try:
                # Generate reference image
                ref_images = self.generate_images(prompt, generator_name, 1, **kwargs)
                if ref_images:
                    test_data["reference_images"].append(ref_images[0])
                
                # Generate similar image (same prompt, different seed)
                similar_images = self.generate_images(
                    prompt, generator_name, 1, 
                    seed=np.random.randint(0, 1000000), **kwargs
                )
                if similar_images:
                    test_data["similar_images"].append(similar_images[0])
                
                # Generate different image (different prompt)
                different_prompt = f"different style: {prompt}"
                different_images = self.generate_images(
                    different_prompt, generator_name, 1, **kwargs
                )
                if different_images:
                    test_data["different_images"].append(different_images[0])
                
                # Generate "AI-generated" version (slight prompt variation)
                gen_prompt = f"AI generated: {prompt}"
                gen_images = self.generate_images(
                    gen_prompt, generator_name, 1, **kwargs
                )
                if gen_images:
                    test_data["generated_images"].append(gen_images[0])
                    
            except Exception as e:
                self.logger.error(f"Failed to generate test data for '{prompt}': {e}")
        
        return test_data
    
    def save_images(self, images: List[np.ndarray], prefix: str = "ai_gen") -> List[str]:
        """
        Save generated images to disk.
        
        Args:
            images: List of images to save
            prefix: Prefix for filenames
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for i, img in enumerate(images):
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}_{i+1:03d}.png"
            filepath = self.output_dir / filename
            
            pil_img = Image.fromarray(img)
            pil_img.save(filepath)
            saved_paths.append(str(filepath))
            
            self.logger.info(f"Saved image: {filepath}")
        
        return saved_paths
    
    def save_dataset(self, dataset: Dict[str, List[np.ndarray]], prefix: str = "dataset") -> Dict[str, List[str]]:
        """
        Save a dataset of generated images.
        
        Args:
            dataset: Dictionary mapping prompts to lists of images
            prefix: Prefix for filenames
            
        Returns:
            Dictionary mapping prompts to lists of saved file paths
        """
        saved_paths = {}
        
        for prompt, images in dataset.items():
            # Create safe filename from prompt
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt[:50]  # Limit length
            
            saved_paths[prompt] = []
            for i, img in enumerate(images):
                timestamp = int(time.time())
                filename = f"{prefix}_{safe_prompt}_{timestamp}_{i+1:03d}.png"
                filepath = self.output_dir / filename
                
                pil_img = Image.fromarray(img)
                pil_img.save(filepath)
                saved_paths[prompt].append(str(filepath))
        
        return saved_paths


def setup_generators_from_env() -> AIImageGenerator:
    """
    Setup AI generators from environment variables.
    
    Environment variables:
    - STABILITY_API_KEY: Stability AI API key
    - OPENAI_API_KEY: OpenAI API key
    - HF_API_KEY: Hugging Face API key
    - SD_MODEL_PATH: Local Stable Diffusion model path
    """
    generator = AIImageGenerator()
    
    # Stability AI
    stability_key = os.getenv("STABILITY_API_KEY")
    if stability_key:
        try:
            sd_generator = StableDiffusionGenerator(stability_key)
            generator.add_generator("stability", sd_generator)
        except Exception as e:
            print(f"Failed to setup Stability AI generator: {e}")
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            openai_generator = OpenAIGenerator(openai_key)
            generator.add_generator("openai", openai_generator)
        except Exception as e:
            print(f"Failed to setup OpenAI generator: {e}")
    
    # Hugging Face
    hf_key = os.getenv("HF_API_KEY")
    if hf_key:
        try:
            hf_generator = HuggingFaceGenerator(hf_key)
            generator.add_generator("huggingface", hf_generator)
        except Exception as e:
            print(f"Failed to setup Hugging Face generator: {e}")
    
    # Local Stable Diffusion
    sd_path = os.getenv("SD_MODEL_PATH")
    try:
        local_sd = LocalStableDiffusionGenerator(sd_path)
        if local_sd.is_available():
            generator.add_generator("local_sd", local_sd)
    except Exception as e:
        print(f"Failed to setup local Stable Diffusion: {e}")
    
    return generator


def main():
    """Example usage of the AI Image Generator tool."""
    print("Setting up AI Image Generator...")
    
    # Setup generators from environment
    generator = setup_generators_from_env()
    
    if not generator.generators:
        print("No AI generators configured. Please set up API keys or install local models.")
        print("Available environment variables:")
        print("  - STABILITY_API_KEY: Stability AI API key")
        print("  - OPENAI_API_KEY: OpenAI API key")
        print("  - HF_API_KEY: Hugging Face API key")
        print("  - SD_MODEL_PATH: Local Stable Diffusion model path")
        return
    
    print(f"Available generators: {generator.list_generators()}")
    
    # Example prompts for testing
    test_prompts = [
        "A beautiful landscape with mountains and a lake",
        "A futuristic city skyline at night",
        "A cute cartoon cat playing with a ball",
        "An abstract painting with vibrant colors",
        "A professional portrait of a business person"
    ]
    
    # Generate test dataset
    print("Generating test dataset...")
    try:
        dataset = generator.generate_test_dataset(
            test_prompts[:2],  # Use first 2 prompts for demo
            images_per_prompt=2
        )
        
        # Save dataset
        saved_paths = generator.save_dataset(dataset, "ai_test")
        
        print(f"Generated {sum(len(images) for images in dataset.values())} images")
        print(f"Images saved to: {generator.output_dir}")
        
        for prompt, paths in saved_paths.items():
            print(f"  {prompt}: {len(paths)} images")
            
    except Exception as e:
        print(f"Generation failed: {e}")
        print("This might be due to missing API keys or network issues.")


if __name__ == "__main__":
    main() 