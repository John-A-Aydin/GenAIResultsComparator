#!/usr/bin/env python3
"""
Image Generation Tool for GAICo

This tool provides various methods to create images for testing, comparison, and visualization purposes.
It can generate synthetic images, create test patterns, and integrate with AI image generation APIs.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import os
from typing import Tuple, List, Optional, Dict, Any
import json
from pathlib import Path


class ImageGenerator:
    """
    A comprehensive tool for generating various types of images.
    
    Features:
    - Synthetic image generation (geometric shapes, patterns, noise)
    - Test image creation for AI model evaluation
    - Integration with AI image generation APIs
    - Image manipulation and processing
    - Batch image generation for testing
    """
    
    def __init__(self, output_dir: str = "generated_images"):
        """
        Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_synthetic_image(
        self, 
        shape: Tuple[int, int] = (512, 512), 
        image_type: str = "geometric",
        **kwargs
    ) -> np.ndarray:
        """
        Generate synthetic images for testing and comparison.
        
        Args:
            shape: Image dimensions (height, width)
            image_type: Type of image to generate
                - "geometric": Geometric shapes
                - "noise": Random noise
                - "gradient": Color gradients
                - "text": Text-based images
                - "pattern": Repeating patterns
            **kwargs: Additional parameters for specific image types
            
        Returns:
            numpy array of the generated image
        """
        if image_type == "geometric":
            return self._generate_geometric_image(shape, **kwargs)
        elif image_type == "noise":
            return self._generate_noise_image(shape, **kwargs)
        elif image_type == "gradient":
            return self._generate_gradient_image(shape, **kwargs)
        elif image_type == "text":
            return self._generate_text_image(shape, **kwargs)
        elif image_type == "pattern":
            return self._generate_pattern_image(shape, **kwargs)
        else:
            raise ValueError(f"Unknown image type: {image_type}")
    
    def _generate_geometric_image(
        self, 
        shape: Tuple[int, int], 
        num_shapes: int = 5,
        colors: List[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Generate image with random geometric shapes."""
        img = np.zeros((*shape, 3), dtype=np.uint8)
        
        if colors is None:
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255)
            ]
        
        for _ in range(num_shapes):
            # Random shape type
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
            color = colors[np.random.randint(0, len(colors))]
            
            if shape_type == 'circle':
                center = (np.random.randint(50, shape[1]-50), 
                         np.random.randint(50, shape[0]-50))
                radius = np.random.randint(20, 100)
                self._draw_circle(img, center, radius, color)
                
            elif shape_type == 'rectangle':
                x1 = np.random.randint(0, shape[1]-100)
                y1 = np.random.randint(0, shape[0]-100)
                x2 = x1 + np.random.randint(50, 100)
                y2 = y1 + np.random.randint(50, 100)
                self._draw_rectangle(img, (x1, y1), (x2, y2), color)
                
            elif shape_type == 'triangle':
                points = [
                    (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
                    for _ in range(3)
                ]
                self._draw_triangle(img, points, color)
        
        return img
    
    def _generate_noise_image(
        self, 
        shape: Tuple[int, int], 
        noise_type: str = "gaussian",
        intensity: float = 0.5
    ) -> np.ndarray:
        """Generate noise-based images."""
        if noise_type == "gaussian":
            noise = np.random.normal(0, intensity * 255, (*shape, 3))
            img = np.clip(noise, 0, 255).astype(np.uint8)
        elif noise_type == "uniform":
            img = np.random.randint(0, 256, (*shape, 3), dtype=np.uint8)
        elif noise_type == "salt_pepper":
            img = np.random.randint(0, 256, (*shape, 3), dtype=np.uint8)
            # Add salt and pepper noise
            mask = np.random.random(shape) < intensity
            img[mask] = np.random.choice([0, 255], size=(mask.sum(), 3))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return img
    
    def _generate_gradient_image(
        self, 
        shape: Tuple[int, int], 
        gradient_type: str = "linear",
        colors: List[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Generate gradient images."""
        if colors is None:
            colors = [(255, 0, 0), (0, 0, 255)]  # Red to blue
        
        img = np.zeros((*shape, 3), dtype=np.uint8)
        
        if gradient_type == "linear":
            for i in range(shape[0]):
                for j in range(shape[1]):
                    t = j / shape[1]  # Interpolation parameter
                    pixel = []
                    for c in range(3):
                        pixel.append(int(colors[0][c] * (1-t) + colors[1][c] * t))
                    img[i, j] = pixel
                    
        elif gradient_type == "radial":
            center = (shape[1] // 2, shape[0] // 2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    dist = np.sqrt((i - center[1])**2 + (j - center[0])**2)
                    t = dist / max_dist
                    pixel = []
                    for c in range(3):
                        pixel.append(int(colors[0][c] * (1-t) + colors[1][c] * t))
                    img[i, j] = pixel
        
        return img
    
    def _generate_text_image(
        self, 
        shape: Tuple[int, int], 
        text: str = "Sample Text",
        font_size: int = 48,
        color: Tuple[int, int, int] = (0, 0, 0),
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Generate text-based images."""
        # Create PIL image
        pil_img = Image.new('RGB', (shape[1], shape[0]), background_color)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (shape[1] - text_width) // 2
        y = (shape[0] - text_height) // 2
        
        draw.text((x, y), text, fill=color, font=font)
        
        return np.array(pil_img)
    
    def _generate_pattern_image(
        self, 
        shape: Tuple[int, int], 
        pattern_type: str = "checkerboard",
        tile_size: int = 50
    ) -> np.ndarray:
        """Generate pattern-based images."""
        img = np.zeros((*shape, 3), dtype=np.uint8)
        
        if pattern_type == "checkerboard":
            for i in range(0, shape[0], tile_size):
                for j in range(0, shape[1], tile_size):
                    if (i // tile_size + j // tile_size) % 2 == 0:
                        img[i:i+tile_size, j:j+tile_size] = [255, 255, 255]
                    else:
                        img[i:i+tile_size, j:j+tile_size] = [0, 0, 0]
                        
        elif pattern_type == "stripes":
            for i in range(0, shape[0], tile_size * 2):
                img[i:i+tile_size, :] = [255, 255, 255]
                img[i+tile_size:i+2*tile_size, :] = [0, 0, 0]
        
        return img
    
    def _draw_circle(self, img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]):
        """Draw a circle on the image."""
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = color
    
    def _draw_rectangle(self, img: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a rectangle on the image."""
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = color
    
    def _draw_triangle(self, img: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int]):
        """Draw a triangle on the image."""
        # Simple triangle drawing using PIL
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.polygon(points, fill=color)
        img[:] = np.array(pil_img)
    
    def generate_test_dataset(
        self, 
        num_images: int = 10, 
        base_shape: Tuple[int, int] = (256, 256),
        variations: List[str] = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        Generate a dataset of test images for AI model evaluation.
        
        Args:
            num_images: Number of images to generate per category
            base_shape: Base image dimensions
            variations: List of image variations to generate
            
        Returns:
            Dictionary with image categories as keys and lists of images as values
        """
        if variations is None:
            variations = ["geometric", "noise", "gradient", "text", "pattern"]
        
        dataset = {}
        
        for variation in variations:
            dataset[variation] = []
            for i in range(num_images):
                # Add some randomness to parameters
                if variation == "geometric":
                    img = self.generate_synthetic_image(
                        base_shape, variation, 
                        num_shapes=np.random.randint(3, 8)
                    )
                elif variation == "noise":
                    img = self.generate_synthetic_image(
                        base_shape, variation,
                        noise_type=np.random.choice(["gaussian", "uniform", "salt_pepper"]),
                        intensity=np.random.uniform(0.3, 0.7)
                    )
                elif variation == "gradient":
                    img = self.generate_synthetic_image(
                        base_shape, variation,
                        gradient_type=np.random.choice(["linear", "radial"])
                    )
                elif variation == "text":
                    img = self.generate_synthetic_image(
                        base_shape, variation,
                        text=f"Text {i+1}",
                        font_size=np.random.randint(24, 72)
                    )
                elif variation == "pattern":
                    img = self.generate_synthetic_image(
                        base_shape, variation,
                        pattern_type=np.random.choice(["checkerboard", "stripes"]),
                        tile_size=np.random.randint(20, 80)
                    )
                
                dataset[variation].append(img)
        
        return dataset
    
    def save_image(self, img: np.ndarray, filename: str, format: str = "PNG") -> str:
        """
        Save an image to disk.
        
        Args:
            img: Image as numpy array
            filename: Output filename
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Path to saved image
        """
        filepath = self.output_dir / filename
        pil_img = Image.fromarray(img)
        pil_img.save(filepath, format=format)
        return str(filepath)
    
    def save_dataset(self, dataset: Dict[str, List[np.ndarray]], prefix: str = "test") -> Dict[str, List[str]]:
        """
        Save a dataset of images to disk.
        
        Args:
            dataset: Dictionary of image categories and lists of images
            prefix: Prefix for filenames
            
        Returns:
            Dictionary mapping categories to lists of saved file paths
        """
        saved_paths = {}
        
        for category, images in dataset.items():
            saved_paths[category] = []
            for i, img in enumerate(images):
                filename = f"{prefix}_{category}_{i+1:03d}.png"
                filepath = self.save_image(img, filename)
                saved_paths[category].append(filepath)
        
        return saved_paths
    
    def create_comparison_images(self, base_image: np.ndarray, num_variations: int = 5) -> List[np.ndarray]:
        """
        Create variations of a base image for comparison testing.
        
        Args:
            base_image: Base image to create variations from
            num_variations: Number of variations to create
            
        Returns:
            List of image variations
        """
        variations = []
        
        for i in range(num_variations):
            if i == 0:
                # Original image
                variations.append(base_image.copy())
            elif i == 1:
                # Add noise
                noise = np.random.normal(0, 30, base_image.shape).astype(np.int16)
                noisy_img = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                variations.append(noisy_img)
            elif i == 2:
                # Brightness adjustment
                bright_img = np.clip(base_image * 1.3, 0, 255).astype(np.uint8)
                variations.append(bright_img)
            elif i == 3:
                # Contrast adjustment
                contrast_img = np.clip((base_image - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
                variations.append(contrast_img)
            elif i == 4:
                # Blur effect (simple box blur)
                from scipy.ndimage import uniform_filter
                blurred_img = uniform_filter(base_image, size=3)
                variations.append(blurred_img.astype(np.uint8))
        
        return variations
    
    def generate_for_gaico_testing(self, num_sets: int = 3) -> Dict[str, Any]:
        """
        Generate images specifically for testing GAICo image metrics.
        
        Args:
            num_sets: Number of test sets to generate
            
        Returns:
            Dictionary with test data for GAICo
        """
        test_data = {
            "reference_images": [],
            "generated_images": [],
            "similar_images": [],
            "different_images": []
        }
        
        for i in range(num_sets):
            # Generate base reference image
            ref_img = self.generate_synthetic_image((256, 256), "geometric", num_shapes=5)
            test_data["reference_images"].append(ref_img)
            
            # Generate similar image (same type, different parameters)
            similar_img = self.generate_synthetic_image((256, 256), "geometric", num_shapes=5)
            test_data["similar_images"].append(similar_img)
            
            # Generate different image (different type)
            different_img = self.generate_synthetic_image((256, 256), "noise", noise_type="gaussian")
            test_data["different_images"].append(different_img)
            
            # Generate "AI-generated" version (variation of reference)
            variations = self.create_comparison_images(ref_img, 1)
            test_data["generated_images"].append(variations[0])
        
        return test_data


def main():
    """Example usage of the ImageGenerator tool."""
    print("Creating Image Generation Tool...")
    
    # Initialize generator
    generator = ImageGenerator("generated_images")
    
    # Generate different types of images
    print("Generating sample images...")
    
    # Geometric image
    geometric_img = generator.generate_synthetic_image((512, 512), "geometric", num_shapes=8)
    generator.save_image(geometric_img, "geometric_sample.png")
    
    # Noise image
    noise_img = generator.generate_synthetic_image((512, 512), "noise", noise_type="gaussian", intensity=0.6)
    generator.save_image(noise_img, "noise_sample.png")
    
    # Gradient image
    gradient_img = generator.generate_synthetic_image((512, 512), "gradient", gradient_type="radial")
    generator.save_image(gradient_img, "gradient_sample.png")
    
    # Text image
    text_img = generator.generate_synthetic_image((512, 512), "text", text="AI Generated Text", font_size=64)
    generator.save_image(text_img, "text_sample.png")
    
    # Pattern image
    pattern_img = generator.generate_synthetic_image((512, 512), "pattern", pattern_type="checkerboard", tile_size=40)
    generator.save_image(pattern_img, "pattern_sample.png")
    
    # Generate test dataset
    print("Generating test dataset...")
    dataset = generator.generate_test_dataset(num_images=5)
    saved_paths = generator.save_dataset(dataset, "test")
    
    # Generate GAICo test data
    print("Generating GAICo test data...")
    gaico_data = generator.generate_for_gaico_testing(num_sets=2)
    
    # Save GAICo test images
    for i, (ref, gen, sim, diff) in enumerate(zip(
        gaico_data["reference_images"],
        gaico_data["generated_images"], 
        gaico_data["similar_images"],
        gaico_data["different_images"]
    )):
        generator.save_image(ref, f"gaico_ref_{i+1}.png")
        generator.save_image(gen, f"gaico_gen_{i+1}.png")
        generator.save_image(sim, f"gaico_sim_{i+1}.png")
        generator.save_image(diff, f"gaico_diff_{i+1}.png")
    
    print(f"Generated images saved to: {generator.output_dir}")
    print("Sample images created:")
    for file in generator.output_dir.glob("*.png"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main() 