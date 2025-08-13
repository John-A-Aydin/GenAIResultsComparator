# Image Generation Tools for GAICo

This directory contains tools for generating images to test and evaluate GAICo's image similarity metrics.

## Overview

The image generation tools provide two main capabilities:

1. **Synthetic Image Generation** (`image_generation_tool.py`) - Creates programmatic test images
2. **AI Image Generation** (`ai_image_generator.py`) - Uses AI models to generate realistic images

## Quick Start

### 1. Install Dependencies

```bash
# Install additional requirements for image generation
pip install -r image_generation_requirements.txt

# For AI generation (optional)
pip install torch diffusers transformers
```

### 2. Basic Usage

```python
from image_generation_tool import ImageGenerator
from gaico.metrics.image import ImageSSIM, ImageAverageHash

# Create synthetic images
generator = ImageGenerator("output_dir")
geometric_img = generator.generate_synthetic_image(
    shape=(256, 256), 
    image_type="geometric", 
    num_shapes=5
)

# Test with GAICo metrics
ssim_metric = ImageSSIM()
score = ssim_metric._single_calculate(geometric_img, reference_img)
print(f"SSIM Score: {score:.4f}")
```

### 3. Run the Example Notebook

```bash
jupyter notebook image_generation_example.ipynb
```

## Synthetic Image Generation

The `ImageGenerator` class creates various types of synthetic images:

### Image Types

- **Geometric**: Random shapes (circles, rectangles, triangles)
- **Noise**: Random noise patterns (Gaussian, uniform, salt & pepper)
- **Gradient**: Color gradients (linear, radial)
- **Text**: Text-based images with customizable fonts
- **Pattern**: Repeating patterns (checkerboard, stripes)

### Example Usage

```python
from image_generation_tool import ImageGenerator

generator = ImageGenerator("test_images")

# Generate different types
geometric = generator.generate_synthetic_image((512, 512), "geometric", num_shapes=8)
noise = generator.generate_synthetic_image((512, 512), "noise", noise_type="gaussian")
gradient = generator.generate_synthetic_image((512, 512), "gradient", gradient_type="radial")
text = generator.generate_synthetic_image((512, 512), "text", text="Hello World")
pattern = generator.generate_synthetic_image((512, 512), "pattern", pattern_type="checkerboard")

# Create variations for testing
variations = generator.create_comparison_images(geometric, num_variations=5)

# Generate test dataset
dataset = generator.generate_test_dataset(num_images=10)
```

## AI Image Generation

The `AIImageGenerator` class integrates with various AI image generation APIs:

### Supported Backends

- **Stability AI** (Stable Diffusion XL)
- **OpenAI** (DALL-E)
- **Hugging Face** (Inference API)
- **Local Stable Diffusion** (requires local installation)

### Setup

#### Environment Variables

```bash
# Stability AI
export STABILITY_API_KEY="your-stability-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Hugging Face
export HF_API_KEY="your-huggingface-api-key"

# Local Stable Diffusion (optional)
export SD_MODEL_PATH="/path/to/stable-diffusion-model"
```

#### Example Usage

```python
from ai_image_generator import setup_generators_from_env

# Setup from environment variables
generator = setup_generators_from_env()

# Generate images
images = generator.generate_images(
    prompt="A beautiful landscape with mountains",
    num_images=3,
    width=512,
    height=512
)

# Generate test dataset
dataset = generator.generate_test_dataset([
    "A cat playing with a ball",
    "A futuristic city skyline",
    "An abstract painting"
], images_per_prompt=2)
```

## Integration with GAICo

### Using Generated Images with GAICo Metrics

```python
from gaico.metrics.image import ImageSSIM, ImageAverageHash, ImageHistogramMatch
from image_generation_tool import ImageGenerator

# Generate test images
generator = ImageGenerator()
reference = generator.generate_synthetic_image((256, 256), "geometric")
variations = generator.create_comparison_images(reference, 3)

# Test with GAICo metrics
ssim = ImageSSIM()
hash_metric = ImageAverageHash()
histogram = ImageHistogramMatch()

for i, variation in enumerate(variations):
    ssim_score = ssim._single_calculate(variation, reference)
    hash_score = hash_metric._single_calculate(variation, reference)
    histogram_score = histogram._single_calculate(variation, reference)
    
    print(f"Variation {i+1}:")
    print(f"  SSIM: {ssim_score:.4f}")
    print(f"  Hash: {hash_score:.4f}")
    print(f"  Histogram: {histogram_score:.4f}")
```

### Creating GAICo Test Data

```python
# Generate data specifically for GAICo testing
gaico_data = generator.generate_for_gaico_testing(num_sets=5)

# This creates:
# - reference_images: Base images
# - generated_images: AI-generated variations
# - similar_images: Similar but different images
# - different_images: Completely different images
```

## Advanced Features

### Batch Processing

```python
# Generate large datasets
dataset = generator.generate_test_dataset(
    num_images=100,
    base_shape=(256, 256),
    variations=["geometric", "noise", "gradient"]
)

# Save to disk
saved_paths = generator.save_dataset(dataset, "large_test")
```

### Custom Image Effects

```python
# Create custom variations
base_img = generator.generate_synthetic_image((256, 256), "geometric")
variations = generator.create_comparison_images(base_img, 5)

# Variations include:
# - Original image
# - Noise added
# - Brightness adjustment
# - Contrast adjustment
# - Blur effect
```

### AI Generation with Custom Parameters

```python
# Custom AI generation parameters
images = generator.generate_images(
    prompt="A futuristic robot in a cyberpunk city",
    generator_name="stability",
    num_images=4,
    width=1024,
    height=1024,
    guidance_scale=8.0,
    num_inference_steps=50,
    seed=42  # For reproducible results
)
```

## File Structure

```
examples/
├── image_generation_tool.py          # Synthetic image generation
├── ai_image_generator.py             # AI-powered image generation
├── image_generation_example.ipynb    # Jupyter notebook example
├── image_generation_requirements.txt # Additional dependencies
├── README_image_generation.md        # This file
└── generated_images/                 # Output directory (created automatically)
    ├── synthetic_images/             # Synthetic image outputs
    └── ai_generated_images/          # AI-generated image outputs
```

## Configuration

### Output Directories

Both tools create output directories automatically:

- `ImageGenerator`: Creates `generated_images/` by default
- `AIImageGenerator`: Creates `ai_generated_images/` by default

### Image Formats

- All images are saved as PNG files
- Internal processing uses NumPy arrays
- Compatible with PIL/Pillow for additional processing

### Performance Tips

1. **Batch Processing**: Generate multiple images at once for better efficiency
2. **Caching**: Save generated images to avoid regeneration
3. **Parallel Processing**: Use multiple generators for large datasets
4. **Memory Management**: Process images in batches for large datasets

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r image_generation_requirements.txt
   ```

2. **API Key Issues**: Verify environment variables are set correctly
   ```bash
   echo $STABILITY_API_KEY
   echo $OPENAI_API_KEY
   ```

3. **Memory Issues**: Reduce image size or batch size for large datasets

4. **Local Model Issues**: Ensure PyTorch and diffusers are installed for local generation

### Getting Help

- Check the example notebook for detailed usage
- Review the docstrings in the source code
- Ensure all dependencies are properly installed
- Verify API keys and network connectivity for AI generation

## Contributing

To extend the image generation tools:

1. **Add New Image Types**: Extend the `ImageGenerator` class with new `_generate_*_image` methods
2. **Add New AI Backends**: Create new classes inheriting from `BaseAIGenerator`
3. **Add New Metrics**: Integrate additional image similarity metrics
4. **Improve Performance**: Optimize image generation and processing algorithms

## License

These tools are part of the GAICo project and follow the same license terms. 