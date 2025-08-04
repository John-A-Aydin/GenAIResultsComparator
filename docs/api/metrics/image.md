
# Image Metrics

This module provides a suite of image similarity metrics designed to compare generated and reference images across perceptual, structural, and low-level feature dimensions. Metrics support single and batch mode comparisons with input formats including NumPy arrays and PIL Images.

---

## SSIM (Structural Similarity Index)

**Path:** `gaico.metrics.image.SSIM`

Structural Similarity Index measures perceived quality based on structural information, luminance, and contrast. SSIM returns a similarity score between 0 and 1, where 1 indicates identical images.

### Features
- Per-channel SSIM for RGB images
- Normalized to [0, 1]
- Optional auto-resize for mismatched shapes
- Handles both grayscale and color images

### Input Formats
- `np.ndarray` (2D grayscale or 3D RGB)
- `PIL.Image`
- Float arrays are rescaled internally to 8-bit

### Example
```python
from gaico.metrics.image import SSIM
from PIL import Image
import numpy as np

metric = SSIM()
img1 = np.array(Image.open("ref.png"))
img2 = np.array(Image.open("gen.png"))

score = metric.calculate(img2, img1)
print(f"SSIM Score: {score:.3f}")

```

## PSNR (Peak Signal-to-Noise Ratio)

**Path:** `gaico.metrics.image.PSNR`

Peak Signal-to-Noise Ratio is a pixel-wise metric that quantifies the fidelity of a generated image with respect to a reference image. It is computed based on the Mean Squared Error (MSE) between the two images. A higher PSNR indicates better similarity.

### Features
- Supports grayscale and RGB images
- Auto-resizes images if shapes don't match (configurable)
- Normalized for 8-bit image data
- Returns float (in decibels)

### Input Formats
- `np.ndarray` (float or uint8)
- `PIL.Image.Image`
- Float arrays are scaled internally to `uint8`

### Internals
1. Converts both images to RGB (if applicable)
2. Rescales float inputs to 8-bit uint8
3. Computes MSE = mean((img1 - img2)^2)
4. PSNR = 10 * log10((MAX^2) / MSE), where MAX = 255

### Edge Case
- If MSE = 0 (perfect match), returns `inf` (infinite PSNR)

### Example
```python
from gaico.metrics.image import PSNR
from PIL import Image
import numpy as np

metric = PSNR()
img1 = np.array(Image.open("ref.png"))
img2 = np.array(Image.open("gen.png"))

score = metric.calculate(img2, img1)
print(f"PSNR Score: {score:.2f} dB")
```

## AverageHash (aHash)

**Path:** `gaico.metrics.image.AverageHash`

The AverageHash metric compares two images using perceptual hashing. It converts each image to an 8×8 grayscale image, computes the mean pixel value, and generates a binary hash based on whether each pixel is above or below the mean. The similarity score is the Hamming similarity (1 - normalized Hamming distance) between the two hashes.

### Features
- Perceptual image similarity metric
- Invariant to minor changes (blur, brightness)
- Efficient, lightweight, no dependencies beyond PIL and NumPy
- Score in [0.0, 1.0], where 1.0 means identical hashes

### Input Formats
- `np.ndarray` (RGB or grayscale)
- `PIL.Image.Image`

### ⚙️ Internals
1. Convert both inputs to grayscale and resize to 8×8
2. Compute mean pixel value for each
3. Generate binary hash: 1 if pixel > mean, else 0
4. Compute Hamming similarity: 1 - (# differing bits / 64)

### Limitations
- Not sensitive to high-frequency content (e.g., texture)
- Cannot detect small geometric misalignments

### Example
```python
from gaico.metrics.image import AverageHash
from PIL import Image

metric = AverageHash()
img1 = Image.open("ref.jpg")
img2 = Image.open("gen.jpg")

score = metric.calculate(img2, img1)
print(f"aHash Similarity Score: {score:.3f}")
```

## HistogramMatch

**Path:** `gaico.metrics.image.HistogramMatch`

The `HistogramMatch` metric evaluates similarity between two images based on their color histograms. It computes the intersection between histograms across the RGB channels, normalized to the range [0.0, 1.0].

### Features
- Histogram-based perceptual similarity
- Captures color distribution differences
- Channel-wise histogram intersection
- Normalized score ∈ [0.0, 1.0]

### Input Formats
- `np.ndarray` (RGB)
- `PIL.Image.Image`

### Internals
1. Convert inputs to RGB
2. Compute histogram for each channel (default 256 bins)
3. Use histogram intersection:  
   `similarity = sum(min(ref_hist[i], gen_hist[i])) / sum(gen_hist)`
4. Average across channels

### Parameters
- `bins`: Number of bins per channel (default: 256)

### Limitations
- Sensitive to lighting changes and minor shifts
- Ignores spatial layout of pixels

### Example
```python
from gaico.metrics.image import HistogramMatch
from PIL import Image

metric = HistogramMatch()
img1 = Image.open("ref.jpg")
img2 = Image.open("gen.jpg")

score = metric.calculate(img2, img1)
print(f"HistogramMatch Score: {score:.3f}")
```
