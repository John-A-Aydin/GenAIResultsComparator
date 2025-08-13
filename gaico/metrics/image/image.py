from abc import ABC
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from ..base import BaseMetric
from PIL import Image
from scipy.ndimage import gaussian_filter

class ImageMetric(BaseMetric, ABC):
    """
    Abstract base class for metrics that operate on image data.
    Input can be various image representations (e.g., np.array, image path).
    """

    pass

class ImageSSIM(ImageMetric):
    """
    Structural Similarity Index (ImageSSIM) for perceptual image similarity.

    ImageSSIM compares two images by analyzing local patterns of pixel intensities 
    that have been normalized for luminance and contrast. Instead of treating 
    each pixel independently (like MSE or ImagePSNR), ImageSSIM captures structural 
    distortions that are more aligned with how humans perceive differences 
    between images.

    The score is averaged across RGB channels and normalized to [0, 1],
    where 1.0 indicates perfect structural similarity.
    """

    def __init__(self, resize: bool = True, **kwargs: Any):
        """Initialize ImageSSIM similarity metric with optional resizing."""
        super().__init__(**kwargs)
        self.resize = resize

    def _compute_ssim_grayscale(self, img1: np.ndarray, img2: np.ndarray) -> float:
        # Constants for stability in division (as per ImageSSIM paper)
        K1, K2 = 0.01, 0.03
        L = 255.0  # Pixel value dynamic range
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        # Compute local means using Gaussian blur
        mu1 = gaussian_filter(img1, sigma=1.5, mode='reflect')
        mu2 = gaussian_filter(img2, sigma=1.5, mode='reflect')

        # Compute variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(img1 * img1, sigma=1.5, mode='reflect') - mu1_sq
        sigma2_sq = gaussian_filter(img2 * img2, sigma=1.5, mode='reflect') - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma=1.5, mode='reflect') - mu1_mu2

        # ImageSSIM formula numerator and denominator
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        # Avoid division by zero without degrading score quality
        denominator = np.where(denominator == 0, 1e-8, denominator)
        ssim_map = numerator / denominator

        return np.mean(ssim_map)

    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        Compute ImageSSIM similarity between a pair of images.
        Handles preprocessing, resizing, color conversion, and averaging across channels.
        """
        # Convert images to NumPy arrays if given as PIL
        if isinstance(generated_item, Image.Image):
            generated_item = np.array(generated_item.convert("RGB"))
        if isinstance(reference_item, Image.Image):
            reference_item = np.array(reference_item.convert("RGB"))

        # Normalize float images to 0-255 uint8
        if generated_item.dtype != np.uint8:
            generated_item = (generated_item * 255.0).round().clip(0, 255).astype(np.uint8)
        if reference_item.dtype != np.uint8:
            reference_item = (reference_item * 255.0).round().clip(0, 255).astype(np.uint8)

        # Resize if dimensions mismatch
        if generated_item.shape != reference_item.shape:
            if not self.resize:
                raise ValueError("Input images for ImageSSIM must have the same dimensions.")
            ref_img = Image.fromarray(reference_item)
            gen_img = Image.fromarray(generated_item).resize(ref_img.size, Image.Resampling.LANCZOS)
            generated_item = np.array(gen_img)
            reference_item = np.array(ref_img)

        # Compute ImageSSIM per channel and take the mean for RGB
        if generated_item.ndim == 3 and generated_item.shape[2] == 3:
            ssim_val = np.mean([
                self._compute_ssim_grayscale(generated_item[..., c], reference_item[..., c])
                for c in range(3)
            ])
        else:
            ssim_val = self._compute_ssim_grayscale(generated_item, reference_item)

        # Clamp output to [0, 1]
        return float(np.clip(ssim_val, 0.0, 1.0))

    def _batch_calculate(self, generated_items: Iterable, reference_items: Iterable, **kwargs: Any) -> List[float]:
        return [self._single_calculate(gen, ref, **kwargs) for gen, ref in zip(generated_items, reference_items)]


class ImagePSNR(ImageMetric):
    """
    Peak Signal-to-Noise Ratio (ImagePSNR) for pixel-level image similarity.

    ImagePSNR is a simple metric based on the Mean Squared Error (MSE) between two images.
    It measures the ratio between the maximum possible pixel value and the magnitude 
    of noise (error). While easy to compute and interpret, ImagePSNR does not account 
    for perceptual factors and may overestimate similarity in some cases.

    Output is in decibels (dB), where higher values indicate better quality.
    """
    def __init__(self, resize: bool = True, **kwargs: Any):
        """Initialize ImagePSNR similarity metric with optional resizing."""
        super().__init__(**kwargs)
        self.resize = resize

    def _single_calculate(self, generated_item: Any, reference_item: Any, **kwargs: Any) -> float:
        """
        Compute ImagePSNR between a pair of images.
        Handles resizing, RGB conversion, and normalization.
        """
        # Convert to NumPy arrays if inputs are PIL Images
        if isinstance(generated_item, Image.Image):
            generated_item = np.array(generated_item.convert("RGB"))
        if isinstance(reference_item, Image.Image):
            reference_item = np.array(reference_item.convert("RGB"))

        # Normalize float inputs to 0-255 uint8
        if generated_item.dtype != np.uint8:
            generated_item = (np.clip(generated_item, 0, 1) * 255).astype(np.uint8)
        if reference_item.dtype != np.uint8:
            reference_item = (np.clip(reference_item, 0, 1) * 255).astype(np.uint8)

        # Resize if images differ in shape
        if generated_item.shape != reference_item.shape:
            if not self.resize:
                raise ValueError("Input images for ImagePSNR must have the same dimensions.")
            ref_img = Image.fromarray(reference_item)
            gen_img = Image.fromarray(generated_item).resize(ref_img.size, Image.Resampling.LANCZOS)
            generated_item = np.array(gen_img)
            reference_item = np.array(ref_img)

        # Compute Mean Squared Error (MSE)
        mse = np.mean((generated_item.astype(np.float32) - reference_item.astype(np.float32)) ** 2)

        # ImagePSNR formula: 10 * log10(MAX^2 / MSE)
        if mse == 0:
            return float("inf")  # Perfect match
        return float(10 * np.log10((255.0 ** 2) / mse))

    def _batch_calculate(self, generated_items: Iterable, reference_items: Iterable, **kwargs: Any) -> List[float]:
        return [self._single_calculate(gen, ref, **kwargs) for gen, ref in zip(generated_items, reference_items)]

class ImageAverageHash(ImageMetric):
    """
    Normalized average hash (aHash) similarity for images.
    This method compares 8x8 grayscale downsampled images and computes
    Hamming similarity of the resulting binary hash. The score is normalized to [0, 1].
    """

    def __init__(self, **kwargs: Any):
        """Initialize the aHash-based image similarity metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        Calculate normalized aHash similarity for a single pair of images.

        :param generated_item: The generated image (e.g., np.array, path).
        :type generated_item: Any
        :param reference_item: The reference image.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (not used here).
        :return: Normalized aHash similarity score between 0 and 1.
        :rtype: float | dict
        """
        # Convert to PIL image if input is a NumPy array.
        if isinstance(generated_item, np.ndarray):
            generated_item = Image.fromarray(generated_item)
        if isinstance(reference_item, np.ndarray):
            reference_item = Image.fromarray(reference_item)

        # Resize to 8x8 and convert to grayscale.
        gen_resized = generated_item.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        ref_resized = reference_item.convert("L").resize((8, 8), Image.Resampling.LANCZOS)

        # Compute average pixel values.
        gen_array = np.array(gen_resized, dtype=np.float32)
        ref_array = np.array(ref_resized, dtype=np.float32)
        gen_mean = gen_array.mean()
        ref_mean = ref_array.mean()

        # Compute binary hash: 1 if pixel > mean, else 0.
        gen_hash = (gen_array > gen_mean).astype(np.uint8).flatten()
        ref_hash = (ref_array > ref_mean).astype(np.uint8).flatten()

        # Hamming similarity: proportion of matching bits.
        similarity = 1.0 - np.sum(gen_hash != ref_hash) / len(gen_hash)
        return float(similarity)

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        Calculate normalized aHash similarity for a batch of image pairs.

        :param generated_items: Iterable of generated images.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference images.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments (not used here).
        :return: List of normalized aHash similarity scores.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        results = []
        for gen, ref in zip(generated_items, reference_items):
            results.append(self._single_calculate(gen, ref, **kwargs))
        return results


class ImageHistogramMatch(ImageMetric):
    """
    Color histogram-based similarity metric for images.
    Computes normalized histogram intersection between RGB histograms of two images.
    The output is a similarity score in the range [0, 1], where 1 means the histograms are identical.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the histogram-based similarity metric."""
        super().__init__(**kwargs)

    def _single_calculate(
        self, generated_item: Any, reference_item: Any, **kwargs: Any
    ) -> float | dict:
        """
        Calculate histogram intersection similarity for a single pair of images.

        :param generated_item: The generated image (e.g., np.array, path).
        :type generated_item: Any
        :param reference_item: The reference image.
        :type reference_item: Any
        :param kwargs: Additional keyword arguments (e.g., number of histogram bins).
        :return: Normalized histogram intersection score in the range [0, 1].
        :rtype: float | dict
        """
        # Convert to PIL image if input is a NumPy array.
        if isinstance(generated_item, np.ndarray):
            generated_item = Image.fromarray(generated_item)
        if isinstance(reference_item, np.ndarray):
            reference_item = Image.fromarray(reference_item)

        # Get number of bins for histogram, default to 256.
        bins = kwargs.get("bins", 256)

        # Convert both images to RGB and extract arrays.
        gen_arr = np.array(generated_item.convert("RGB"))
        ref_arr = np.array(reference_item.convert("RGB"))

        # Compute histogram intersection across R, G, B channels.
        intersection = 0.0
        total = 0.0
        for ch in range(3):  # Iterate over R, G, B.
            gen_hist = np.histogram(gen_arr[:, :, ch], bins=bins, range=(0, 255))[0]
            ref_hist = np.histogram(ref_arr[:, :, ch], bins=bins, range=(0, 255))[0]

            # Sum minimum of each bin across histograms.
            intersection += np.sum(np.minimum(gen_hist, ref_hist))
            total += np.sum(gen_hist)

        # Normalize similarity score to [0, 1].
        similarity = intersection / total if total > 0 else 0.0
        return similarity

    def _batch_calculate(
        self,
        generated_items: Iterable | np.ndarray | pd.Series,
        reference_items: Iterable | np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> List[float] | List[dict] | np.ndarray | pd.Series:
        """
        Calculate histogram intersection similarity for a batch of image pairs.

        :param generated_items: Iterable of generated images.
        :type generated_items: Iterable | np.ndarray | pd.Series
        :param reference_items: Iterable of reference images.
        :type reference_items: Iterable | np.ndarray | pd.Series
        :param kwargs: Additional keyword arguments (e.g., number of histogram bins).
        :return: List of normalized histogram intersection scores for all pairs.
        :rtype: List[float] | List[dict] | np.ndarray | pd.Series
        """
        results = []
        for gen, ref in zip(generated_items, reference_items):
            results.append(self._single_calculate(gen, ref, **kwargs))
        return results



