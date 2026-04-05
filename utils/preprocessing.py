"""
Fingerprint Image Preprocessing Module
======================================
Handles all image preprocessing steps:
- Grayscale conversion
- Resizing to standard dimensions
- Denoising with Gaussian blur
- Contrast enhancement (CLAHE)
- Binary thresholding for ridge extraction
"""

import cv2
import numpy as np
from PIL import Image


# Standard image size for processing
IMG_SIZE = (256, 256)


def load_image(image_input):
    """
    Load image from file path or PIL Image object.
    
    Args:
        image_input: str (file path) or PIL.Image or numpy array
        
    Returns:
        numpy array (BGR format)
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        # Convert PIL Image to OpenCV format
        img_array = np.array(image_input)
        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image: numpy array (BGR or grayscale)
        
    Returns:
        numpy array (grayscale)
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def resize_image(image, size=IMG_SIZE):
    """
    Resize image to standard dimensions.
    
    Args:
        image: numpy array
        size: tuple (width, height)
        
    Returns:
        numpy array (resized)
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def denoise_image(image, kernel_size=5):
    """
    Apply Gaussian blur for denoising.
    
    Args:
        image: numpy array (grayscale)
        kernel_size: int (must be odd)
        
    Returns:
        numpy array (denoised)
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def enhance_contrast(image, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for contrast enhancement.
    
    Args:
        image: numpy array (grayscale)
        clip_limit: float - threshold for contrast limiting
        tile_size: int - size of grid for histogram equalization
        
    Returns:
        numpy array (enhanced)
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    return clahe.apply(image)


def apply_threshold(image, block_size=11, C=2):
    """
    Apply adaptive binary thresholding for ridge extraction.
    
    Args:
        image: numpy array (grayscale)
        block_size: int - size of pixel neighborhood
        C: int - constant subtracted from mean
        
    Returns:
        numpy array (binary)
    """
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )


def preprocess_fingerprint(image_input):
    """
    Full preprocessing pipeline for fingerprint images.
    
    Args:
        image_input: str (file path), PIL.Image, or numpy array
        
    Returns:
        dict with keys:
            - 'original': original image
            - 'grayscale': grayscale version
            - 'resized': resized to standard size
            - 'denoised': after Gaussian blur
            - 'enhanced': after CLAHE enhancement
            - 'thresholded': after binary thresholding
            - 'processed': final processed image (enhanced version for feature extraction)
    """
    # Load image
    original = load_image(image_input)
    
    # Convert to grayscale
    grayscale = convert_to_grayscale(original)
    
    # Resize to standard size
    resized = resize_image(grayscale)
    
    # Denoise
    denoised = denoise_image(resized)
    
    # Enhance contrast
    enhanced = enhance_contrast(denoised)
    
    # Binary threshold
    thresholded = apply_threshold(enhanced)
    
    return {
        'original': original,
        'grayscale': grayscale,
        'resized': resized,
        'denoised': denoised,
        'enhanced': enhanced,
        'thresholded': thresholded,
        'processed': enhanced  # Use enhanced as main input for feature extraction
    }
