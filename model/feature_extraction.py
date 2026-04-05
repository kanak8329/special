"""
Feature Extraction Module
=========================
Extracts features from preprocessed fingerprint images:
1. Orientation Features - Ridge orientation analysis using gradients
2. Wavelet Features - 2D DWT using Haar and Daubechies wavelets

Combined feature vector: ~56 features per fingerprint image.
"""

import cv2
import numpy as np
import pywt
from scipy import stats


def extract_orientation_features(image, block_size=16):
    """
    Extract orientation features from fingerprint ridges.
    
    Uses Sobel gradients to compute local ridge orientations,
    then extracts statistical features from the orientation field.
    
    Args:
        image: numpy array (grayscale, preprocessed)
        block_size: int - size of local blocks for orientation computation
        
    Returns:
        numpy array of orientation features
    """
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    h, w = image.shape
    orientations = []
    coherences = []
    
    # Compute orientation in local blocks
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_gx = grad_x[i:i+block_size, j:j+block_size]
            block_gy = grad_y[i:i+block_size, j:j+block_size]
            
            # Compute orientation angle using gradient moments
            Gxx = np.sum(block_gx ** 2)
            Gyy = np.sum(block_gy ** 2)
            Gxy = np.sum(block_gx * block_gy)
            
            # Ridge orientation (perpendicular to gradient)
            angle = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
            orientations.append(angle)
            
            # Coherence (reliability of orientation estimate)
            denom = Gxx + Gyy
            if denom > 0:
                coherence = np.sqrt((Gxx - Gyy)**2 + 4*Gxy**2) / denom
            else:
                coherence = 0.0
            coherences.append(coherence)
    
    orientations = np.array(orientations)
    coherences = np.array(coherences)
    
    # Extract statistical features from orientation field
    features = []
    
    # Orientation histogram (16 bins from -pi/2 to pi/2)
    hist, _ = np.histogram(orientations, bins=16, range=(-np.pi/2, np.pi/2))
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist = hist / hist.sum()  # Normalize
    features.extend(hist.tolist())
    
    # Statistical features of orientations
    features.append(np.mean(orientations))
    features.append(np.std(orientations))
    features.append(float(stats.skew(orientations)) if len(orientations) > 2 else 0.0)
    features.append(float(stats.kurtosis(orientations)) if len(orientations) > 3 else 0.0)
    
    # Orientation entropy
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero)) if len(hist_nonzero) > 0 else 0.0
    features.append(entropy)
    
    # Coherence statistics
    features.append(np.mean(coherences))
    features.append(np.std(coherences))
    features.append(np.median(coherences))
    
    return np.array(features)


def extract_wavelet_features(image, wavelet='haar', level=2):
    """
    Extract wavelet features using 2D Discrete Wavelet Transform.
    
    Decomposes the image into approximation and detail sub-bands,
    then extracts statistical features from each sub-band.
    
    Args:
        image: numpy array (grayscale, preprocessed)
        wavelet: str - wavelet type ('haar', 'db4', etc.)
        level: int - decomposition level
        
    Returns:
        numpy array of wavelet features
    """
    # Normalize image to float
    img_float = image.astype(np.float64) / 255.0
    
    # Perform 2D DWT
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)
    
    features = []
    
    # Extract features from each decomposition level
    for i, coeff in enumerate(coeffs):
        if i == 0:
            # Approximation coefficients (cA)
            subbands = [coeff]
            names = ['cA']
        else:
            # Detail coefficients (cH, cV, cD)
            subbands = list(coeff)
            names = ['cH', 'cV', 'cD']
        
        for subband in subbands:
            # Statistical features for each sub-band
            features.append(np.mean(subband))       # Mean
            features.append(np.std(subband))         # Standard deviation
            features.append(np.sum(subband ** 2))    # Energy
            
            # Entropy
            flat = np.abs(subband).flatten()
            flat_norm = flat / (flat.sum() + 1e-10)
            nonzero = flat_norm[flat_norm > 0]
            entropy = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) > 0 else 0.0
            features.append(entropy)
    
    return np.array(features)


def extract_all_features(image):
    """
    Extract complete feature vector from a preprocessed fingerprint image.
    Combines orientation and wavelet features.
    
    Args:
        image: numpy array (grayscale, preprocessed, 256x256)
        
    Returns:
        numpy array - combined feature vector (~56 features)
        dict - feature breakdown for visualization
    """
    # Extract orientation features (24 features)
    orientation_feats = extract_orientation_features(image)
    
    # Extract wavelet features - Haar (28 features)
    wavelet_feats_haar = extract_wavelet_features(image, wavelet='haar', level=2)
    
    # Extract additional wavelet features - Daubechies (4 features from level 1)
    wavelet_feats_db = extract_wavelet_features(image, wavelet='db4', level=1)
    
    # Combine all features
    combined = np.concatenate([
        orientation_feats,
        wavelet_feats_haar,
        wavelet_feats_db
    ])
    
    # Replace any NaN or Inf values
    combined = np.nan_to_num(combined, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Feature breakdown for visualization
    breakdown = {
        'orientation': {
            'values': orientation_feats,
            'count': len(orientation_feats),
            'names': (
                [f'orient_hist_{i}' for i in range(16)] +
                ['orient_mean', 'orient_std', 'orient_skew', 'orient_kurtosis',
                 'orient_entropy', 'coherence_mean', 'coherence_std', 'coherence_median']
            )
        },
        'wavelet_haar': {
            'values': wavelet_feats_haar,
            'count': len(wavelet_feats_haar),
            'names': [f'haar_{i}' for i in range(len(wavelet_feats_haar))]
        },
        'wavelet_db4': {
            'values': wavelet_feats_db,
            'count': len(wavelet_feats_db),
            'names': [f'db4_{i}' for i in range(len(wavelet_feats_db))]
        },
        'total_features': len(combined)
    }
    
    return combined, breakdown
