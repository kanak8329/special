"""
Helper Utilities
================
Common helper functions used across the project.
"""

import os
import numpy as np


# Blood group labels
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Blood group colors for visualization
BLOOD_GROUP_COLORS = {
    'A+': '#FF6B6B',
    'A-': '#FF8E8E',
    'B+': '#4ECDC4',
    'B-': '#7EDDD6',
    'AB+': '#45B7D1',
    'AB-': '#72C9DE',
    'O+': '#96CEB4',
    'O-': '#B4DEC9',
}

# Blood group descriptions
BLOOD_GROUP_INFO = {
    'A+': 'Type A Positive — Can donate to A+ and AB+. Can receive from A+, A-, O+, O-.',
    'A-': 'Type A Negative — Can donate to A+, A-, AB+, AB-. Can receive from A- and O-.',
    'B+': 'Type B Positive — Can donate to B+ and AB+. Can receive from B+, B-, O+, O-.',
    'B-': 'Type B Negative — Can donate to B+, B-, AB+, AB-. Can receive from B- and O-.',
    'AB+': 'Type AB Positive — Universal Recipient. Can receive from all blood types.',
    'AB-': 'Type AB Negative — Can donate to AB+ and AB-. Can receive from A-, B-, AB-, O-.',
    'O+': 'Type O Positive — Can donate to A+, B+, AB+, O+. Can receive from O+ and O-.',
    'O-': 'Type O Negative — Universal Donor. Can donate to all blood types.',
}


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_dir():
    """Get the saved model directory path."""
    return os.path.join(get_project_root(), 'model', 'saved_model')


def get_data_dir():
    """Get the data directory path."""
    return os.path.join(get_project_root(), 'data')


def get_sample_dir():
    """Get the sample fingerprints directory path."""
    return os.path.join(get_data_dir(), 'sample_fingerprints')


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def format_confidence(confidence_scores, blood_groups=BLOOD_GROUPS):
    """
    Format confidence scores for display.
    
    Args:
        confidence_scores: numpy array of probabilities
        blood_groups: list of blood group labels
        
    Returns:
        list of dicts with 'group' and 'confidence' keys, sorted by confidence
    """
    results = []
    for group, conf in zip(blood_groups, confidence_scores):
        results.append({
            'group': group,
            'confidence': float(conf),
            'percentage': f"{conf * 100:.1f}%"
        })
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results
