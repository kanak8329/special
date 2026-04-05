"""
Prediction Module — Dual Model Support
=======================================
Auto-detects whether a CNN (EfficientNet-B0) or Random Forest model
is available and uses the appropriate inference pipeline.

Priority: CNN model (.pth) >> Random Forest model (.pkl)

Usage:
    from model.predict import predict_blood_group
    result = predict_blood_group(image_input)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import preprocess_fingerprint
from utils.helpers import BLOOD_GROUPS, get_model_dir, format_confidence

# ── Model availability detection ──────────────────────────────────────

def _cnn_model_exists() -> bool:
    """Check if CNN model file is present."""
    return os.path.exists(os.path.join(get_model_dir(), 'cnn_model.pth'))


def _rf_model_exists() -> bool:
    """Check if Random Forest model files are present."""
    model_dir = get_model_dir()
    return (os.path.exists(os.path.join(model_dir, 'blood_group_model.pkl')) and
            os.path.exists(os.path.join(model_dir, 'scaler.pkl')))


def get_active_model_type() -> str:
    """Returns 'cnn', 'rf', or 'none'."""
    if _cnn_model_exists():
        return 'cnn'
    if _rf_model_exists():
        return 'rf'
    return 'none'


# ── Public API ────────────────────────────────────────────────────────

def load_model():
    """
    Load whichever model is available (CNN preferred).
    Returns a tuple indicating the model type.

    Returns:
        ('cnn', model, class_names)  or  ('rf', model, scaler)
    """
    if _cnn_model_exists():
        from model.cnn_model import load_cnn_model
        model, class_names, _ = load_cnn_model()
        return ('cnn', model, class_names)
    elif _rf_model_exists():
        import joblib
        model_dir = get_model_dir()
        model  = joblib.load(os.path.join(model_dir, 'blood_group_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        return ('rf', model, scaler)
    else:
        raise FileNotFoundError(
            "No trained model found. Please run:\n"
            "  python model/train.py --mode cnn --dataset-path data/sample_fingerprints"
        )


def predict_blood_group(image_input) -> dict:
    """
    Predict blood group from a fingerprint image.
    Auto-selects CNN or RF model based on what's available.

    Args:
        image_input: str (file path), PIL.Image, or numpy array

    Returns:
        dict with keys:
            predicted_group  : str   — e.g. 'A+'
            confidence       : float — top class probability (0–1)
            all_scores       : list  — confidence for all 8 classes, sorted desc
            model_type       : str   — 'cnn' or 'rf'
            feature_breakdown: dict  — feature info (for display)
            preprocessing_steps: dict — intermediate images
    """
    # Preprocess image (works the same for both models)
    preprocessing_results = preprocess_fingerprint(image_input)
    processed_image = preprocessing_results['processed']

    model_type = get_active_model_type()

    if model_type == 'cnn':
        return _predict_cnn(processed_image, preprocessing_results)
    elif model_type == 'rf':
        return _predict_rf(processed_image, preprocessing_results)
    else:
        raise FileNotFoundError(
            "No trained model found. Run: python model/train.py --mode cnn"
        )


# ── CNN Inference ─────────────────────────────────────────────────────

def _predict_cnn(processed_image: np.ndarray, preprocessing_results: dict) -> dict:
    """Run inference using the CNN model."""
    import torch
    from PIL import Image as PILImage
    from model.cnn_model import load_cnn_model, get_inference_transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_names, metrics = load_cnn_model(device=device)

    # Convert numpy array → PIL Image → transform → tensor
    transform = get_inference_transform()
    pil_img   = PILImage.fromarray(processed_image)
    tensor    = transform(pil_img).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    with torch.no_grad():
        if torch.cuda.is_available():
            from torch.amp import autocast
            with autocast('cuda'):
                logits = model(tensor)
        else:
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # class_names is the sorted alphabetical order from ImageFolder
    # Map back to standard BLOOD_GROUPS ordering for consistent display
    # Build a probs array in BLOOD_GROUPS order
    reordered_probs = np.zeros(len(BLOOD_GROUPS))
    for idx, cls_name in enumerate(class_names):
        # Normalize cls_name (handle A_pos → A+ etc.)
        normalized = cls_name.upper().replace('_POS', '+').replace('_NEG', '-') \
                                      .replace('POS', '+').replace('NEG', '-')
        if normalized in BLOOD_GROUPS:
            bg_idx = BLOOD_GROUPS.index(normalized)
            reordered_probs[bg_idx] = probs[idx]

    pred_idx        = int(np.argmax(reordered_probs))
    predicted_group = BLOOD_GROUPS[pred_idx]
    confidence      = float(reordered_probs[pred_idx])
    all_scores      = format_confidence(reordered_probs, BLOOD_GROUPS)

    return {
        'predicted_group'    : predicted_group,
        'confidence'         : confidence,
        'all_scores'         : all_scores,
        'model_type'         : 'cnn',
        'features'           : None,
        'feature_breakdown'  : {
            'model_type'  : 'cnn',
            'architecture': 'EfficientNet-B0',
            'input_size'  : '224×224 px',
            'parameters'  : '5.3M',
            'pretrained'  : 'ImageNet (ILSVRC2012)',
            'total_features': 0,   # CNN learns features internally
        },
        'preprocessing_steps': preprocessing_results,
    }


# ── RF Inference ──────────────────────────────────────────────────────

def _predict_rf(processed_image: np.ndarray, preprocessing_results: dict) -> dict:
    """Run inference using the Random Forest model."""
    import joblib
    from model.feature_extraction import extract_all_features

    model_dir = get_model_dir()
    model  = joblib.load(os.path.join(model_dir, 'blood_group_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    features, feature_breakdown = extract_all_features(processed_image)
    features_scaled = scaler.transform(features.reshape(1, -1))

    prediction    = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    predicted_group = BLOOD_GROUPS[prediction]
    confidence      = float(probabilities[prediction])
    all_scores      = format_confidence(probabilities, BLOOD_GROUPS)

    feature_breakdown['model_type']   = 'rf'
    feature_breakdown['architecture'] = 'Random Forest (300 trees)'

    return {
        'predicted_group'    : predicted_group,
        'confidence'         : confidence,
        'all_scores'         : all_scores,
        'model_type'         : 'rf',
        'features'           : features,
        'feature_breakdown'  : feature_breakdown,
        'preprocessing_steps': preprocessing_results,
    }


# ── Legacy API (backward compat) ─────────────────────────────────────

def predict_from_features(features: np.ndarray) -> dict:
    """RF-only: Predict from pre-extracted features."""
    import joblib
    model_dir = get_model_dir()
    model  = joblib.load(os.path.join(model_dir, 'blood_group_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction      = model.predict(features_scaled)[0]
    probabilities   = model.predict_proba(features_scaled)[0]

    return {
        'predicted_group': BLOOD_GROUPS[prediction],
        'confidence'     : float(probabilities[prediction]),
        'all_scores'     : format_confidence(probabilities, BLOOD_GROUPS),
    }
