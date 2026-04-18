"""
Prediction Module — Dual Model Support
=======================================
Auto-detects whether a CNN (EfficientNet-B3 / B0) or Random Forest
model is available and uses the appropriate inference pipeline.

Fix #1: Transform input size is now read from the saved checkpoint
         (300x300 for B3, 224x224 for B0) instead of being hardcoded.

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

def _get_available_cnn_variants() -> list:
    """Check which CNN model variants are present."""
    model_dir = get_model_dir()
    variants = []
    for variant in ['efficientnet_b3', 'efficientnet_b0']:
        if os.path.exists(os.path.join(model_dir, f'cnn_model_{variant}.pth')):
            variants.append(variant)
    return variants

def _cnn_model_exists() -> bool:
    """Check if any CNN model file is present."""
    return len(_get_available_cnn_variants()) > 0

def _rf_model_exists() -> bool:
    """Check if Random Forest model files are present."""
    model_dir = get_model_dir()
    return (os.path.exists(os.path.join(model_dir, 'blood_group_model.pkl')) and
            os.path.exists(os.path.join(model_dir, 'scaler.pkl')))


def get_active_model_type() -> str:
    """Returns 'cnn_multi', 'cnn', 'rf', or 'none'."""
    variants = _get_available_cnn_variants()
    if len(variants) > 1:
        return 'cnn_multi'
    elif len(variants) == 1:
        return 'cnn'
    if _rf_model_exists():
        return 'rf'
    return 'none'


# ── Public API ────────────────────────────────────────────────────────

def load_model():
    """
    Load whichever model is available (CNN preferred).
    Returns a tuple indicating the model type.
    """
    variants = _get_available_cnn_variants()
    if variants:
        from model.cnn_model import load_cnn_model
        # Load the first best available
        model, class_names, _ = load_cnn_model(model_variant=variants[0])
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
            "  python model/train.py --mode cnn --model efficientnet_b3 --dataset-path data/sample_fingerprints"
        )


def predict_blood_group(image_input, selected_variants=None) -> dict:
    """
    Predict blood group from a fingerprint image.
    If multiple CNN models exist, returns results for all of them.

    Args:
        image_input: str (file path), PIL.Image, or numpy array
        selected_variants: optional list of strings specifying which variants to run

    Returns:
        dict with keys mapping 'variant_name' -> result dict
        Even if it's RF, it returns {'rf': result_dict}
    """
    preprocessing_results = preprocess_fingerprint(image_input)
    processed_image = preprocessing_results['processed']

    model_type = get_active_model_type()
    results = {}

    if model_type in ['cnn', 'cnn_multi']:
        variants = _get_available_cnn_variants()
        if selected_variants is not None:
            variants = [v for v in variants if v in selected_variants]
        for variant in variants:
            results[variant] = _predict_cnn(processed_image, preprocessing_results, variant)
    elif model_type == 'rf':
        results['rf'] = _predict_rf(processed_image, preprocessing_results)
    else:
        raise FileNotFoundError(
            "No trained model found. Run: python model/train.py --mode cnn"
        )
    return results


# ── CNN Inference ─────────────────────────────────────────────────────

_cached_cnn_models = {}

def _predict_cnn(processed_image: np.ndarray, preprocessing_results: dict, model_variant: str = 'efficientnet_b3') -> dict:
    """Run inference using the CNN model."""
    import torch
    from PIL import Image as PILImage
    from model.cnn_model import load_cnn_model, get_inference_transform, EFFICIENTNET_VARIANTS

    global _cached_cnn_models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_variant not in _cached_cnn_models:
        _cached_cnn_models[model_variant] = load_cnn_model(device=device, model_variant=model_variant)
        
    model, class_names, metrics = _cached_cnn_models[model_variant]

    _, _, input_size, in_features = EFFICIENTNET_VARIANTS.get(
        model_variant, EFFICIENTNET_VARIANTS['efficientnet_b0']
    )
    param_millions = model.count_parameters() / 1e6

    # Convert numpy array -> PIL Image -> correct-size transform -> tensor
    transform = get_inference_transform(model_variant=model_variant)
    pil_img   = PILImage.fromarray(processed_image)
    tensor    = transform(pil_img).unsqueeze(0).to(device)  # [1,3,input_size,input_size]

    with torch.no_grad():
        if torch.cuda.is_available():
            from torch.amp import autocast
            with autocast('cuda'):
                logits = model(tensor)
        else:
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Map ImageFolder alphabetical order -> BLOOD_GROUPS standard order
    reordered_probs = np.zeros(len(BLOOD_GROUPS))
    for idx, cls_name in enumerate(class_names):
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
            'model_type'    : 'cnn',
            'architecture'  : model_variant.replace('efficientnet_', 'EfficientNet-').upper().replace('_B', '-B'),
            'input_size'    : f'{input_size}x{input_size} px',
            'parameters'    : f'{param_millions:.1f}M',
            'pretrained'    : 'ImageNet',
            'total_features': 0,
        },
        'preprocessing_steps': preprocessing_results,
    }


# ── RF Inference ──────────────────────────────────────────────────────

_cached_rf_model = None

def _predict_rf(processed_image: np.ndarray, preprocessing_results: dict) -> dict:
    """Run inference using the Random Forest model."""
    import joblib
    from model.feature_extraction import extract_all_features

    global _cached_rf_model
    if _cached_rf_model is None:
        model_dir = get_model_dir()
        model  = joblib.load(os.path.join(model_dir, 'blood_group_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        _cached_rf_model = (model, scaler)
    else:
        model, scaler = _cached_rf_model

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
