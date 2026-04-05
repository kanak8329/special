# Blood Group Detection Using Fingerprint — System Architecture

## Overview

This document describes the technical architecture of the Blood Group Detection system
that uses fingerprint image analysis with machine learning to predict blood groups.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT WEB APPLICATION                         │
│                              (app.py)                                      │
├─────────────┬──────────────────┬───────────────────┬───────────────────────┤
│  Sidebar    │  Detection Page  │ Architecture Page │    About Page         │
│  - Upload   │  - Results       │ - Pipeline        │    - Overview         │
│  - Samples  │  - Preprocessing │ - Modules         │    - Features         │
│             │  - Features      │ - Tech Stack      │    - References       │
│             │  - Charts        │                   │                       │
└──────┬──────┴────────┬─────────┴───────────────────┴───────────────────────┘
       │               │
       ▼               ▼
┌──────────────┐ ┌──────────────────────────────────────────────────────────┐
│  Image Input │ │               PREDICTION PIPELINE                       │
│  (JPEG/PNG)  │ │                (model/predict.py)                       │
└──────┬───────┘ │  1. Load Model  →  2. Preprocess  →  3. Extract Features│
       │         │  →  4. Scale  →  5. Predict  →  6. Format Results       │
       │         └─────────┬──────────────┬────────────────────┬───────────┘
       │                   │              │                    │
       ▼                   ▼              ▼                    ▼
┌──────────────────┐ ┌────────────────────────┐ ┌──────────────────────────┐
│  PREPROCESSING   │ │   FEATURE EXTRACTION   │ │   CLASSIFICATION MODEL   │
│  (utils/         │ │   (model/              │ │   (model/saved_model/)   │
│   preprocessing  │ │    feature_extraction  │ │                          │
│   .py)           │ │    .py)                │ │   • blood_group_model    │
│                  │ │                        │ │     .pkl                 │
│  • Grayscale     │ │  Orientation (24):     │ │   • scaler.pkl           │
│  • Resize 256²   │ │  • Sobel gradients     │ │                          │
│  • Gaussian blur │ │  • Ridge angles        │ │   Logistic Regression    │
│  • CLAHE         │ │  • Orientation hist    │ │   (multinomial, L-BFGS)  │
│  • Adaptive      │ │  • Statistical moments │ │   8 classes              │
│    thresholding  │ │                        │ │                          │
│                  │ │  Wavelet (32):         │ │                          │
│                  │ │  • Haar DWT (level 2)  │ │                          │
│                  │ │  • DB4 DWT (level 1)   │ │                          │
│                  │ │  • Sub-band stats      │ │                          │
│                  │ │  (mean, std, energy,   │ │                          │
│                  │ │   entropy)             │ │                          │
└──────────────────┘ └────────────────────────┘ └──────────────────────────┘
```

---

## Data Flow

```
Input Image (JPEG/PNG)
    │
    ├──→ Load Image (cv2.imread / PIL)
    │
    ├──→ Convert to Grayscale (BGR → Gray)
    │
    ├──→ Resize to 256×256 (cv2.resize)
    │
    ├──→ Denoise (Gaussian Blur, kernel=5)
    │
    ├──→ Enhance (CLAHE, clipLimit=2.0, tileGrid=8×8)
    │
    ├──→ Threshold (Adaptive Gaussian, blockSize=11)
    │
    ├──→ Feature Extraction
    │       ├── Orientation Features (24)
    │       │     ├── Sobel X, Y gradients
    │       │     ├── Block-wise orientation (16×16 grid)
    │       │     ├── Orientation histogram (16 bins)
    │       │     ├── Mean, Std, Skewness, Kurtosis
    │       │     ├── Entropy
    │       │     └── Coherence (mean, std, median)
    │       │
    │       └── Wavelet Features (32)
    │             ├── Haar DWT Level 2 (28 features)
    │             │     └── cA, cH, cV, cD sub-bands × 4 stats
    │             └── DB4 DWT Level 1 (4 features)
    │                   └── cA sub-band × 4 stats
    │
    ├──→ Feature Scaling (StandardScaler)
    │
    ├──→ Logistic Regression Prediction
    │
    └──→ Output: Blood Group + Confidence Scores
              (A+, A-, B+, B-, AB+, AB-, O+, O-)
```

---

## Technology Stack

| Layer          | Technology         | Purpose                           |
|----------------|--------------------|-----------------------------------|
| Web Framework  | Streamlit          | Interactive web UI                |
| ML Framework   | Scikit-learn       | Model training & prediction       |
| Image Processing | OpenCV           | Fingerprint preprocessing         |
| Signal Processing | PyWavelets      | Wavelet feature extraction        |
| Visualization  | Matplotlib, Seaborn| Charts and plots                  |
| Serialization  | Joblib             | Model persistence (.pkl)          |
| Language       | Python 3.8+        | Core implementation               |

---

## File Structure

```
collage_project_special/
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── model/
│   ├── __init__.py
│   ├── feature_extraction.py      # Orientation + Wavelet features
│   ├── train.py                   # Dataset generation + training
│   ├── predict.py                 # Prediction pipeline
│   └── saved_model/               # Trained model files
│       ├── blood_group_model.pkl  # Logistic Regression model
│       ├── scaler.pkl             # StandardScaler
│       └── metrics.pkl            # Training metrics
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py           # Image preprocessing pipeline
│   └── helpers.py                 # Constants & utility functions
├── data/
│   └── sample_fingerprints/       # Sample images for demo
│       ├── A_pos/
│       ├── A_neg/
│       ├── B_pos/ ...
└── docs/
    └── architecture.md            # This file
```

---

## Blood Groups Classified

| Blood Group | Description                          |
|-------------|--------------------------------------|
| A+          | Type A Positive                      |
| A-          | Type A Negative                      |
| B+          | Type B Positive                      |
| B-          | Type B Negative                      |
| AB+         | Type AB Positive (Universal Recipient)|
| AB-         | Type AB Negative                     |
| O+          | Type O Positive                      |
| O-          | Type O Negative (Universal Donor)    |
