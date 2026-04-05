# Project Changes & PyCharm Execution Guide

This document outlines all the upgrades made to the Blood Group Detection project and provides the exact terminal commands for you to run in PyCharm.

## 🛠️ Summary of Changes

### 1. Model Optimization (`model/train.py`)
- **Real Dataset Support**: Added the ability to load a real dataset folder structure instead of relying purely on synthetic data.
- **Multiprocessing**: Re-wrote the feature extraction and data generation pipeline to use multiprocessing (`multiprocessing.Pool`), utilizing all CPU cores to drastically reduce extraction time for large datasets.
- **Optimized Training**: Updated Logistic Regression parameters for better convergence on large datasets (`max_iter=3000`, `n_jobs=-1`).
- **New CLI Arguments**: You can now pass `--dataset-path` and `--samples` directly from the terminal.

### 2. UI Enhancements & Features (`app.py`)
- **Premium Aesthetics**: Replaced the default styling with a modern "Glassmorphism" design using dark sleek colors, hover animations, and premium gradients.
- **Processing Metrics**: Added an interactive timer that evaluates exactly how long the machine learning model takes to process the image and extract the 56 features.
- **Result Export (New Feature)**: Added a "Download Analysis Report" button allowing users to export the prediction, confidence scores, and processing times as a JSON file.
- **Font Integration**: Imported the modern 'Outfit' font directly using CSS.

---

## 🏃‍♂️ How to Run in PyCharm

Follow these instructions exactly in your PyCharm terminal.

### Step 1: Download the Larger Dataset
There are several fingerprint datasets for blood group detection on Kaggle. Because Kaggle requires you to accept terms and log in, it's best to download it manually:
1. Go to this URL: [Fingerprint Dataset for Blood Group Classification](https://www.kaggle.com/datasets/praveengovi/blood-group-detection-using-fingerprint) (or search for "blood group fingerprint dataset kaggle").
2. Download the ZIP file.
3. Extract the contents inside `c:\anti_gravity_\collage_project_special\data\real_dataset`.
   *(Ensure inside `real_dataset`, there are folders like `A+`, `B-`, `O+`, etc. containing the images)*

### Step 2: Install/Verify Dependencies
Make sure you have all the required libraries for the newly added features (like multiprocessing support). Run:
```bash
pip install streamlit numpy opencv-python scikit-learn matplotlib seaborn Pillow joblib
```

### Step 3: Train the Model
You now have two ways to train the model. Stop any running Streamlit server, then run **one** of the following in your PyCharm terminal:

**Option A - Train using the real dataset (Recommended):**
```bash
python model/train.py --dataset-path data/real_dataset
```

**Option B - Train using large synthetic data (If you don't want to download Kaggle data right now):**
```bash
python model/train.py --samples 500
```
*(This will use multiprocessing to rapidly generate 4,000 images, extract features, and train the model).*

### Step 4: Run the Application
Finally, start your beautifully redesigned application:
```bash
streamlit run app.py
```
Click the local URL (usually `http://localhost:8501`) shown in the terminal.
