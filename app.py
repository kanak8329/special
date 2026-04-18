# -*- coding: utf-8 -*-
"""
Blood Group Detection — Premium Health UI
==========================================
Run: streamlit run app.py
"""

from utils.helpers import BLOOD_GROUPS, BLOOD_GROUP_COLORS, BLOOD_GROUP_INFO, get_sample_dir, get_model_dir
from model.predict import predict_blood_group, load_model, get_active_model_type, _get_available_cnn_variants
import streamlit as st
import numpy as np
import cv2
import os
import sys
import time
import json
import textwrap
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="HemaType AI | Blood Group Detection",
                   page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root { --app-bg: #fbfbfd; --card-bg: #ffffff; --text-h: #1d1d1f; --text-p: #86868b; --accent: #0071e3; --accent-hover: #005cbf; --red: #ff3b30; --success: #34c759; --border: rgba(0,0,0,0.06); }

* { box-sizing: border-box; } .stApp { background: var(--app-bg); font-family: 'Inter', -apple-system, sans-serif; color: var(--text-h); }
/* Remove background orbs for cleaner Apple-like minimal feel */ .orb1, .orb2 { display: none; }

@keyframes fadeUp { from{opacity:0;transform:translateY(15px)} to{opacity:1;transform:translateY(0)} } @keyframes pulseHeart { 0%,100%{transform:scale(1)} 15%{transform:scale(1.1)} 30%{transform:scale(1)} 45%{transform:scale(1.05)} }

/* ── Premium Cards ── */ .glass { background: var(--card-bg); border: 1px solid var(--border); border-radius: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.03); transition: all 0.3s cubic-bezier(.25,.8,.25,1); } .glass:hover { box-shadow: 0 12px 32px rgba(0,0,0,0.06); transform: translateY(-2px); }

/* ── Hero Section ── */ .hero { background: #ffffff; border: 1px solid var(--border); border-radius: 24px; padding: 3rem 2.5rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.02); animation: fadeUp .6s ease-out; } .hero h1 { font-family:'Inter', sans-serif; font-size:3rem; font-weight:800; color: var(--text-h); margin:0; letter-spacing:-1.2px; } .hero p { color:var(--text-p); font-size:1.15rem; font-weight:500; margin:.8rem 0 0; } .blood-drop { font-size:3.5rem; animation: pulseHeart 2s ease-in-out infinite; display:inline-block; }

/* ── Metric Cards ── */ .metric-card { background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 1.4rem; text-align: center; transition: all .3s ease; box-shadow: 0 2px 12px rgba(0,0,0,0.02); } .metric-card:hover { transform: translateY(-3px) scale(1.02); border-color: rgba(0,113,227,.2); box-shadow: 0 8px 24px rgba(0,113,227,.06); } .metric-val { font-family:'Inter', sans-serif; font-size:1.8rem; font-weight:700; color:var(--accent); } .metric-lbl { color:var(--text-p); font-size:.78rem; text-transform:uppercase; font-weight:600; letter-spacing:1px; margin-top:.4rem; }

/* ── Clinical Paper Report ── */ .clinical-report { background: #ffffff; border: 1px solid #e5e5ea; border-left: 6px solid #0071e3; border-radius: 12px; padding: 2.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.04); font-family: 'Inter', sans-serif; color: #1d1d1f; margin: 1.5rem 0; animation: fadeUp .8s ease; } .cr-header { border-bottom: 1px solid #f2f2f7; padding-bottom: 1rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: flex-end; } .cr-header h2 { margin: 0; color: #1d1d1f; font-size: 1.4rem; font-weight: 700; letter-spacing: -0.5px; } .cr-header p { margin: 0; color: #86868b; font-size: 0.85rem; } .cr-header .date { font-family: monospace; color: #86868b; font-size:0.8rem; } .cr-row { display: flex; justify-content: space-between; padding: 0.8rem 0; font-size: 0.95rem; border-bottom: 1px solid #f9f9f9;} .cr-row:last-child { border-bottom: none; } .cr-label { color: #86868b; font-weight: 500; font-size: 0.85rem; } .cr-value { font-weight: 600; color: #1d1d1f; } .cr-highlight { font-size: 1.5rem; font-weight: 800; color: var(--red); } .cr-footer { border-top: 1px solid #f2f2f7; margin-top: 2rem; padding-top: 1rem; text-align: center; color: #aeaeb2; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Upload zone ── */ div[data-testid="stFileUploader"] { border: 2px dashed #d1d1d6; border-radius:16px; padding:1.5rem; background: #fafafa; transition:all .2s; } div[data-testid="stFileUploader"]:hover { border-color:var(--accent); background: #f0f8ff; }

/* ── Sidebar ── */ [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5e5ea; } [data-testid="stSidebar"] .stRadio label { font-weight: 500; color: #1d1d1f !important; } [data-testid="stSidebar"] .stRadio label:hover { color:var(--accent) !important; }

/* ── Buttons ── */ .stButton button, .stDownloadButton button { background: var(--accent) !important; color:#fff !important; border:none !important; border-radius:12px !important; font-weight:600 !important; box-shadow: 0 4px 10px rgba(0,113,227,0.2) !important; transition:all .2s ease !important; } .stButton button:hover, .stDownloadButton button:hover { transform:scale(1.03) !important; background: var(--accent-hover) !important; box-shadow:0 6px 14px rgba(0,113,227,0.3) !important; }

/* ── Miscellaneous ── */ .conf-bar-track { background:#f2f2f7; border-radius:99px; height:8px; overflow:hidden; } .conf-bar-fill { height:100%; border-radius:99px; background:var(--accent); transition:width 1s ease; } /* ── Compatibility Matrix ── */ .compat-matrix { display: grid; grid-template-columns: repeat(9, 1fr); gap: 6px; padding: 2rem; background: #fff; border-radius: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.03); border: 1px solid var(--border); overflow-x: auto; font-family:'Inter',sans-serif;} .cm-header { display: flex; align-items: center; justify-content: center; font-size: 0.9rem; font-weight: 700; color: #86868b; } .cm-row-header { display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1rem; color: #1d1d1f; background: #f2f2f7; border-radius: 12px; padding: 0.8rem 0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02); } .cm-cell { display: flex; align-items: center; justify-content: center; padding: 0.8rem 0; border-radius: 12px; font-weight: 800; font-size: 1.3rem; transition: all 0.2s cubic-bezier(.25,.8,.25,1); border: 1px solid transparent; } .cm-cell:hover { transform: scale(1.15); box-shadow: 0 8px 16px rgba(0,0,0,0.1); border-color: rgba(0,0,0,0.1); z-index: 10; cursor: default; } .cm-can { background: rgba(52, 199, 89, 0.15); color: #34c759; } .cm-cannot { background: #fafafa; color: #d1d1d6; font-weight: 400; font-size: 1rem; }
/* ── Pills ── */ .pill-green { display:flex; align-items:center; gap:.8rem; padding:1.2rem; margin:.6rem 0; background: #ffffff; border: 1px solid rgba(52,199,89,0.3); border-left: 6px solid #34c759; border-radius: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); transition: 0.2s; } .pill-green:hover { transform: translateX(5px); box-shadow: 0 6px 16px rgba(52,199,89,0.1); } .pill-blue { display:flex; align-items:center; gap:.8rem; padding:1.2rem; margin:.6rem 0; background: #ffffff; border: 1px solid rgba(0,113,227,0.3); border-left: 6px solid #0071e3; border-radius: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); transition: 0.2s; } .pill-blue:hover { transform: translateX(5px); box-shadow: 0 6px 16px rgba(0,113,227,0.1); }

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  BLOOD COMPATIBILITY DATA
# ══════════════════════════════════════════════════════════════
DONATE_TO = {
    'A+': ['A+','AB+'], 'A-': ['A+','A-','AB+','AB-'],
    'B+': ['B+','AB+'], 'B-': ['B+','B-','AB+','AB-'],
    'AB+': ['AB+'],     'AB-': ['AB+','AB-'],
    'O+': ['A+','B+','AB+','O+'], 'O-': BLOOD_GROUPS,
}
RECEIVE_FROM = {
    'A+': ['A+','A-','O+','O-'], 'A-': ['A-','O-'],
    'B+': ['B+','B-','O+','O-'], 'B-': ['B-','O-'],
    'AB+': BLOOD_GROUPS,         'AB-': ['A-','B-','AB-','O-'],
    'O+': ['O+','O-'],           'O-': ['O-'],
}
BG_FACTS = {
    'A+': '30% of people have A+ blood — the second most common type.',
    'A-': 'A- can donate to 4 blood types including A+ and AB+.',
    'B+': 'B+ is found in about 9% of the population worldwide.',
    'B-': 'B- is rare — only ~2% of people have this blood type.',
    'AB+': 'AB+ is the Universal Recipient — can receive all blood types!',
    'AB-': 'AB- is very rare — less than 1% of people worldwide.',
    'O+': 'O+ is the most common blood type — 38% of people.',
    'O-': 'O- is the Universal Donor — can donate to all blood types!',
}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0'>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:2.5rem;animation:heartbeat 1.8s ease-in-out infinite;display:inline-block'>🩸</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:.5rem 0 .2rem;font-family:Space Grotesk,sans-serif;font-size:1.3rem'>HemaType AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7f8fa4;font-size:.78rem;margin:0'>Blood Group Detection</p></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    page = st.radio("Navigate", ["🔬 Detection", "🔄 Compatibility",
                    "📊 Architecture", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    active_model = get_active_model_type()
    badge_cls = "badge-cnn" if active_model == 'cnn' else "badge-rf"
    badge_txt = "🧠 EfficientNet-B3" if active_model == 'cnn' else "🌲 Random Forest"
    st.markdown(
        f"<p style='color:#86868b;font-size:.75rem;text-align:center;'>Active Model</p><div style='text-align:center'><span style='background:#0071e3;color:white;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600;'>{badge_txt}</span></div>", unsafe_allow_html=True)

    if page == "🔬 Detection":
        st.markdown("<p style='font-size:.85rem;color:grey;padding-top:1rem;'>Use the main page controls to upload your fingerprint images or select clinically pre-processed samples.</p>", unsafe_allow_html=True)

    st.markdown("<p style='color:#90a4ae;font-size:.75rem;text-align:center;margin-top:2.5rem;font-weight:500'>HemaType Clinical • 2026</p>", unsafe_allow_html=True)

# Inject Orbs
st.markdown("<div class='orb1'></div><div class='orb2'></div>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: DETECTION
# ══════════════════════════════════════════════════════════════
if page == "🔬 Detection":
    model_type_label = "Deep Learning (EfficientNet-B3 CNN)" if active_model == 'cnn' else "Machine Learning (Random Forest)"
    st.markdown(f'''
    <div class="hero">
      <div class="blood-drop">🩸</div>
      <h1>HemaType AI</h1>
      <p>Non-invasive blood group analysis via {model_type_label}</p>
    </div>
    ''', unsafe_allow_html=True)

    model_dir = get_model_dir()
    cnn_exists = os.path.exists(os.path.join(model_dir, 'cnn_model.pth'))
    rf_exists = os.path.exists(os.path.join(
        model_dir, 'blood_group_model.pkl'))

    if not cnn_exists and not rf_exists:
        st.error("⚠️ **No trained model found!** Please run training first:\n```\npython model/train.py --mode cnn --dataset-path data/sample_fingerprints\n```")
        st.stop()

    st.markdown("### 📥 Load Patient Fingerprint")
    up_col, smp_col = st.columns([1.5, 1])

    with up_col:
        uploaded_file = st.file_uploader("Upload patient fingerprint:", type=[
                                         'png','jpg','jpeg','bmp','tiff'])

    use_sample = None
    with smp_col:
        st.markdown(
            "<div style='margin-bottom:.5rem'>Or select clinical sample:</div>", unsafe_allow_html=True)
        sample_dir = get_sample_dir()
        if os.path.exists(sample_dir):
            groups = sorted(os.listdir(sample_dir))
            if groups:
                sel = st.selectbox("Select Blood Group",
                                   groups, label_visibility="collapsed")
                gp = os.path.join(sample_dir, sel)
                samples = [f for f in os.listdir(gp) if f.lower().endswith(
                    ('.png','.bmp','.jpg'))] if os.path.exists(gp) else []
                if samples and st.button("🔍 Use Sample Data", use_container_width=True):
                    use_sample = os.path.join(gp, samples[0])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    image_to_process = None
    if 'uploaded_file' in dir() and uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif use_sample:
        image_to_process = use_sample
        
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Diagnostic Model Selection")
    
    available_variants = _get_available_cnn_variants()
    if not available_variants:
        available_variants = ['efficientnet_b3'] # fallback
        
    dropdown_options = ["Run Both (Side-by-Side Comparison)"] + [f"EfficientNet-{v.split('_')[-1].upper()}" for v in available_variants]
    selected_mode = st.selectbox("Select Model Architecture", dropdown_options)
    
    variants_to_run = available_variants
    if selected_mode != "Run Both (Side-by-Side Comparison)":
        selected_b_type = selected_mode.split('-')[-1].lower() # e.g. b3
        variants_to_run = [v for v in available_variants if v.endswith(selected_b_type)]

    if image_to_process is not None:
        with st.spinner(""):
            st.markdown(
                "<div style='text-align:center;padding:2rem;'>"
                "<div style='font-size:2rem;animation:spin 1.2s linear infinite;display:inline-block'>⚙️</div>"
                "<p style='color:#7f8fa4;margin:.5rem 0 0'>Analyzing fingerprint with AI...</p>"
                "</div>",
                unsafe_allow_html=True
            )
            t0 = time.time()
            try:
                results_dict = predict_blood_group(image_to_process, selected_variants=variants_to_run)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
            total_elapsed = time.time() - t0


        if not results_dict:
            st.error("No predictions returned from the model.")
            st.stop()
            
        cols = st.columns(len(results_dict))
        
        for idx, (variant_name, result) in enumerate(results_dict.items()):
            with cols[idx]:
                predicted = result['predicted_group']
                confidence = result['confidence']
                all_scores = result['all_scores']
                breakdown = result['feature_breakdown']
                
                # Clinical Report
                import datetime
                report_id = f"HT-{np.random.randint(10000, 99999)}"
                current_time = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                
                html_parts = []
                html_parts.append("<div class='clinical-report'>")
                html_parts.append("  <div class='cr-header'>")
                html_parts.append("    <div>")
                html_parts.append(f"      <h2>{variant_name.replace('efficientnet_', 'EfficientNet-').upper()} Report</h2>")
                html_parts.append("      <p>HemaType AI Diagnostic System v4.0</p>")
                html_parts.append("    </div>")
                html_parts.append(f"    <div class='date'>{current_time}</div>")
                html_parts.append("  </div>")
                html_parts.append("  <div class='cr-body'>")
                html_parts.append("    <div class='cr-row'>")
                html_parts.append("      <span class='cr-label'>Report ID</span>")
                html_parts.append(f"      <span class='cr-value'>{report_id}</span>")
                html_parts.append("    </div>")
                html_parts.append("    <div class='cr-row' style='margin-top:1.5rem; border-bottom:none; align-items:center;'>")
                html_parts.append("      <span class='cr-label'>Detected Blood Group</span>")
                html_parts.append(f"      <span class='cr-highlight'>{predicted}</span>")
                html_parts.append("    </div>")
                html_parts.append("    <div class='cr-row' style='border-bottom:none;'>")
                html_parts.append("      <span class='cr-label'>Analysis Confidence</span>")
                html_parts.append(f"      <span class='cr-value' style='color:#0071e3;font-size:1.1rem;'>✓ {confidence*100:.1f}%</span>")
                html_parts.append("    </div>")
                html_parts.append("  </div>")
                html_parts.append("</div>")
                st.markdown("".join(html_parts), unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(f"### 📊 Confidence ({variant_name.upper()})")
                bars_html = ""
                for s in all_scores:
                    pct = s['confidence'] * 100
                    is_top = s['group'] == predicted
                    fill_color = "linear-gradient(90deg,#e63946,#ff6b6b)" if is_top else "linear-gradient(90deg,#4361ee,#06d6a0)"
                    weight = "700" if is_top else "400"
                    color_group_name = '#1d1d1f' if is_top else '#86868b'
                    color_pct = '#0071e3' if is_top else '#86868b'
                    
                    b_parts = []
                    b_parts.append("<div class='conf-bar-wrap'>")
                    b_parts.append("  <div class='conf-bar-label'>")
                    b_parts.append(f"    <span style='font-weight:{weight};color:{color_group_name}'>{s['group']}</span>")
                    b_parts.append(f"    <span style='color:{color_pct}'>{pct:.1f}%</span>")
                    b_parts.append("  </div>")
                    b_parts.append("  <div class='conf-bar-track'>")
                    b_parts.append(f"    <div class='conf-bar-fill' style='width:{min(pct,100):.1f}%;background:{fill_color}'></div>")
                    b_parts.append("  </div>")
                    b_parts.append("</div>")
                    bars_html += "".join(b_parts)
                st.markdown(bars_html, unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                # Export
                report = {
                    "variant": variant_name,
                    "predicted_blood_group": predicted,
                    "confidence_score": float(confidence),
                    "processing_time_seconds": round(total_elapsed, 3),
                    "model_architecture": breakdown.get('architecture','Unknown'),
                    "can_donate_to": DONATE_TO.get(predicted,[]),
                    "can_receive_from": RECEIVE_FROM.get(predicted,[]),
                    "blood_group_fact": BG_FACTS.get(predicted,''),
                    "all_scores": {s['group']: round(s['confidence']*100,2) for s in all_scores},
                }
                st.download_button("📄 JSON Report", json.dumps(
                    report, indent=4), f"hematype_{variant_name}_report.json", "application/json", use_container_width=True, key=f"dl_{variant_name}")


    else:
        # ── Upload Prompt ─────────────────────────────────────────────
        # ── Upload Prompt ─────────────────────────────────────────────
        up_parts = [
            "<div style='display:flex;align-items:center;justify-content:center;min-height:200px'>",
            "  <div class='glass' style='text-align:center;padding:2.5rem;max-width:520px;margin:1rem auto;animation:fadeUp .6s ease'>",
            "    <div style='font-size:3rem;margin-bottom:.5rem;color:#0071e3;'>📄</div>",
            "    <h3 style='color:#1d1d1f;font-family:Inter,sans-serif;margin:.5rem 0'>Awaiting Scan</h3>",
            "    <p style='color:#86868b;font-size:.95rem;line-height:1.4'>Use the upload box above to provide a patient fingerprint image.</p>",
            "  </div>",
            "</div>"
        ]
        st.markdown("\n".join(up_parts), unsafe_allow_html=True)

        st.markdown("### 🔬 How It Works")
        steps_info = [
            ("📤", "Upload", "Upload a fingerprint image in BMP, PNG or JPG format"),
            ("🔄", "Preprocess", "Image resized to 300×300, Gaussian blur, CLAHE enhanced"),
            ("🧠", "CNN Inference", "EfficientNet-B3 runs a forward pass through MBConv stages"),
            ("🩸", "Predict", "Softmax layer outputs confidence for all 8 blood groups"),
        ]
        cols = st.columns(4)
        for i, (col, (icon, title, desc)) in enumerate(zip(cols, steps_info)):
            with col:
                c_parts = [
                    f"<div class='glass' style='padding:1.8rem;text-align:center;min-height:200px;animation:fadeUp {0.2+i*.1}s ease'>",
                    f"  <div style='font-size:2.2rem;margin-bottom:.8rem'>{icon}</div>",
                    f"  <h4 style='color:#1d1d1f;font-family:Inter,sans-serif;margin:.3rem 0;font-weight:600'>{title}</h4>",
                    f"  <p style='color:#86868b;font-size:.85rem;line-height:1.4'>{desc}</p>",
                    "</div>"
                ]
                st.markdown("\n".join(c_parts), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: COMPATIBILITY CHECKER
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Compatibility":
    h_parts = [
        "<div class='hero'>",
        "  <div class='blood-drop'>🔄</div>",
        "  <h1>Blood Type Compatibility</h1>",
        "  <p>Interactive blood group compatibility checker — find who you can donate to or receive from</p>",
        "</div>"
    ]
    st.markdown("\n".join(h_parts), unsafe_allow_html=True)

    sel_bg = st.selectbox("Select a Blood Group to explore:", BLOOD_GROUPS, key="compat_select")

    c1, c2 = st.columns(2)
    with c1:
        donate_html = [
            "<div class='glass' style='padding:2rem;'>",
            "  <h3 style='color:#1d1d1f;font-family:Inter,sans-serif;margin-bottom:1.2rem'>💉 Can Donate To</h3>",
            ''.join([f"<div class='pill-green'><span style='font-size:1.4rem;font-weight:800;color:#34c759;min-width:45px'>{g}</span><span style='color:#86868b;font-size:.85rem;line-height:1.4'>{BLOOD_GROUP_INFO.get(g,'')}</span></div>" for g in DONATE_TO.get(sel_bg,[])]),
            "</div>"
        ]
        st.markdown("\n".join(donate_html), unsafe_allow_html=True)

    with c2:
        receive_html = [
            "<div class='glass' style='padding:2rem;'>",
            "  <h3 style='color:#1d1d1f;font-family:Inter,sans-serif;margin-bottom:1.2rem'>🩸 Can Receive From</h3>",
            ''.join([f"<div class='pill-blue'><span style='font-size:1.4rem;font-weight:800;color:#0071e3;min-width:45px'>{g}</span><span style='color:#86868b;font-size:.85rem;line-height:1.4'>{BLOOD_GROUP_INFO.get(g,'')}</span></div>" for g in RECEIVE_FROM.get(sel_bg,[])]),
            "</div>"
        ]
        st.markdown("\n".join(receive_html), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🗓️ Full Compatibility Matrix")

    header = "<div class='cm-header'></div>" + "".join(f"<div class='cm-header'>{g}</div>" for g in BLOOD_GROUPS)
    rows = ""
    for donor in BLOOD_GROUPS:
        rows += f"<div class='cm-row-header'>{donor}</div>"
        for recipient in BLOOD_GROUPS:
            can = recipient in DONATE_TO.get(donor, [])
            cls_name = "cm-can" if can else "cm-cannot"
            icon = "✓" if can else "✕"
            rows += f"<div class='cm-cell {cls_name}'>{icon}</div>"
    
    st.markdown(f"<div class='compat-matrix'>{header}{rows}</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-box'>💡 <strong>About {sel_bg}:</strong> {BLOOD_GROUP_INFO.get(sel_bg,'')}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Architecture":
    a_parts = [
        "<div class='hero'>",
        "  <div class='blood-drop'>📊</div>",
        "  <h1>System Architecture</h1>",
        "  <p>Technical overview of the HemaType AI pipeline — Dual-Model Architecture</p>",
        "</div>"
    ]
    st.markdown("\n".join(a_parts), unsafe_allow_html=True)

    st.markdown("### 🧠 Core Neural Network Pipeline")
    
    # Beautiful flow pipeline UI
    pipeline_parts = [
"<div style='background:#ffffff; border: 1px solid rgba(0,0,0,0.06); border-radius:24px; padding: 2.5rem; box-shadow: 0 4px 24px rgba(0,0,0,0.02); margin-bottom: 2rem;'>",
"<style>",
".flow-node { padding: 1.2rem; border-radius: 16px; background: #fbfbfd; border: 1px solid #e5e5ea; display: flex; align-items: center; gap: 1.2rem; position: relative; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); }",
".flow-node:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,113,227,0.08); border-color: rgba(0,113,227,0.3); background:#ffffff;}",
".flow-icon { font-size: 2.2rem; background: rgba(0,113,227,0.1); width: 65px; height: 65px; display: flex; align-items: center; justify-content: center; border-radius: 16px; }",
".flow-text h4 { margin: 0 0 0.2rem 0; color: #1d1d1f; font-weight: 700; font-family: \'Inter\', sans-serif; font-size:1.1rem;}",
".flow-text p { margin: 0; color: #86868b; font-size: 0.9rem; line-height: 1.5; }",
".flow-arrow { text-align: center; color: #d1d1d6; font-size: 1.5rem; padding: 0.5rem 0; animation: fadeUp 1s ease infinite alternate; }",
".head-layer { font-family: monospace; font-size: 0.75rem; background: #f2f2f7; padding: 4px 8px; border-radius: 6px; color: #0071e3; margin-top: 6px; display: inline-block; }",
"</style>",
"<div class='flow-node'>",
"    <div class='flow-icon'>📤</div>",
"    <div class='flow-text'>",
"        <h4>Input Phase (High-Resolution Fingerprint)</h4>",
"        <p>Receives localized image streams mapped up to 300x300 pixels internally.</p>",
"    </div>",
"</div>",
"<div class='flow-arrow'>⬇</div>",
"<div class='flow-node'>",
"    <div class='flow-icon'>🔄</div>",
"    <div class='flow-text'>",
"        <h4>Clinical Preprocessing Pipeline</h4>",
"        <p>Translates image to Grayscale, executes CLAHE contrast enhancement, and Normalizes tensor means to ImageNet standard.</p>",
"    </div>",
"</div>",
"<div class='flow-arrow'>⬇</div>",
"<div class='flow-node'>",
"    <div class='flow-icon'>📱</div>",
"    <div class='flow-text'>",
"        <h4>EfficientNet Backbones (B3 & B0)</h4>",
"        <p>Parallel deep learning pathways extracting high-level semantic edge topologies using MBConv blocks. Yields dynamic Global Average Pooled vectors.</p>",
"        <div class='head-layer'>GAP Output: 1536 Channels (B3) / 1280 Channels (B0)</div>",
"    </div>",
"</div>",
"<div class='flow-arrow'>⬇</div>",
"<div class='flow-node'>",
"    <div class='flow-icon'>🧠</div>",
"    <div class='flow-text'>",
"        <h4>Linear Classification Head & Optimizer</h4>",
"        <p>Aggressive layered dropout strategy preventing structural overfitting on biological scans.</p>",
"        <div class='head-layer'>Dropout(0.35) → Linear(1536 to 512) → ReLU → Dropout(0.20) → Linear(256 to 8)</div>",
"    </div>",
"</div>",
"<div class='flow-arrow'>⬇</div>",
"<div class='flow-node' style='border-left: 6px solid #34c759; background:#ffffff;'>",
"    <div class='flow-icon' style='background:rgba(52,199,89,0.15)'>🩸</div>",
"    <div class='flow-text'>",
"        <h4>Output State & Validation</h4>",
"        <p>Softmax optimization mapping directly into the final 8 discrete ABO/Rh blood groups natively.</p>",
"    </div>",
"</div>",
"</div>"
    ]
    st.markdown("".join(pipeline_parts), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### ⚡ Technology Stack & Modules")

    cols = st.columns(3)
    arch_cards = [
        ("🔥 PyTorch 2.5", "Framework", "Bleeding edge CUDA 12.1 operations, fully vectorized tensors, and AdamW optimizing."),
        ("📱 Parametric B3 & B0", "Architecture", "Scaling from ~5.3M (B0) to ~12.2M (B3) parameters depending on accuracy requirements."),
        ("🔄 Fluid Augmentation", "Resilience", "MixUp (α=0.2), RandomPerspective, and intense native ColorJittering against scanner variance."),
        ("⚡ Tensor Cores", "Acceleration", "Native PyTorch AMP mapped explicitly to float16 reducing VRAM and doubling memory bus throughput."),
        ("📈 Cosine Annealing", "Optimization", "Warm restarts for rapidly escaping local loss minima natively through epoch cyclical drops."),
        ("🎛 Adaptive Drop", "Generalization", "Multi-Stage fully connected neural nodes isolating feature detection from direct rote memory."),
    ]
    for i, (col, (title, cat, desc)) in enumerate(zip(cols*2, arch_cards)):
        with col:
            st.markdown(f"""<div class='glass' style='padding:2rem;margin-bottom:1rem;min-height:190px;animation:fadeUp {.2+i*.1}s ease'>
              <p style='display:inline-block;background:rgba(0,113,227,0.1);color:#0071e3;font-size:.7rem;padding: 4px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:1px;margin:0 0 1rem;font-weight:800'>{cat}</p>
              <h4 style='color:#1d1d1f;font-family:Inter,sans-serif;margin:0 0 .5rem;font-size:1.15rem'>{title}</h4>
              <p style='color:#86868b;font-size:.88rem;line-height:1.5;margin:0'>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('''
    <div class="hero" style="background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%);">
      <div class="blood-drop">✨</div>
      <h1 style="color:#0071e3">HemaType AI v4.0</h1>
      <p style="color:#1d1d1f; font-weight:600">The Future of Non-Invasive Blood Diagnostics</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("### 🌟 Project Vision")
    
    about_cols = st.columns([1, 1])
    with about_cols[0]:
        st.markdown('''
        <div class="glass" style="padding: 2.5rem; height: 100%;">
            <h3 style="color:#1d1d1f; margin-top:0; border-bottom: 2px solid #f2f2f7; padding-bottom: 0.8rem; margin-bottom:1.5rem;">The Clinical Problem</h3>
            <p style="color:#86868b; line-height: 1.6; font-size: 1.05rem;">
                Traditional blood typing requires invasive phlebotomy (drawing blood), biochemical reagents, and laboratory processing time. This causes significant friction in emergency triage, remote health screening, and patient comfort.
            </p>
            <p style="color:#86868b; line-height: 1.6; font-size: 1.05rem;">
                <strong>HemaType AI</strong> bypasses the needle entirely by identifying subtle, deep-dermal topographical correlations between fingerprint ridge patterns and ABO/Rh gene expression using extreme-scale Convolutional Neural Networks.
            </p>
        </div>
        ''', unsafe_allow_html=True)

    with about_cols[1]:
        st.markdown('''
        <div class="glass" style="padding: 2.5rem; height: 100%;">
            <h3 style="color:#1d1d1f; margin-top:0; border-bottom: 2px solid #f2f2f7; padding-bottom: 0.8rem; margin-bottom:1.5rem;">Core Features</h3>
            <ul style="color:#86868b; line-height: 1.8; font-size: 1.05rem; padding-left: 1.2rem;">
                <li><strong style="color:#34c759">Zero Biomaterial:</strong> Requires absolutely no chemical reagents.</li>
                <li><strong style="color:#0071e3">Real-Time Inference:</strong> Processes sub-second analysis natively using PyTorch AMP acceleration.</li>
                <li><strong style="color:#1d1d1f">Clinical Precision:</strong> Models leverage millions of synthetic and augmented parameters across dual-backbones (EfficientNet-B3/B0).</li>
                <li><strong style="color:#ff3b30">Privacy First:</strong> Local tensor execution. No cloud patient data transmission required.</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div class='divider' style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("### 🧑‍💻 Technical Foundations")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-val'>PyTorch</div><div class='metric-lbl'>Core Framework</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-val'>Streamlit</div><div class='metric-lbl'>UI Rendering</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-val'>EfficientNet</div><div class='metric-lbl'>Biological Backbone</div></div>", unsafe_allow_html=True)

    st.markdown('''
    <div style="text-align:center; padding-top: 4rem; padding-bottom: 2rem;">
        <p style="color:#d1d1d6; font-size: 0.85rem; font-weight: 500; font-family: monospace;">© 2026 HemaType AI Research Group • Built with deeply layered neural abstraction.</p>
    </div>
    ''', unsafe_allow_html=True)
