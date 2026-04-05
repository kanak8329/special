"""
Blood Group Detection — Premium Health UI
==========================================
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2, os, sys, time, json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.predict import predict_blood_group, load_model, get_active_model_type
from utils.helpers import BLOOD_GROUPS, BLOOD_GROUP_COLORS, BLOOD_GROUP_INFO, get_sample_dir, get_model_dir

st.set_page_config(page_title="HemaType AI | Blood Group Detection", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg:       #050b18;
  --bg2:      #0a1628;
  --card:     rgba(255,255,255,0.04);
  --card-b:   rgba(255,255,255,0.09);
  --red:      #e63946;
  --red-g:    linear-gradient(135deg,#e63946,#c1121f);
  --teal:     #06d6a0;
  --blue:     #4361ee;
  --purple:   #7b2d8b;
  --gold:     #ffd60a;
  --text:     #f1f5f9;
  --muted:    #7f8fa4;
  --border:   rgba(255,255,255,0.08);
}

* { box-sizing: border-box; }

.stApp { background: var(--bg); font-family: 'Inter', sans-serif; color: var(--text); }

/* ── Floating orbs background ── */
.stApp::before {
  content:''; position:fixed; top:-20%; left:-10%; width:600px; height:600px;
  background: radial-gradient(circle, rgba(230,57,70,0.12) 0%, transparent 65%);
  border-radius:50%; pointer-events:none; z-index:0; animation: drift1 12s ease-in-out infinite alternate;
}
.stApp::after {
  content:''; position:fixed; bottom:-10%; right:-5%; width:500px; height:500px;
  background: radial-gradient(circle, rgba(67,97,238,0.1) 0%, transparent 65%);
  border-radius:50%; pointer-events:none; z-index:0; animation: drift2 15s ease-in-out infinite alternate;
}

@keyframes drift1 { from{transform:translate(0,0) scale(1)} to{transform:translate(80px,60px) scale(1.15)} }
@keyframes drift2 { from{transform:translate(0,0) scale(1)} to{transform:translate(-60px,-80px) scale(1.2)} }
@keyframes pulse  { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.7;transform:scale(1.05)} }
@keyframes fadeUp { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
@keyframes spin   { to{transform:rotate(360deg)} }
@keyframes glow   { 0%,100%{box-shadow:0 0 20px rgba(230,57,70,.3)} 50%{box-shadow:0 0 40px rgba(230,57,70,.6)} }
@keyframes heartbeat { 0%,100%{transform:scale(1)} 14%{transform:scale(1.15)} 28%{transform:scale(1)} 42%{transform:scale(1.08)} }

/* ── Glass cards ── */
.glass {
  background: var(--card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 24px;
  transition: all .35s cubic-bezier(.16,1,.3,1);
}
.glass:hover { background: var(--card-b); border-color: rgba(230,57,70,.3); transform: translateY(-4px); box-shadow: 0 24px 60px rgba(0,0,0,.4); }

/* ── Hero header ── */
.hero {
  background: linear-gradient(135deg, rgba(230,57,70,.15) 0%, rgba(67,97,238,.1) 50%, rgba(6,214,160,.08) 100%);
  border: 1px solid rgba(230,57,70,.2);
  border-radius: 28px; padding: 3rem 2.5rem; text-align: center; margin-bottom: 2rem;
  position: relative; overflow: hidden; animation: fadeUp .7s cubic-bezier(.16,1,.3,1);
}
.hero::before {
  content:''; position:absolute; top:0; left:0; right:0; bottom:0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23e63946' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none;
}
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2.8rem; font-weight:800; background:linear-gradient(135deg,#fff 30%,#e63946); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0; letter-spacing:-1px; }
.hero p  { color:var(--muted); font-size:1.1rem; margin:.8rem 0 0; }

/* ── Blood drop icon ── */
.blood-drop { font-size:3.5rem; animation: heartbeat 1.8s ease-in-out infinite; display:inline-block; }

/* ── Metric cards ── */
.metric-card {
  background: var(--card); backdrop-filter: blur(16px); border: 1px solid var(--border);
  border-radius: 20px; padding: 1.4rem; text-align: center; transition: all .3s ease;
}
.metric-card:hover { transform: translateY(-6px) scale(1.02); border-color: rgba(230,57,70,.4); box-shadow: 0 16px 40px rgba(230,57,70,.15); }
.metric-val { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:800; background:linear-gradient(135deg,var(--teal),var(--blue)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.metric-lbl { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:2px; margin-top:.3rem; }

/* ── Result reveal card ── */
.result-reveal {
  background: linear-gradient(135deg, rgba(230,57,70,.08), rgba(67,97,238,.06));
  border: 1px solid rgba(230,57,70,.3); border-radius: 28px; padding: 3.5rem 2rem;
  text-align: center; animation: fadeUp .6s cubic-bezier(.16,1,.3,1);
}
.bg-type { font-family:'Space Grotesk',sans-serif; font-size:5.5rem; font-weight:900; background:linear-gradient(135deg,#e63946,#ff6b6b,#ffd60a); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1.1; animation: glow 2s ease-in-out infinite; }
.bg-label { color:var(--muted); font-size:.85rem; text-transform:uppercase; letter-spacing:4px; font-weight:600; }
.bg-conf { color:var(--teal); font-size:1.4rem; font-weight:700; margin-top:.5rem; }

/* ── Confidence bar ── */
.conf-bar-wrap { margin: .6rem 0; }
.conf-bar-label { display:flex; justify-content:space-between; color:var(--muted); font-size:.82rem; margin-bottom:.35rem; }
.conf-bar-track { background:rgba(255,255,255,.07); border-radius:99px; height:10px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--red),var(--blue)); transition:width 1.2s cubic-bezier(.16,1,.3,1); position:relative; }
.conf-bar-fill::after { content:''; position:absolute; top:0; left:0; right:0; bottom:0; background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent); animation: shimmer 2s infinite; }
@keyframes shimmer { from{transform:translateX(-100%)} to{transform:translateX(100%)} }

/* ── Preprocessing steps ── */
.step-badge { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:.5rem .8rem; text-align:center; font-size:.78rem; color:var(--muted); margin-top:.4rem; }
.step-badge span { color:var(--teal); font-weight:600; }

/* ── Compatibility matrix ── */
.compat-grid { display:grid; grid-template-columns:repeat(8,1fr); gap:6px; }
.compat-cell { border-radius:10px; padding:.55rem .2rem; text-align:center; font-size:.7rem; font-weight:700; transition:transform .2s; cursor:default; }
.compat-cell:hover { transform:scale(1.12); z-index:1; }
.can { background:rgba(6,214,160,.2); border:1px solid rgba(6,214,160,.4); color:#06d6a0; }
.cannot { background:rgba(230,57,70,.12); border:1px solid rgba(230,57,70,.25); color:#e63946; }

/* ── Upload zone ── */
div[data-testid="stFileUploader"] { border:2px dashed rgba(230,57,70,.35); border-radius:20px; padding:2rem; background:rgba(230,57,70,.02); transition:all .3s; }
div[data-testid="stFileUploader"]:hover { border-color:rgba(230,57,70,.7); background:rgba(230,57,70,.05); }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:rgba(5,11,24,.97); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .stRadio label { color:var(--muted) !important; transition:color .2s; }
[data-testid="stSidebar"] .stRadio label:hover { color:var(--text) !important; }

/* ── Buttons ── */
.stButton button, .stDownloadButton button {
  background: var(--red-g) !important; color:#fff !important; border:none !important;
  border-radius:14px !important; font-weight:600 !important; letter-spacing:.3px !important;
  transition:all .25s cubic-bezier(.16,1,.3,1) !important;
}
.stButton button:hover, .stDownloadButton button:hover {
  transform:translateY(-2px) scale(1.02) !important;
  box-shadow:0 12px 28px rgba(230,57,70,.4) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(230,57,70,.4); border-radius:99px; }

/* ── Divider ── */
.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(230,57,70,.3),transparent); margin:2rem 0; }

/* ── Feature pill ── */
.pill { display:inline-block; padding:.3rem .9rem; border-radius:99px; border:1px solid var(--border); font-size:.78rem; color:var(--muted); margin:.2rem; background:var(--card); }

/* ── Info box ── */
.info-box { background:rgba(6,214,160,.06); border:1px solid rgba(6,214,160,.2); border-radius:16px; padding:1.2rem 1.5rem; color:#c8f7ec; font-size:.88rem; line-height:1.7; }

/* ── model badge ── */
.model-badge { display:inline-block; padding:.3rem 1rem; border-radius:99px; font-size:.8rem; font-weight:700; letter-spacing:1px; margin-left:8px; animation:pulse 2.5s ease-in-out infinite; }
.badge-cnn { background:linear-gradient(135deg,#7b2d8b,#4361ee); color:#fff; }
.badge-rf  { background:linear-gradient(135deg,#059669,#06d6a0); color:#fff; }
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
    st.markdown("<div style='text-align:center;padding:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:2.5rem;animation:heartbeat 1.8s ease-in-out infinite;display:inline-block'>🩸</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:.5rem 0 .2rem;font-family:Space Grotesk,sans-serif;font-size:1.3rem'>HemaType AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7f8fa4;font-size:.78rem;margin:0'>Blood Group Detection</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    page = st.radio("Navigate", ["🔬 Detection", "🔄 Compatibility", "📊 Architecture", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    active_model = get_active_model_type()
    badge_cls = "badge-cnn" if active_model == 'cnn' else "badge-rf"
    badge_txt = "🧠 EfficientNet-B0" if active_model == 'cnn' else "🌲 Random Forest"
    st.markdown(f"<p style='color:#7f8fa4;font-size:.75rem;text-align:center;'>Active Model</p><div style='text-align:center'><span class='model-badge {badge_cls}'>{badge_txt}</span></div>", unsafe_allow_html=True)

    if page == "🔬 Detection":
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#7f8fa4;font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px'>Upload Fingerprint</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop fingerprint image", type=['png','jpg','jpeg','bmp','tiff'], help="Upload BMP, JPG, or PNG fingerprint image", label_visibility="collapsed")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#7f8fa4;font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px'>Sample Images</p>", unsafe_allow_html=True)
        sample_dir = get_sample_dir()
        use_sample = None
        if os.path.exists(sample_dir):
            groups = sorted(os.listdir(sample_dir))
            if groups:
                sel = st.selectbox("Blood group:", groups, label_visibility="collapsed")
                gp = os.path.join(sample_dir, sel)
                samples = [f for f in os.listdir(gp) if f.lower().endswith(('.png','.bmp','.jpg'))] if os.path.exists(gp) else []
                if samples and st.button("🔍 Use Sample", use_container_width=True):
                    use_sample = os.path.join(gp, samples[0])

    st.markdown("<p style='color:#3d4b60;font-size:.7rem;text-align:center;margin-top:2rem'>HemaType AI • 2026</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: DETECTION
# ══════════════════════════════════════════════════════════════
if page == "🔬 Detection":
    model_type_label = "Deep Learning (EfficientNet-B0 CNN)" if active_model == 'cnn' else "Machine Learning (Random Forest)"
    st.markdown(f"""
    <div class='hero'>
      <div class='blood-drop'>🩸</div>
      <h1>HemaType AI</h1>
      <p>Non-invasive blood group detection from fingerprint patterns using {model_type_label}</p>
      <div style='margin-top:1rem'>
        <span class='pill'>🎮 GPU Accelerated</span>
        <span class='pill'>🧠 Transfer Learning</span>
        <span class='pill'>📊 8 Blood Groups</span>
        <span class='pill'>⚡ ~2s Prediction</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    model_dir = get_model_dir()
    cnn_exists = os.path.exists(os.path.join(model_dir, 'cnn_model.pth'))
    rf_exists = os.path.exists(os.path.join(model_dir, 'blood_group_model.pkl'))

    if not cnn_exists and not rf_exists:
        st.error("⚠️ **No trained model found!** Please run training first:\n```\npython model/train.py --mode cnn --dataset-path data/sample_fingerprints\n```")
        st.stop()

    image_to_process = None
    if 'uploaded_file' in dir() and uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif 'use_sample' in dir() and use_sample:
        image_to_process = use_sample

    if image_to_process is not None:
        with st.spinner(""):
            st.markdown("""
            <div style='text-align:center;padding:2rem;'>
              <div style='font-size:2rem;animation:spin 1.2s linear infinite;display:inline-block'>⚙️</div>
              <p style='color:#7f8fa4;margin:.5rem 0 0'>Analyzing fingerprint with AI...</p>
            </div>""", unsafe_allow_html=True)
            t0 = time.time()
            try:
                result = predict_blood_group(image_to_process)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()
            elapsed = time.time() - t0

        predicted = result['predicted_group']
        confidence = result['confidence']
        all_scores = result['all_scores']
        breakdown = result['feature_breakdown']

        # ── Result Card ──────────────────────────────────────────────
        r1, r2, r3 = st.columns([1, 1.2, 1])
        with r2:
            st.markdown(f"""
            <div class='result-reveal'>
              <div class='bg-label'>Predicted Blood Group</div>
              <div class='bg-type'>{predicted}</div>
              <div class='bg-conf'>✓ {confidence*100:.1f}% Confidence</div>
              <div style='color:#7f8fa4;font-size:.85rem;margin-top:.5rem'>⏱ {elapsed:.2f}s  •  {badge_txt}</div>
              <div style='margin-top:1.2rem' class='info-box'>{BG_FACTS.get(predicted,'')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        metrics = [
            (mc1, f"{confidence*100:.1f}%", "Confidence"),
            (mc2, f"{elapsed:.2f}s", "Process Time"),
            (mc3, "CNN" if active_model=='cnn' else "RF", "Model Type"),
            (mc4, "8", "Blood Groups"),
        ]
        for col, val, lbl in metrics:
            with col:
                st.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Confidence Bars ───────────────────────────────────────────
        col_bar, col_prep = st.columns([1, 1])

        with col_bar:
            st.markdown("### 📊 Confidence Scores")
            bars_html = ""
            for s in all_scores:
                pct = s['confidence'] * 100
                is_top = s['group'] == predicted
                fill_color = "linear-gradient(90deg,#e63946,#ff6b6b)" if is_top else "linear-gradient(90deg,#4361ee,#06d6a0)"
                weight = "700" if is_top else "400"
                bars_html += f"""
                <div class='conf-bar-wrap'>
                  <div class='conf-bar-label'><span style='font-weight:{weight};color:{'#fff' if is_top else '#7f8fa4'}'>{s['group']}</span><span style='color:{'#e63946' if is_top else '#7f8fa4'}'>{pct:.1f}%</span></div>
                  <div class='conf-bar-track'><div class='conf-bar-fill' style='width:{min(pct,100):.1f}%;background:{fill_color}'></div></div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

        with col_prep:
            st.markdown("### 🔄 Preprocessing Steps")
            try:
                from utils.preprocessing import preprocess_fingerprint
                steps = result.get('preprocessing_steps', {})
                step_keys = [('original','Original'),('grayscale','Grayscale'),('denoised','Denoised'),('enhanced','CLAHE'),('thresholded','Binary')]
                scols = st.columns(5)
                for sc, (key, name) in zip(scols, step_keys):
                    img = steps.get(key)
                    if img is not None:
                        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape)==3 else img
                        sc.image(disp, use_container_width=True)
                        sc.markdown(f"<div class='step-badge'><span>{name}</span></div>", unsafe_allow_html=True)
            except Exception:
                st.info("Preprocessing steps unavailable.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Donation Info ─────────────────────────────────────────────
        st.markdown("### 💉 Blood Compatibility")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("<p style='color:#7f8fa4;font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.8rem'>Can Donate To</p>", unsafe_allow_html=True)
            donate_html = "".join([f"<span style='display:inline-block;margin:.3rem;padding:.5rem 1.2rem;background:rgba(6,214,160,.15);border:1px solid rgba(6,214,160,.3);border-radius:12px;color:#06d6a0;font-weight:700;font-family:Space Grotesk,sans-serif'>{g}</span>" for g in DONATE_TO.get(predicted, [])])
            st.markdown(f"<div style='animation:fadeUp .5s ease'>{donate_html}</div>", unsafe_allow_html=True)
        with dc2:
            st.markdown("<p style='color:#7f8fa4;font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.8rem'>Can Receive From</p>", unsafe_allow_html=True)
            recv_html = "".join([f"<span style='display:inline-block;margin:.3rem;padding:.5rem 1.2rem;background:rgba(67,97,238,.15);border:1px solid rgba(67,97,238,.3);border-radius:12px;color:#7da8ff;font-weight:700;font-family:Space Grotesk,sans-serif'>{g}</span>" for g in RECEIVE_FROM.get(predicted, [])])
            st.markdown(f"<div style='animation:fadeUp .5s ease'>{recv_html}</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────────────
        st.markdown("### 📥 Export Report")
        report = {
            "predicted_blood_group": predicted,
            "confidence_score": float(confidence),
            "processing_time_seconds": round(elapsed, 3),
            "model_type": result.get('model_type','unknown'),
            "model_architecture": breakdown.get('architecture','Unknown'),
            "can_donate_to": DONATE_TO.get(predicted,[]),
            "can_receive_from": RECEIVE_FROM.get(predicted,[]),
            "blood_group_fact": BG_FACTS.get(predicted,''),
            "all_scores": {s['group']: round(s['confidence']*100,2) for s in all_scores},
        }
        st.download_button("📄 Download Full Report (JSON)", json.dumps(report, indent=4), "hematype_report.json", "application/json", use_container_width=True)

    else:
        # ── Upload Prompt ─────────────────────────────────────────────
        st.markdown("""
        <div style='display:flex;align-items:center;justify-content:center;min-height:300px'>
          <div class='glass' style='text-align:center;padding:4rem 3rem;max-width:480px;margin:2rem auto;animation:fadeUp .6s ease'>
            <div style='font-size:4rem;margin-bottom:1rem;animation:heartbeat 1.8s ease-in-out infinite'>🖐️</div>
            <h3 style='color:#fff;font-family:Space Grotesk,sans-serif;margin:.5rem 0'>Upload a Fingerprint</h3>
            <p style='color:#7f8fa4;font-size:.9rem;line-height:1.6'>Use the sidebar to upload a fingerprint image (BMP, PNG, JPG) or select a sample image for a quick demo.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔬 How It Works")
        steps_info = [
            ("📤","Upload","Upload a fingerprint image in BMP, PNG or JPG format from the sidebar"),
            ("🔄","Preprocess","Image is resized to 224×224, denoised, CLAHE enhanced & normalized"),
            ("🧠","CNN Inference","EfficientNet-B0 runs a forward pass through 7 MBConv stages on GPU"),
            ("🩸","Predict","Softmax outputs confidence scores for all 8 blood group classes"),
        ]
        cols = st.columns(4)
        for i, (col, (icon, title, desc)) in enumerate(zip(cols, steps_info)):
            with col:
                st.markdown(f"""
                <div class='glass' style='padding:1.8rem;text-align:center;min-height:200px;animation:fadeUp {.3+i*.1}s ease'>
                  <div style='font-size:2.2rem;margin-bottom:.8rem'>{icon}</div>
                  <h4 style='color:#fff;font-family:Space Grotesk,sans-serif;margin:.3rem 0'>{title}</h4>
                  <p style='color:#7f8fa4;font-size:.82rem;line-height:1.5'>{desc}</p>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: COMPATIBILITY CHECKER
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Compatibility":
    st.markdown("""
    <div class='hero'>
      <div class='blood-drop'>🔄</div>
      <h1>Blood Type Compatibility</h1>
      <p>Interactive blood group compatibility checker — find who you can donate to or receive from</p>
    </div>""", unsafe_allow_html=True)

    sel_bg = st.selectbox("Select a Blood Group to explore:", BLOOD_GROUPS, key="compat_select")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='glass' style='padding:2rem;'>
          <h3 style='color:#06d6a0;font-family:Space Grotesk,sans-serif;margin-bottom:1.2rem'>💉 Can Donate To</h3>
          {''.join([f"<div style='display:flex;align-items:center;gap:.8rem;padding:.8rem;margin:.4rem 0;background:rgba(6,214,160,.08);border:1px solid rgba(6,214,160,.2);border-radius:14px;transition:.2s'><span style='font-size:1.4rem;font-weight:900;font-family:Space Grotesk,sans-serif;color:#06d6a0;min-width:45px'>{g}</span><span style='color:#7f8fa4;font-size:.85rem'>{BLOOD_GROUP_INFO.get(g,'')[:60]}...</span></div>" for g in DONATE_TO.get(sel_bg,[])])}
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='glass' style='padding:2rem;'>
          <h3 style='color:#4361ee;font-family:Space Grotesk,sans-serif;margin-bottom:1.2rem'>🩸 Can Receive From</h3>
          {''.join([f"<div style='display:flex;align-items:center;gap:.8rem;padding:.8rem;margin:.4rem 0;background:rgba(67,97,238,.08);border:1px solid rgba(67,97,238,.2);border-radius:14px'><span style='font-size:1.4rem;font-weight:900;font-family:Space Grotesk,sans-serif;color:#7da8ff;min-width:45px'>{g}</span><span style='color:#7f8fa4;font-size:.85rem'>{BLOOD_GROUP_INFO.get(g,'')[:60]}...</span></div>" for g in RECEIVE_FROM.get(sel_bg,[])])}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🗓️ Full Compatibility Matrix")

    header = "<div class='compat-grid' style='margin-bottom:6px'><div></div>" + "".join(f"<div style='text-align:center;font-size:.7rem;color:#7f8fa4;font-weight:700'>{g}</div>" for g in BLOOD_GROUPS) + "</div>"
    rows = ""
    for donor in BLOOD_GROUPS:
        rows += f"<div class='compat-grid' style='margin-bottom:6px'><div style='font-size:.7rem;color:#fff;font-weight:700;display:flex;align-items:center'>{donor}</div>"
        for recipient in BLOOD_GROUPS:
            can = recipient in DONATE_TO.get(donor, [])
            rows += f"<div class='compat-cell {'can' if can else 'cannot'}'>{'✓' if can else '✗'}</div>"
        rows += "</div>"
    st.markdown(f"<div style='overflow-x:auto'>{header}{rows}</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-box'>💡 <strong>About {sel_bg}:</strong> {BLOOD_GROUP_INFO.get(sel_bg,'')}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Architecture":
    st.markdown("""
    <div class='hero'>
      <div class='blood-drop'>📊</div>
      <h1>System Architecture</h1>
      <p>Technical overview of the HemaType AI pipeline — EfficientNet-B0 with GPU acceleration</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔄 CNN Pipeline")
    st.code("""
Input (BMP/PNG/JPG)  →  Preprocess (224×224, ImageNet Norm)
   ↓
EfficientNet-B0 Backbone (7 MBConv Stages, pretrained ImageNet)
   ↓  GlobalAveragePooling  →  [1280]
Custom Head:
   Dropout(0.35) → Linear(1280→512) → BN → ReLU
   Dropout(0.30) → Linear(512→256)  → BN → ReLU
   Dropout(0.20) → Linear(256→8)
   ↓  Softmax  →  Blood Group + Confidence Score
    """, language="text")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    cols = st.columns(3)
    arch_cards = [
        ("🔥 PyTorch 2.5", "Deep Learning", "CUDA 12.1, Mixed Precision FP16, AdamW optimizer"),
        ("🏺 EfficientNet-B0", "CNN Backbone", "5.3M params, ImageNet pretrained, 7 MBConv stages"),
        ("🔄 Data Augmentation", "Regularization", "Flip, Rotate±20°, ColorJitter, Perspective, Erasing"),
        ("⚡ Mixed Precision", "GPU Training", "GradScaler + autocast for 2× speed on GTX 1660 Ti"),
        ("📈 Cosine Annealing", "LR Schedule", "Warm restarts (T₀=20), eta_min=1e-6"),
        ("🏋️ Progressive Unfreeze", "Fine-tuning", "Head-only (ep 1-5), Full model (ep 6+) for stability"),
    ]
    for i, (col, (title, cat, desc)) in enumerate(zip(cols*2, arch_cards)):
        with col:
            st.markdown(f"""<div class='glass' style='padding:1.5rem;margin-bottom:1rem;animation:fadeUp {.2+i*.1}s ease'>
              <p style='color:#e63946;font-size:.75rem;text-transform:uppercase;letter-spacing:1.5px;margin:0'>{cat}</p>
              <h4 style='color:#fff;font-family:Space Grotesk,sans-serif;margin:.4rem 0'>{title}</h4>
              <p style='color:#7f8fa4;font-size:.82rem;line-height:1.5;margin:0'>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class='hero'>
      <div class='blood-drop'>ℹ️</div>
      <h1>About HemaType AI</h1>
      <p>Non-invasive blood group detection using Deep Learning — a college research project</p>
    </div>""", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class='glass' style='padding:2rem;'>
          <h3 style='color:#fff;font-family:Space Grotesk,sans-serif'>📋 Project Overview</h3>
          <p style='color:#7f8fa4;line-height:1.8;font-size:.9rem'>
          This project detects blood groups from fingerprint images using <strong style='color:#e63946'>EfficientNet-B0</strong>, 
          a state-of-the-art CNN architecture. Traditional blood group testing requires blood sampling, 
          but our non-invasive approach uses only a fingerprint image — making testing faster, safer, and more accessible.
          <br><br>
          Trained on <strong style='color:#06d6a0'>7,470 real fingerprint images</strong> across 8 blood groups with 
          GPU acceleration on NVIDIA GTX 1660 Ti using PyTorch 2.5 and CUDA 12.1.
          </p>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class='glass' style='padding:2rem;'>
          <h3 style='color:#fff;font-family:Space Grotesk,sans-serif'>✨ Key Features</h3>
          <div style='display:flex;flex-direction:column;gap:.7rem;margin-top:.5rem'>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#e63946;font-size:1rem'>🧠</span> EfficientNet-B0 CNN (Transfer Learning)</div>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#06d6a0;font-size:1rem'>⚡</span> GPU + Mixed Precision FP16 Training</div>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#4361ee;font-size:1rem'>🔄</span> Progressive Backbone Unfreezing</div>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#ffd60a;font-size:1rem'>📊</span> Class-Weighted Loss for Imbalanced Data</div>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#e63946;font-size:1rem'>🔄</span> Interactive Compatibility Checker</div>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#06d6a0;font-size:1rem'>📥</span> JSON Report Export</div>
            <div style='display:flex;align-items:center;gap:.8rem;color:#7f8fa4;font-size:.88rem'><span style='color:#4361ee;font-size:1rem'>🎨</span> Premium Glassmorphism UI</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass' style='padding:2rem;text-align:center;'>
      <p style='color:#7f8fa4;font-size:.82rem;text-transform:uppercase;letter-spacing:2px'>References</p>
      <p style='color:#7f8fa4;font-size:.85rem;line-height:2'>
        Tan et al. (2019) — EfficientNet: Rethinking Model Scaling for CNNs  •  
        PyTorch Documentation — Transfer Learning Tutorial  •  
        Kaggle — Blood Group Fingerprint Dataset  •  
        Krishna et al. — Fingerprint Blood Group Detection (GitHub)
      </p>
      <p style='color:#3d4b60;font-size:.78rem;margin-top:1rem'>🎓 Machine Learning Research Project · 2026</p>
    </div>""", unsafe_allow_html=True)
