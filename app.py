# -*- coding: utf-8 -*-
"""
HemaType AI v4.0 — Slide-Deck Clinical Interface
=================================================
Run: streamlit run app.py
"""

import os
import sys
import time
import json
import datetime
import numpy as np
from PIL import Image

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import (
    BLOOD_GROUPS, BLOOD_GROUP_COLORS, BLOOD_GROUP_INFO,
    get_sample_dir, get_model_dir
)
from model.predict import (
    predict_blood_group, load_model,
    get_active_model_type, _get_available_cnn_variants
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HemaType AI | Clinical Diagnostics",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_SLIDES = 7

def _init_state():
    defaults = {
        "slide": 0,
        "image_source": None,
        "scan_results": None,
        "selected_variants": None,
        "scan_elapsed": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
#  COMPATIBILITY DATA
# ─────────────────────────────────────────────────────────────────────────────
DONATE_TO = {
    'A+':  ['A+', 'AB+'],
    'A-':  ['A+', 'A-', 'AB+', 'AB-'],
    'B+':  ['B+', 'AB+'],
    'B-':  ['B+', 'B-', 'AB+', 'AB-'],
    'AB+': ['AB+'],
    'AB-': ['AB+', 'AB-'],
    'O+':  ['A+', 'B+', 'AB+', 'O+'],
    'O-':  BLOOD_GROUPS[:],
}
RECEIVE_FROM = {
    'A+':  ['A+', 'A-', 'O+', 'O-'],
    'A-':  ['A-', 'O-'],
    'B+':  ['B+', 'B-', 'O+', 'O-'],
    'B-':  ['B-', 'O-'],
    'AB+': BLOOD_GROUPS[:],
    'AB-': ['A-', 'B-', 'AB-', 'O-'],
    'O+':  ['O+', 'O-'],
    'O-':  ['O-'],
}
BG_FACTS = {
    'A+':  '30% of people have A+ — the second most common blood type.',
    'A-':  'A- can donate to 4 blood types including A+ and AB+.',
    'B+':  'B+ is found in about 9% of the population worldwide.',
    'B-':  'B- is rare — only ~2% of people have this blood type.',
    'AB+': 'AB+ is the Universal Recipient — can receive all blood types!',
    'AB-': 'AB- is very rare — less than 1% of people worldwide.',
    'O+':  'O+ is the most common blood type — 38% of all people.',
    'O-':  'O- is the Universal Donor — can donate to all blood types!',
}

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

/* ── Reset & Core ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --red:    #e11d48;
    --red-glow: rgba(225, 29, 72, 0.35);
    --teal:   #0d9488;
    --blue:   #2563eb;
    --gold:   #f59e0b;
    --bg:     #070c18;
    --card:   #0f172a;
    --card2:  #111827;
    --border: rgba(255,255,255,0.07);
    --text:   #f1f5f9;
    --muted:  #64748b;
    --accent: #e11d48;
}

/* ── Hide all Streamlit chrome ── */
#MainMenu, footer, header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
section[data-testid="stSidebar"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── App Background ── */
.stApp,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"] {
    background: var(--bg) !important;
}

div[data-testid="block-container"] {
    padding-top: 1rem !important;
    padding-bottom: 6rem !important; /* Space for nav bar */
}

/* ── Slide Container ── */
.slide-wrap {
    min-height: 100vh;
    width: 100%;
    display: flex;
    flex-direction: column;
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    color: var(--text);
    position: relative;
    overflow: hidden;
}

/* ── Navigation Bar ── */
.slide-nav {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 72px;
    background: rgba(7, 12, 24, 0.92);
    backdrop-filter: blur(20px);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    z-index: 9999;
}
.slide-dots {
    display: flex;
    gap: 10px;
    align-items: center;
}
.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--muted);
    transition: all 0.3s ease;
    cursor: pointer;
}
.dot.active {
    background: var(--red);
    width: 28px;
    border-radius: 4px;
    box-shadow: 0 0 10px var(--red-glow);
}
.nav-label {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Slide Content Padding (accounts for nav bar) ── */
.slide-content {
    flex: 1;
    margin-bottom: 2rem;
}

/* ── Hero / Slide 0 ── */
.hero-bg {
    background:
        radial-gradient(circle at 20% 50%, rgba(225,29,72,0.12) 0%, transparent 50%),
        radial-gradient(circle at 80% 50%, rgba(37,99,235,0.1) 0%, transparent 50%),
        var(--bg);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(3rem, 7vw, 6rem);
    font-weight: 900;
    letter-spacing: -3px;
    background: linear-gradient(135deg, #ffffff 0%, #fda4af 40%, #e11d48 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 1.5rem;
}
.hero-sub {
    font-size: 1.2rem;
    color: var(--muted);
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}
.hero-badge {
    display: inline-block;
    background: rgba(225,29,72,0.12);
    border: 1px solid rgba(225,29,72,0.3);
    color: #fda4af;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Slide Header ── */
.slide-header {
    margin-bottom: 2.5rem;
}
.slide-tag {
    display: inline-block;
    background: rgba(225,29,72,0.12);
    border: 1px solid rgba(225,29,72,0.25);
    color: #fda4af;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.slide-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.slide-desc {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 400;
    max-width: 700px;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    overflow: hidden;
    word-wrap: break-word;
}
.card:hover {
    border-color: rgba(225,29,72,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}
.card-sm {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
}

/* ── Upload Zone ── */
div[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed rgba(225,29,72,0.3) !important;
    border-radius: 18px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(225,29,72,0.6) !important;
    background: rgba(225,29,72,0.04) !important;
    box-shadow: 0 0 30px rgba(225,29,72,0.08) !important;
}
div[data-testid="stFileUploader"] * {
    color: var(--text) !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--muted) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(225,29,72,0.35) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #f43f5e !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(225,29,72,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Secondary / ghost button ── */
.btn-ghost > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}
.btn-ghost > button:hover {
    background: var(--card2) !important;
    color: var(--text) !important;
    border-color: rgba(255,255,255,0.15) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Select / Radio ── */
div[data-baseweb="select"] > div {
    background: var(--card2) !important;
    border-color: var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: var(--text) !important;
}
div[role="listbox"] {
    background: #0d1424 !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[role="option"]:hover {
    background: rgba(225,29,72,0.1) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: rgba(13,148,136,0.15) !important;
    color: #5eead4 !important;
    border: 1px solid rgba(13,148,136,0.4) !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: rgba(13,148,136,0.25) !important;
    box-shadow: 0 4px 20px rgba(13,148,136,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Step Progress Bar ── */
.step-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 2.5rem;
}
.step-item {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.08);
    transition: background 0.4s ease;
}
.step-item.done {
    background: var(--red);
    box-shadow: 0 0 8px var(--red-glow);
}
.step-item.active {
    background: linear-gradient(90deg, var(--red), #f43f5e);
    box-shadow: 0 0 12px var(--red-glow);
}

/* ── Fingerprint Preview ── */
.fp-preview-wrap {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: #000;
}
.scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
    box-shadow: 0 0 12px var(--red);
    animation: scanAnim 2.5s ease-in-out infinite;
}
@keyframes scanAnim {
    0%   { top: 0%; opacity: 1; }
    49%  { top: 97%; opacity: 1; }
    50%  { top: 97%; opacity: 0; }
    51%  { top: 0%;  opacity: 0; }
    52%  { top: 0%;  opacity: 1; }
    100% { top: 0%; opacity: 1; }
}
@keyframes scanPulse {
    0%, 100% { box-shadow: 0 0 10px var(--red-glow); }
    50%       { box-shadow: 0 0 25px rgba(225,29,72,0.6); }
}

/* ── Result / Report ── */
.result-card {
    background: linear-gradient(135deg, rgba(225,29,72,0.08) 0%, var(--card) 60%);
    border: 1px solid rgba(225,29,72,0.2);
    border-left: 5px solid var(--red);
    border-radius: 20px;
    padding: 2.5rem;
    animation: slideIn 0.5s cubic-bezier(0.25,0.8,0.25,1);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-blood-type {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(4rem, 10vw, 8rem);
    font-weight: 900;
    line-height: 1;
    color: var(--red);
    text-shadow: 0 0 50px rgba(225,29,72,0.4);
    letter-spacing: -3px;
}
.result-meta-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.result-meta-value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
}

/* ── Confidence Bars ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.conf-group {
    width: 36px;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--muted);
    flex-shrink: 0;
    text-align: right;
}
.conf-group.top { color: var(--text); }
.conf-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1.2s cubic-bezier(0.19,1,0.22,1);
}
.conf-pct {
    width: 42px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    flex-shrink: 0;
    text-align: right;
}
.conf-pct.top { color: var(--text); }

/* ── Compatibility Pills ── */
.pill {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-radius: 14px;
    transition: all 0.2s ease;
}
.pill-donate {
    background: rgba(13,148,136,0.08);
    border: 1px solid rgba(13,148,136,0.2);
    border-left: 4px solid var(--teal);
}
.pill-donate:hover { transform: translateX(5px); background: rgba(13,148,136,0.15); }
.pill-receive {
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.2);
    border-left: 4px solid var(--blue);
}
.pill-receive:hover { transform: translateX(5px); background: rgba(37,99,235,0.15); }
.pill-type {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    min-width: 44px;
}
.pill-donate .pill-type { color: #5eead4; }
.pill-receive .pill-type { color: #93c5fd; }
.pill-info { font-size: 0.82rem; color: var(--muted); line-height: 1.4; }

/* ── Compat Matrix ── */
.compat-matrix {
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 6px;
}
.cm-hdr, .cm-row-hdr {
    display: flex; align-items: center; justify-content: center;
    padding: 0.6rem 0;
    font-size: 0.78rem;
    font-weight: 700;
    border-radius: 10px;
}
.cm-hdr { color: var(--muted); }
.cm-row-hdr {
    color: var(--text);
    background: rgba(255,255,255,0.04);
    font-size: 0.82rem;
}
.cm-cell {
    display: flex; align-items: center; justify-content: center;
    padding: 0.6rem 0;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
    transition: transform 0.15s ease;
    cursor: default;
}
.cm-cell:hover { transform: scale(1.2); z-index: 2; position: relative; }
.cm-yes { background: rgba(13,148,136,0.15); color: #5eead4; }
.cm-no  { background: rgba(255,255,255,0.02); color: rgba(255,255,255,0.1); font-size: 0.8rem; font-weight: 400; }

/* ── Architecture Flow ── */
.flow-node {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: all 0.3s ease;
    margin-bottom: 0;
}
.flow-node:hover {
    border-color: rgba(225,29,72,0.3);
    transform: translateX(4px);
}
.flow-icon {
    font-size: 2rem;
    width: 58px; height: 58px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(225,29,72,0.1);
    border-radius: 14px;
    flex-shrink: 0;
}
.flow-text h4 {
    font-size: 0.95rem; font-weight: 700;
    color: var(--text); margin-bottom: 0.25rem;
}
.flow-text p {
    font-size: 0.82rem; color: var(--muted); line-height: 1.5;
}
.flow-badge {
    font-family: monospace;
    font-size: 0.7rem;
    background: rgba(37,99,235,0.15);
    color: #93c5fd;
    border: 1px solid rgba(37,99,235,0.2);
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-top: 0.4rem;
}
.flow-arrow {
    text-align: center;
    color: var(--muted);
    font-size: 1.2rem;
    padding: 0.3rem 0;
    margin: 2px 0;
}

/* ── Tech Cards ── */
.tech-card {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    transition: all 0.3s ease;
}
.tech-card:hover {
    border-color: rgba(225,29,72,0.2);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.tech-cat {
    font-size: 0.65rem;
    font-weight: 700;
    color: #fda4af;
    background: rgba(225,29,72,0.1);
    border: 1px solid rgba(225,29,72,0.2);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    display: inline-block;
    margin-bottom: 0.8rem;
}
.tech-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.tech-desc {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── About ── */
.about-stat {
    text-align: center;
    padding: 1.5rem;
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
}
.about-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--red);
    text-shadow: 0 0 20px var(--red-glow);
}
.about-stat-lbl {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.06); }
}
.anim-fadeup { animation: fadeUp 0.5s ease both; }
.anim-pulse  { animation: pulse 2s ease-in-out infinite; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* ── Streamlit image centering ── */
div[data-testid="stImage"] img {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  NAVIGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SLIDE_LABELS = [
    "Welcome", "Upload", "Scan", "Report",
    "Compatibility", "Architecture", "About"
]

def go_to(n):
    st.session_state.slide = max(0, min(TOTAL_SLIDES - 1, n))

def render_nav():
    """Fixed bottom navigation: dots + prev/next."""
    cur = st.session_state.slide
    dots_html = "<div class='slide-dots'>"
    for i, lbl in enumerate(SLIDE_LABELS):
        cls = "dot active" if i == cur else "dot"
        dots_html += f"<div class='{cls}' title='{lbl}'></div>"
    dots_html += "</div>"

    nav_html = f"""
    <div class='slide-nav'>
        <span class='nav-label'>{SLIDE_LABELS[cur]}</span>
        {dots_html}
        <span class='nav-label'>Slide {cur + 1} / {TOTAL_SLIDES}</span>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)

    # Prev / Next buttons
    col_prev, col_spacer, col_next = st.columns([1, 8, 1])
    with col_prev:
        if cur > 0:
            with st.container():
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("← Back", key="nav_prev"):
                    go_to(cur - 1)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    with col_next:
        if cur < TOTAL_SLIDES - 1 and cur not in [1, 2]:
            if st.button("Next →", key="nav_next"):
                go_to(cur + 1)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def slide_hero():
    active_model = get_active_model_type()
    variants = _get_available_cnn_variants()
    model_badge = "Dual CNN (B3 + B0)" if len(variants) > 1 else ("EfficientNet CNN" if variants else "Random Forest")

    st.markdown(f"""
    <div class='hero-bg slide-content' style='padding-bottom:5rem;'>
        <div class='anim-fadeup'>
            <div class='anim-pulse' style='font-size:4rem;margin-bottom:1rem;'>🩸</div>
            <div class='hero-badge'>🤖 {model_badge} Active</div>
            <h1 class='hero-title'>HemaType AI</h1>
            <p class='hero-sub'>
                Non-invasive blood group analysis from fingerprint images.<br>
                Powered by dual EfficientNet-B3 &amp; B0 convolutional neural networks.
            </p>
            <div style='display:flex;gap:2rem;justify-content:center;margin-bottom:3rem;flex-wrap:wrap;'>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem;font-weight:800;color:#e11d48;'>8</div>
                    <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Blood Groups</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem;font-weight:800;color:#e11d48;'>12.2M</div>
                    <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Parameters</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem;font-weight:800;color:#e11d48;'>&lt;1s</div>
                    <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Inference</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-size:1.8rem;font-weight:800;color:#e11d48;'>0</div>
                    <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Needles</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_b:
        if st.button("🔬 Begin Clinical Scan →", key="hero_start"):
            go_to(1)
            st.rerun()

    st.markdown("""
    <div style='display:flex;gap:3rem;justify-content:center;margin-top:3rem;padding-bottom:2rem;flex-wrap:wrap;'>
        <div style='text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:0.4rem;'>📤</div>
            <div style='font-size:0.8rem;color:#64748b;'>Upload Fingerprint</div>
        </div>
        <div style='color:#374151;font-size:1.5rem;align-self:center;'>→</div>
        <div style='text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:0.4rem;'>🧠</div>
            <div style='font-size:0.8rem;color:#64748b;'>AI Analysis</div>
        </div>
        <div style='color:#374151;font-size:1.5rem;align-self:center;'>→</div>
        <div style='text-align:center;'>
            <div style='font-size:1.5rem;margin-bottom:0.4rem;'>🩸</div>
            <div style='font-size:0.8rem;color:#64748b;'>Diagnostic Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def slide_upload():
    st.markdown("""
    <div class='slide-content anim-fadeup'>
        <div class='slide-header'>
            <div class='slide-tag'>Step 1 of 3</div>
            <h1 class='slide-title'>Patient Fingerprint Input</h1>
            <p class='slide-desc'>Upload a clear fingerprint scan (BMP, PNG, JPG) or select a pre-processed clinical sample from our library.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 4rem;'>", unsafe_allow_html=True)

    # Step progress bar
    st.markdown("""
    <div class='step-bar' style='margin-bottom:2rem;'>
        <div class='step-item active'></div>
        <div class='step-item'></div>
        <div class='step-item'></div>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_sample = st.columns([3, 2], gap="large")

    with col_up:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='color:#f1f5f9;font-size:1.1rem;font-weight:700;margin-bottom:1rem;'>
            📁 Upload Patient Scan
        </h3>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload fingerprint image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            label_visibility="collapsed",
            key="uploader"
        )
        if uploaded:
            img = Image.open(uploaded)
            st.session_state.image_source = img
            st.session_state.scan_results = None
            go_to(2)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sample:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='color:#f1f5f9;font-size:1.1rem;font-weight:700;margin-bottom:1rem;'>
            🧪 Clinical Sample Library
        </h3>
        """, unsafe_allow_html=True)
        sample_dir = get_sample_dir()
        if os.path.exists(sample_dir):
            groups = sorted(os.listdir(sample_dir))
            if groups:
                sel = st.selectbox("Select Blood Group Sample", groups, key="sample_sel")
                gp = os.path.join(sample_dir, sel)
                samples = [f for f in os.listdir(gp) if f.lower().endswith(('.png', '.bmp', '.jpg'))] if os.path.exists(gp) else []
                if samples:
                    if st.button("⚗️ Load Clinical Sample", key="load_sample"):
                        st.session_state.image_source = os.path.join(gp, samples[0])
                        st.session_state.scan_results = None
                        go_to(2)
                        st.rerun()
                else:
                    st.markdown("<p style='color:#64748b;font-size:0.85rem;'>No sample images found in this group.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#64748b;font-size:0.85rem;'>No sample groups found.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#64748b;font-size:0.85rem;'>Sample library not configured.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-advance handles progression, no manual button needed.


def slide_scan():
    if st.session_state.image_source is None:
        st.markdown("""
        <div class='slide-content anim-fadeup' style='display:flex;align-items:center;justify-content:center;min-height:70vh;'>
            <div style='text-align:center;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>⚠️</div>
                <h3 style='color:#f1f5f9;margin-bottom:0.5rem;'>No fingerprint loaded</h3>
                <p style='color:#64748b;'>Please go back and upload or select a sample first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <div class='slide-content anim-fadeup'>
        <div class='slide-header'>
            <div class='slide-tag'>Step 2 of 3</div>
            <h1 class='slide-title'>Scan Configuration</h1>
            <p class='slide-desc'>Review the fingerprint image, select the neural network backbone, and initiate the clinical scan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 4rem;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='step-bar' style='margin-bottom:2rem;'>
        <div class='step-item done'></div>
        <div class='step-item active'></div>
        <div class='step-item'></div>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_cfg = st.columns([2, 3], gap="large")

    with col_img:
        st.markdown("<div class='card' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:0.75rem;'>Patient Fingerprint</p>", unsafe_allow_html=True)

        # Load image for display
        try:
            if isinstance(st.session_state.image_source, str):
                img_display = Image.open(st.session_state.image_source).convert("L")
            else:
                img_display = st.session_state.image_source.convert("L")
            st.image(img_display, use_container_width=True)
        except Exception:
            st.error("Could not load image preview.")

        st.markdown("""
        <div style='margin-top:0.75rem;background:rgba(225,29,72,0.08);border:1px solid rgba(225,29,72,0.2);border-radius:10px;padding:0.6rem 1rem;'>
            <div style='display:flex;align-items:center;gap:0.5rem;'>
                <div style='width:8px;height:8px;border-radius:50%;background:#e11d48;animation:pulse 1.5s ease-in-out infinite;'></div>
                <span style='font-size:0.75rem;font-weight:700;color:#fda4af;'>Scan overlay active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cfg:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#f1f5f9;font-size:1.1rem;font-weight:700;margin-bottom:1.5rem;'>🧠 Neural Network Configuration</h3>", unsafe_allow_html=True)

        variants = _get_available_cnn_variants()
        if not variants:
            variants = ['efficientnet_b3']

        options = ["🔬 Run Both (Dual Comparison)"] + [f"EfficientNet-{v.split('_')[-1].upper()}" for v in variants]
        selected_mode = st.selectbox("Select Model Architecture", options, key="model_sel")

        if selected_mode == "🔬 Run Both (Dual Comparison)":
            variants_to_run = variants
        else:
            suffix = selected_mode.split("-")[-1].lower()
            variants_to_run = [v for v in variants if v.endswith(suffix)]

        # Model info cards
        for v in variants_to_run:
            params = "12.2M" if "b3" in v else "5.3M"
            size = "300×300 px" if "b3" in v else "224×224 px"
            st.markdown(f"""
            <div style='background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.2);border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.5rem;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-weight:700;color:#93c5fd;font-size:0.9rem;'>{v.replace("efficientnet_","EfficientNet-").upper()}</span>
                    <span style='font-size:0.7rem;background:rgba(37,99,235,0.15);color:#93c5fd;padding:2px 8px;border-radius:10px;border:1px solid rgba(37,99,235,0.3);'>Active</span>
                </div>
                <div style='display:flex;gap:2rem;margin-top:0.5rem;'>
                    <span style='font-size:0.75rem;color:#64748b;'>Params: <strong style='color:#94a3b8;'>{params}</strong></span>
                    <span style='font-size:0.75rem;color:#64748b;'>Input: <strong style='color:#94a3b8;'>{size}</strong></span>
                    <span style='font-size:0.75rem;color:#64748b;'>Backbone: <strong style='color:#94a3b8;'>ImageNet</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        if st.button("🔴 INITIATE CLINICAL SCAN", key="run_scan"):
            with st.spinner("Running AI inference..."):
                try:
                    t0 = time.time()
                    results = predict_blood_group(
                        st.session_state.image_source,
                        selected_variants=variants_to_run
                    )
                    elapsed = time.time() - t0
                    st.session_state.scan_results = results
                    st.session_state.scan_elapsed = elapsed
                    go_to(3)
                    st.rerun()
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def slide_report():
    if not st.session_state.scan_results:
        st.markdown("""
        <div class='slide-content anim-fadeup' style='display:flex;align-items:center;justify-content:center;min-height:70vh;'>
            <div style='text-align:center;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📋</div>
                <h3 style='color:#f1f5f9;margin-bottom:0.5rem;'>No scan results yet</h3>
                <p style='color:#64748b;'>Please complete the scan first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        _, c, _ = st.columns([2, 2, 2])
        with c:
            if st.button("← Go to Scanner", key="back_to_scan"):
                go_to(2)
                st.rerun()
        return

    results = st.session_state.scan_results
    elapsed = st.session_state.scan_elapsed
    report_id = f"HT-{np.random.randint(10000, 99999)}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown("""
    <div class='slide-content anim-fadeup'>
        <div class='slide-header'>
            <div class='slide-tag'>Step 3 of 3 — Diagnostic Complete</div>
            <h1 class='slide-title'>Clinical Diagnostic Report</h1>
            <p class='slide-desc'>AI analysis complete. Review the blood group classification and donor/recipient compatibility below.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 4rem;'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='step-bar' style='margin-bottom:2rem;'>
        <div class='step-item done'></div>
        <div class='step-item done'></div>
        <div class='step-item done'></div>
    </div>
    """, unsafe_allow_html=True)

    # Report meta row
    st.markdown(f"""
    <div style='display:flex;gap:2rem;margin-bottom:2rem;flex-wrap:wrap;'>
        <div><span style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Report ID</span><br>
             <span style='font-size:0.9rem;font-weight:700;color:#f1f5f9;font-family:monospace;'>{report_id}</span></div>
        <div><span style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Timestamp</span><br>
             <span style='font-size:0.9rem;font-weight:600;color:#94a3b8;'>{timestamp}</span></div>
        <div><span style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Inference Time</span><br>
             <span style='font-size:0.9rem;font-weight:700;color:#5eead4;'>{elapsed:.2f}s</span></div>
        <div><span style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;'>Models Run</span><br>
             <span style='font-size:0.9rem;font-weight:700;color:#93c5fd;'>{len(results)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Results columns
    result_cols = st.columns(len(results), gap="large")

    for col, (variant_name, result) in zip(result_cols, results.items()):
        predicted = result['predicted_group']
        confidence = result['confidence']
        all_scores = result['all_scores']
        arch_label = variant_name.replace("efficientnet_", "EfficientNet-").upper().replace("_B", "-B")

        with col:
            # Main result card
            st.markdown(f"""
            <div class='result-card' style='margin-bottom:1.5rem;'>
                <div style='margin-bottom:0.5rem;'>
                    <span style='font-size:0.7rem;font-weight:700;color:#fda4af;background:rgba(225,29,72,0.1);
                          border:1px solid rgba(225,29,72,0.2);padding:3px 10px;border-radius:10px;
                          letter-spacing:1px;text-transform:uppercase;'>{arch_label}</span>
                </div>
                <div style='margin:0.75rem 0 0.25rem;'>
                    <div class='result-meta-label'>Detected Blood Group</div>
                    <div class='result-blood-type'>{predicted}</div>
                </div>
                <div style='display:flex;gap:2rem;margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);'>
                    <div>
                        <div class='result-meta-label'>Confidence</div>
                        <div style='font-size:1.4rem;font-weight:800;color:#5eead4;'>{confidence*100:.1f}%</div>
                    </div>
                    <div>
                        <div class='result-meta-label'>Rh Factor</div>
                        <div style='font-size:1.4rem;font-weight:800;color:#93c5fd;'>{'Positive' if '+' in predicted else 'Negative'}</div>
                    </div>
                    <div>
                        <div class='result-meta-label'>ABO Group</div>
                        <div style='font-size:1.4rem;font-weight:800;color:#fbbf24;'>{predicted.replace('+','').replace('-','')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown("<div class='card-sm' style='margin-bottom:1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;'>Confidence Distribution</p>", unsafe_allow_html=True)

            bars_html = ""
            for s in all_scores:
                pct = s['confidence'] * 100
                is_top = s['group'] == predicted
                fill = "linear-gradient(90deg,#e11d48,#f43f5e)" if is_top else "linear-gradient(90deg,#1e3a5f,#2563eb)"
                grp_cls = "top" if is_top else ""
                bars_html += f"""
                <div class='conf-row'>
                    <span class='conf-group {grp_cls}'>{s['group']}</span>
                    <div class='conf-track'>
                        <div class='conf-fill' style='width:{min(pct,100):.1f}%;background:{fill};'></div>
                    </div>
                    <span class='conf-pct {grp_cls}'>{pct:.1f}%</span>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Compatibility summary
            donate = DONATE_TO.get(predicted, [])
            receive = RECEIVE_FROM.get(predicted, [])

            st.markdown(f"""
            <div class='card-sm' style='margin-bottom:1.5rem;'>
                <p style='font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;'>Blood Compatibility</p>
                <div style='margin-bottom:0.5rem;'>
                    <span style='font-size:0.78rem;color:#5eead4;font-weight:600;'>💉 Can Donate To: </span>
                    <span style='font-size:0.85rem;color:#f1f5f9;font-weight:700;'>{', '.join(donate)}</span>
                </div>
                <div>
                    <span style='font-size:0.78rem;color:#93c5fd;font-weight:600;'>🩸 Can Receive From: </span>
                    <span style='font-size:0.85rem;color:#f1f5f9;font-weight:700;'>{', '.join(receive)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Fact box
            st.markdown(f"""
            <div style='background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem;'>
                <span style='font-size:0.8rem;color:#fbbf24;line-height:1.5;'>💡 {BG_FACTS.get(predicted,"")}</span>
            </div>
            """, unsafe_allow_html=True)

            # Export
            export_data = {
                "report_id": report_id,
                "timestamp": timestamp,
                "variant": variant_name,
                "predicted_blood_group": predicted,
                "confidence_score": float(confidence),
                "processing_time_seconds": round(elapsed, 3),
                "can_donate_to": donate,
                "can_receive_from": receive,
                "blood_group_fact": BG_FACTS.get(predicted, ""),
                "all_scores": {s['group']: round(s['confidence'] * 100, 2) for s in all_scores},
            }
            st.download_button(
                "📄 Download JSON Report",
                json.dumps(export_data, indent=4),
                file_name=f"hematype_{variant_name}_{report_id}.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_{variant_name}"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # New scan button
    st.markdown("<div style='padding:1.5rem 4rem;'>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([3, 2, 3])
    with col_btn:
        st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
        if st.button("🔄 Start New Patient Scan", key="new_scan"):
            st.session_state.image_source = None
            st.session_state.scan_results = None
            go_to(1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def slide_compatibility():
    st.markdown("""
    <div class='slide-content anim-fadeup'>
        <div class='slide-header'>
            <div class='slide-tag'>Blood Bank Reference</div>
            <h1 class='slide-title'>Compatibility Checker</h1>
            <p class='slide-desc'>Find compatible donors and recipients for any blood type. Full ABO/Rh compatibility matrix included.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 4rem;'>", unsafe_allow_html=True)

    sel_bg = st.selectbox("Select Blood Group to Explore", BLOOD_GROUPS, key="compat_sel")

    col_donate, col_receive = st.columns(2, gap="large")

    with col_donate:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='color:#5eead4;font-size:1rem;font-weight:700;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:0.5rem;'>
            💉 Can Donate To
        </h3>
        """, unsafe_allow_html=True)
        pills_html = ""
        for g in DONATE_TO.get(sel_bg, []):
            pills_html += f"""
            <div class='pill pill-donate'>
                <span class='pill-type'>{g}</span>
                <span class='pill-info'>{BLOOD_GROUP_INFO.get(g,'')[:80]}...</span>
            </div>"""
        st.markdown(pills_html or "<p style='color:#64748b;'>No donation targets.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_receive:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='color:#93c5fd;font-size:1rem;font-weight:700;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:0.5rem;'>
            🩸 Can Receive From
        </h3>
        """, unsafe_allow_html=True)
        pills_html = ""
        for g in RECEIVE_FROM.get(sel_bg, []):
            pills_html += f"""
            <div class='pill pill-receive'>
                <span class='pill-type'>{g}</span>
                <span class='pill-info'>{BLOOD_GROUP_INFO.get(g,'')[:80]}...</span>
            </div>"""
        st.markdown(pills_html or "<p style='color:#64748b;'>No receive sources.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Full compatibility matrix
    st.markdown("<div style='margin-top:2rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#f1f5f9;font-size:1rem;font-weight:700;margin-bottom:1rem;'>Full Compatibility Matrix</h3>", unsafe_allow_html=True)

    matrix_html = "<div class='compat-matrix'>"
    matrix_html += "<div class='cm-hdr'>Donor ↓ / Recipient →</div>"
    for rcp in BLOOD_GROUPS:
        matrix_html += f"<div class='cm-hdr'>{rcp}</div>"
    for donor in BLOOD_GROUPS:
        matrix_html += f"<div class='cm-row-hdr'>{donor}</div>"
        for rcp in BLOOD_GROUPS:
            can = rcp in DONATE_TO.get(donor, [])
            cls = "cm-yes" if can else "cm-no"
            icon = "✓" if can else "·"
            matrix_html += f"<div class='cm-cell {cls}'>{icon}</div>"
    matrix_html += "</div>"

    st.markdown(matrix_html, unsafe_allow_html=True)

    # Blood group fact
    st.markdown(f"""
    <div style='margin-top:1.5rem;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);
                border-radius:12px;padding:1.25rem 1.5rem;'>
        <span style='font-size:0.9rem;color:#fbbf24;font-weight:600;'>💡 About {sel_bg}: </span>
        <span style='font-size:0.9rem;color:#94a3b8;'>{BG_FACTS.get(sel_bg,'')}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)


def slide_architecture():
    st.markdown("""
    <div class='slide-content anim-fadeup'>
        <div class='slide-header'>
            <div class='slide-tag'>Technical Overview</div>
            <h1 class='slide-title'>System Architecture</h1>
            <p class='slide-desc'>Dual EfficientNet backbone pipeline — from raw fingerprint to blood group prediction.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 4rem;'>", unsafe_allow_html=True)

    col_flow, col_tech = st.columns([2, 3], gap="large")

    with col_flow:
        st.markdown("<h3 style='color:#f1f5f9;font-size:0.9rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;'>Neural Network Pipeline</h3>", unsafe_allow_html=True)

        nodes = [
            ("📤", "Fingerprint Input", "High-resolution BMP/PNG/JPG up to 300×300 px"),
            ("🔄", "Clinical Preprocessing", "Grayscale → CLAHE contrast → ImageNet normalization", None),
            ("📱", "EfficientNet Backbone", "MBConv blocks with squeeze-excitation. Dual paths: B3 (1536-ch) and B0 (1280-ch)", "GAP: 1536 ch (B3) / 1280 ch (B0)"),
            ("🧠", "Classification Head", "Aggressive dropout regularization preventing overfitting", "Dropout(0.35)→FC(512)→ReLU→Dropout(0.20)→FC(8)"),
            ("🩸", "Blood Group Output", "Softmax over 8 ABO/Rh classes. A+, A-, B+, B-, AB+, AB-, O+, O-", None),
        ]
        flow_html = ""
        for i, node in enumerate(nodes):
            icon, title, desc = node[0], node[1], node[2]
            badge = node[3] if len(node) > 3 else None
            badge_html = f"<div class='flow-badge'>{badge}</div>" if badge else ""
            flow_html += f"<div class='flow-node'><div class='flow-icon'>{icon}</div><div class='flow-text'><h4 style='margin:0;margin-bottom:0.2rem;'>{title}</h4><p style='margin:0;'>{desc}</p>{badge_html}</div></div>"
            if i < len(nodes) - 1:
                flow_html += "<div class='flow-arrow'>⬇</div>"
        st.markdown(flow_html, unsafe_allow_html=True)

    with col_tech:
        st.markdown("<h3 style='color:#f1f5f9;font-size:0.9rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;'>Technology Stack</h3>", unsafe_allow_html=True)

        tech_cards = [
            ("Framework", "🔥 PyTorch 2.5 + CUDA 12.1", "Fully vectorized tensor ops, native AMP mixed precision, and AdamW optimization."),
            ("Backbones", "📱 EfficientNet B3 & B0", "B3: ~12.2M params (300px), B0: ~5.3M params (224px). Both pretrained on ImageNet-1k."),
            ("Training", "🔄 Data Augmentation", "MixUp (α=0.2), RandomPerspective, ColorJitter, HorizontalFlip for scan variance tolerance."),
            ("Optimization", "📈 Cosine Annealing", "Warm restarts to escape local loss minima across 50-epoch training cycles."),
            ("Regularization", "🎛 Multi-Stage Dropout", "0.35 and 0.20 dropout layers between fully-connected nodes to prevent memorization."),
            ("Inference", "⚡ FP16 Autocast", "Native torch.amp autocast on CUDA for 2× memory efficiency and throughput boost."),
        ]

        rows = [tech_cards[:3], tech_cards[3:]]
        for row in rows:
            cols = st.columns(3, gap="small")
            for col, (cat, title, desc) in zip(cols, row):
                with col:
                    st.markdown(f"<div class='tech-card' style='margin-bottom:0.75rem;'><div class='tech-cat'>{cat}</div><div class='tech-title'>{title}</div><div class='tech-desc'>{desc}</div></div>", unsafe_allow_html=True)

        # Model performance stats
        st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#f1f5f9;font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;'>Active Models</h4>", unsafe_allow_html=True)
        variants = _get_available_cnn_variants()
        for v in variants:
            params = "12.2M" if "b3" in v else "5.3M"
            st.markdown(f"<div style='background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.15);border-radius:12px;padding:0.85rem 1.25rem;margin-bottom:0.5rem;display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:space-between;align-items:center;'><span style='font-weight:700;color:#93c5fd;'>{v.replace('efficientnet_','EfficientNet-').upper()}</span><span style='font-size:0.75rem;color:#64748b;'>Parameters: <strong style='color:#94a3b8;'>{params}</strong></span><span style='font-size:0.7rem;background:rgba(13,148,136,0.15);color:#5eead4;padding:3px 10px;border-radius:8px;border:1px solid rgba(13,148,136,0.3);'>✓ Loaded</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def slide_about():
    st.markdown("""
    <div class='slide-content anim-fadeup'>
        <div class='slide-header'>
            <div class='slide-tag'>About This Project</div>
            <h1 class='slide-title'>HemaType AI v4.0</h1>
            <p class='slide-desc'>The future of non-invasive, needle-free blood group diagnostics powered by extreme-scale convolutional neural networks.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 4rem;'>", unsafe_allow_html=True)

    # Stats row
    stats = [
        ("0", "Needles Required"),
        ("8", "Blood Groups"),
        ("<1s", "Inference Time"),
        ("12.2M", "Max Parameters"),
        ("100%", "Local Processing"),
        ("MIT", "License"),
    ]
    stat_cols = st.columns(6, gap="small")
    for col, (val, lbl) in zip(stat_cols, stats):
        with col:
            st.markdown(f"""
            <div class='about-stat'>
                <div class='about-stat-val'>{val}</div>
                <div class='about-stat-lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    col_prob, col_feat = st.columns(2, gap="large")

    with col_prob:
        st.markdown("""
        <div class='card'>
            <h3 style='color:#f1f5f9;font-size:1rem;font-weight:700;margin-bottom:1.2rem;
                        border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:0.75rem;'>
                🏥 The Clinical Problem
            </h3>
            <p style='color:#64748b;line-height:1.7;font-size:0.9rem;margin-bottom:1rem;'>
                Traditional blood typing requires invasive phlebotomy — drawing blood with needles, biochemical reagents, and laboratory processing that takes time and causes patient discomfort.
            </p>
            <p style='color:#64748b;line-height:1.7;font-size:0.9rem;'>
                <strong style='color:#94a3b8;'>HemaType AI</strong> bypasses the needle entirely by identifying
                subtle, deep-dermal topographical correlations between fingerprint ridge patterns and ABO/Rh
                gene expression using extreme-scale CNNs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat:
        features = [
            ("#5eead4", "Zero Biomaterial", "No chemical reagents or blood samples required whatsoever."),
            ("#93c5fd", "Real-Time Inference", "Sub-second AI classification using PyTorch AMP acceleration."),
            ("#fda4af", "Clinical Precision", "Dual EfficientNet backbones trained on augmented clinical sweeps."),
            ("#fbbf24", "Privacy First", "100% local execution — no patient data ever leaves the device."),
        ]
        feat_html = "<div class='card'><h3 style='color:#f1f5f9;font-size:1rem;font-weight:700;margin-bottom:1.2rem;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:0.75rem;'>✨ Core Features</h3>"
        for color, title, desc in features:
            feat_html += f"""
            <div style='display:flex;gap:0.75rem;margin-bottom:1rem;align-items:flex-start;'>
                <div style='width:3px;background:{color};border-radius:2px;min-height:36px;flex-shrink:0;margin-top:2px;'></div>
                <div>
                    <div style='font-size:0.9rem;font-weight:700;color:{color};margin-bottom:0.2rem;'>{title}</div>
                    <div style='font-size:0.82rem;color:#64748b;line-height:1.5;'>{desc}</div>
                </div>
            </div>"""
        feat_html += "</div>"
        st.markdown(feat_html, unsafe_allow_html=True)

    # References
    st.markdown("""
    <div style='margin-top:2rem;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                border-radius:16px;padding:1.5rem 2rem;'>
        <h4 style='color:#f1f5f9;font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;'>📚 References</h4>
        <div style='font-size:0.82rem;color:#64748b;line-height:1.8;'>
            • Tan, M. &amp; Le, Q. (2019). <em>EfficientNet: Rethinking Model Scaling for CNNs.</em> ICML.<br>
            • Kaggle — Fingerprint-Based Blood Group Detection Dataset.<br>
            • PyTorch Transfer Learning Tutorial — pytorch.org/tutorials.
        </div>
        <div style='margin-top:1.5rem;text-align:center;font-size:0.75rem;color:#374151;font-family:monospace;'>
            © 2026 HemaType AI Research Group • MIT License • Built with deeply layered neural abstraction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
cur = st.session_state.slide

SLIDES = [
    slide_hero,
    slide_upload,
    slide_scan,
    slide_report,
    slide_compatibility,
    slide_architecture,
    slide_about,
]

SLIDES[cur]()
render_nav()
