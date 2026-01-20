"""
🔬 Sign Language Translator - Streamlit Web App (Self-Contained)
Bidirectional: Text → Video (Generation) | Video → Text (Recognition)
Auto-trains classifier on first run - no manual setup required!
"""

import streamlit as st
import os
import cv2
import glob
import numpy as np
import pickle
import tempfile
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Digital Human SLT",
    page_icon="🤖",
    layout="wide"
)

# Premium UI Styling
st.markdown("""
<style>
    .main { background-color: #0f172a; color: white; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; }
    h1, h2, h3 { color: #38bdf8 !important; }
    div[data-testid="stExpander"] { background-color: rgba(56, 189, 248, 0.05); border: 1px solid #38bdf8; border-radius: 10px; }
    .stTextInput>div>div>input { background-color: #1e293b; color: white; border: 1px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ===================== VOCABULARY =====================
PSL_VOCABULARY = {
    "apple": "سیب",
    "world": "دنیا",
    "pakistan": "پاکستان",
    "good": "اچھا",
    "red": "لال",
    "is": "ہے",
    "the": "یہ",
    "that": "وہ"
}

# --- Persistence Paths (Streamlit Cloud Fix) ---
WRITABLE_ROOT = os.path.join(tempfile.gettempdir(), "slt_app_data")
os.makedirs(WRITABLE_ROOT, exist_ok=True)

DATA_DIR = os.path.join(WRITABLE_ROOT, "psl_cv_assets")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
MODEL_PATH = os.path.join(DATA_DIR, "psl_classifier.pkl")
# -----------------------------------------------

# ===================== CORE INITIALIZATION =====================
from sign_language_core import SignLanguageCore, DigitalHumanRenderer

@st.cache_resource
def get_slt_core():
    core = SignLanguageCore(data_dir=DATA_DIR)
    core.load_core()
    return core

@st.cache_resource
def get_avatar_renderer():
    return DigitalHumanRenderer()

@st.cache_resource
def load_slt_engine():
    import sign_language_translator as slt
    import builtins
    
    # --- PRO-ACTIVE PERSISTENCE FIX ---
    assets_dir = os.path.join(tempfile.gettempdir(), "slt_assets_v2")
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Direct path redirection
    slt.Assets.ROOT_DIR = assets_dir
    
    # 2. Aggressive Monkeypatching
    original_open = builtins.open
    original_makedirs = os.makedirs
    original_mkdir = os.mkdir

    def is_protected(path):
        return "site-packages/sign_language_translator" in str(path).replace("\\", "/")

    def patched_makedirs(name, mode=0o777, exist_ok=False):
        if is_protected(name):
            name = os.path.join(assets_dir, os.path.basename(name))
        return original_makedirs(name, mode, exist_ok)

    def patched_mkdir(path, mode=0o777, *args, **kwargs):
        if is_protected(path):
            path = os.path.join(assets_dir, os.path.basename(path))
        return original_mkdir(path, mode, *args, **kwargs)

    def patched_open(file, *args, **kwargs):
        if is_protected(file):
            mode = args[0] if args else kwargs.get('mode', 'r')
            if 'w' in mode or 'a' in mode or '+' in mode:
                file = os.path.join(assets_dir, os.path.basename(str(file)))
        return original_open(file, *args, **kwargs)

    # Apply patches
    builtins.open = patched_open
    os.makedirs = patched_makedirs
    os.mkdir = patched_mkdir
    # ---------------------------------------

    translator = slt.models.ConcatenativeSynthesis(
        text_language="urdu",
        sign_language="psl",
        sign_format="vid"
    )
    return translator, slt

def load_or_train_core(core, translator):
    """Load core model or build dictionary and train"""
    if core.classifier: return True
    
    st.info("🔧 Building Next-Gen Landmark Dictionary (First run)...")
    core.build_landmark_dictionary(translator)
    if core.train_core():
        st.success("✅ Core Model trained successfully.")
        return True
    return False

def video_to_text(video_path, core):
    """Recognize sign from video using Unified Core"""
    return core.predict_sign(video_path)

# ===================== STREAMLIT UI =====================
def main():
    # Header
    st.title("🤟 Sign Language Translator")
    st.markdown("**Bidirectional Translation:** Text ↔ Pakistani Sign Language (PSL)")
    st.markdown("---")
    
    # Architecture Explanation
    with st.expander("📚 System Architecture: Unified Data Representation"):
        st.markdown("""
        The system relies on a **Common Landmark Benchmark**:
        1.  **Text → Video:** Maps text to the Benchmark Dictionary to retrieve **Digital Human Ready** representations.
        2.  **Video → Text:** Extracts landmarks and compares them against the **Same Benchmark Benchmark** for recognition.
        
        *Unified Core for Bi-directional Translation.*
        """)
    
    # Load engines
    with st.spinner("⏳ Loading SLT Core & Avatar Engine..."):
        translator, slt = load_slt_engine()
        core = get_slt_core()
        renderer = get_avatar_renderer()
    
    # Load or train core (AUTO)
    if not load_or_train_core(core, translator):
        st.error("❌ Failed to initialize SLT Core.")
        st.stop()
    
    st.success(f"✅ System Ready | Vocabulary: {list(core.landmark_dict.keys() if core.landmark_dict else core.label_encoder.classes_)}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📝 Text → Video", "🎥 Video → Text"])
    
    # ==================== TAB 1: TEXT TO VIDEO ====================
    with tab1:
        st.header("📝 Text to Sign Language Video")
        st.info(f"**Available words:** {', '.join(PSL_VOCABULARY.keys())}")
        
        text_input = st.text_input(
            "Enter text or use Voice Input:",
            placeholder="e.g., apple good world"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            gen_btn = st.button("🚀 Generate Digital Human Output", key="gen")
        with col2:
            if st.button("🎤 Use Voice Input", key="voice"):
                with st.spinner("🎙️ Listening..."):
                    voice_text = core.speech_to_text()
                    if voice_text:
                        st.session_state.voice_input = voice_text
                        st.experimental_rerun()
        
        if 'voice_input' in st.session_state:
            text_input = st.session_state.voice_input
            st.info(f"🎤 Heard: **{text_input}**")
            del st.session_state.voice_input

        if gen_btn or (text_input and 'gen' in st.session_state):
            if text_input:
                with st.spinner("🧪 Transforming Benchmark Person into Digital Avatar..."):
                    words = text_input.lower().split()
                    v_clips = []
                    dna_list = []
                    
                    for w in words:
                        if w in PSL_VOCABULARY:
                            try:
                                clip = translator.translate(PSL_VOCABULARY[w])
                                v_clips.append(clip)
                                
                                # Collect DNA for seamless stitching
                                dna = core.get_word_dna(w)
                                if dna is not None:
                                    dna_list.append(dna)
                            except: pass
                    
                    if v_clips:
                        col_orig, col_av = st.columns(2)
                        
                        with col_orig:
                            st.markdown("### 📽️ Source Benchmark")
                            st.caption("The original person detected in the system.")
                            f_orig = v_clips[0]
                            for c in v_clips[1:]: f_orig = f_orig + c
                            p_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                            f_orig.save(p_orig, overwrite=True)
                            st.video(p_orig)
                            
                        with col_av:
                            st.markdown("### 🤖 Seamless Digital Avatar")
                            st.caption("Now with **Facial Intelligence (Non-Manual Signals)**")
                            if dna_list:
                                out_p = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                                renderer.stitch_and_render(dna_list, out_p)
                                st.video(out_p)
                        
                        st.markdown("---")
                        st.success("✅ Seamless Digital Human Performance Complete")
                    else:
                        st.error("❌ Word not in Benchmark.")
            else:
                st.warning("⚠️ Please provide input.")
    
    # ==================== TAB 2: VIDEO TO TEXT ====================
    with tab2:
        st.header("🎥 Sign Language Video to Text")
        
        uploaded_file = st.file_uploader(
            "Upload a sign language video (.mp4)",
            type=["mp4", "avi", "mov"]
        )
        
        if uploaded_file:
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            st.video(temp_path)
            
            if st.button("🔍 Recognize Sign", key="rec"):
                with st.spinner("Analyzing with Unified Core Landmarks..."):
                    label, confidence = video_to_text(temp_path, core)
                    
                    if label:
                        color = "#22c55e" if confidence > 70 else "#f59e0b"
                        st.markdown(f"""
                        <div style='border: 3px solid {color}; border-radius: 15px; padding: 20px; text-align: center; background-color: #f9fafb;'>
                            <h1 style='color: {color};'>🏆 {label}</h1>
                            <div style='background: #e5e7eb; border-radius: 10px; height: 30px; margin: 10px 0;'>
                                <div style='width: {confidence}%; background: {color}; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                                    {confidence:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Could not detect landmarks.")
    
    # Footer
    st.markdown("---")
    st.markdown("Designed by **Ahmed Eltaweel** | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    main()
