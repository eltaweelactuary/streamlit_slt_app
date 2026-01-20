import os
import builtins
import tempfile
import streamlit as st
import shutil

# ==============================================================================
# --- CRITICAL: GLOBAL MONKEYPATCH FOR STREAMLIT CLOUD PERMISSIONS ---
# ==============================================================================
# We must redirect any write operations targeting the site-packages directory
# to a writable temporary directory. Streamlit Cloud's env is read-only for venv.

WRITABLE_BASE = os.path.join(tempfile.gettempdir(), "slt_persistent_storage")
os.makedirs(WRITABLE_BASE, exist_ok=True)

_orig_makedirs = os.makedirs
_orig_mkdir = os.mkdir
_orig_open = builtins.open

def _redirect_if_needed(path):
    if not path: return path
    p = str(path).replace("\\", "/")
    # Detect attempts to write to the sign_language_translator package directory
    if "site-packages/sign_language_translator" in p:
        # Extract relative path to maintain structure
        parts = p.split("site-packages/sign_language_translator/")
        rel = parts[1] if len(parts) > 1 else "root"
        new_path = os.path.join(WRITABLE_BASE, rel)
        # Ensure parent dir exists in the new location
        _orig_makedirs(os.path.dirname(new_path), exist_ok=True)
        return new_path
    return path

def _patched_makedirs(name, mode=0o777, exist_ok=False):
    return _orig_makedirs(_redirect_if_needed(name), mode, exist_ok)

def _patched_mkdir(path, mode=0o777, *args, **kwargs):
    return _orig_mkdir(_redirect_if_needed(path), mode, *args, **kwargs)

def _patched_open(file, *args, **kwargs):
    mode = args[0] if args else kwargs.get('mode', 'r')
    # Only redirect if it's a 'write' mode
    if any(m in mode for m in ('w', 'a', '+', 'x')):
        file = _redirect_if_needed(file)
    return _orig_open(file, *args, **kwargs)

# Apply global patches BEFORE any other imports
os.makedirs = _patched_makedirs
os.mkdir = _patched_mkdir
builtins.open = _patched_open

# Pre-empitvely set ROOT_DIR for the Assets class if possible
# (This helps even if the monkeypatch misses something)
try:
    import sign_language_translator as slt
    slt.Assets.ROOT_DIR = WRITABLE_BASE
except:
    pass
# ==============================================================================

import cv2
import numpy as np
import pickle
from pathlib import Path
from sign_language_core import SignLanguageCore, DigitalHumanRenderer

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

# Vocabulary Mapping (Simplified for UI display)
PSL_VOCABULARY = {
    "apple": "سیب", "world": "دنیا", "pakistan": "پاکستان",
    "good": "اچھا", "red": "لال", "is": "ہے", "the": "یہ", "that": "وہ"
}

# App Data Paths
DATA_DIR = os.path.join(WRITABLE_BASE, "app_internal_data")
os.makedirs(DATA_DIR, exist_ok=True)

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
    # Explicitly ensure the library uses our writable path
    slt.Assets.ROOT_DIR = WRITABLE_BASE
    
    translator = slt.models.ConcatenativeSynthesis(
        text_language="urdu",
        sign_language="psl",
        sign_format="vid"
    )
    return translator, slt

def load_or_train_core(core, translator):
    if core.classifier: return True
    st.info("🔧 Building Next-Gen Landmark Dictionary (First run)...")
    core.build_landmark_dictionary(translator)
    if core.train_core():
        st.success("✅ Core Model trained successfully.")
        return True
    return False

def main():
    st.title("🤟 Sign Language Translator")
    st.markdown("**Bidirectional Translation:** Text ↔ Pakistani Sign Language (PSL)")
    st.markdown("---")
    
    # Architecture Explanation
    with st.expander("📚 System Architecture: Unified Data Representation"):
        st.markdown("""
        The system relies on a **Common Landmark Benchmark**:
        1. **Text → Video:** Maps text to the Benchmark Dictionary.
        2. **Video → Text:** Extracts landmarks and compares them against the same Benchmark.
        """)
    
    with st.spinner("⏳ Loading SLT Core & Avatar Engine..."):
        translator, slt = load_slt_engine()
        core = get_slt_core()
        renderer = get_avatar_renderer()
    
    if not load_or_train_core(core, translator):
        st.error("❌ Failed to initialize SLT Core.")
        st.stop()
    
    st.success(f"✅ System Ready | Vocabulary Size: {len(core.landmark_dict if core.landmark_dict else [])}")
    
    tab1, tab2 = st.tabs(["📝 Text → Video", "🎥 Video → Text"])
    
    # TAB 1: TEXT TO VIDEO
    with tab1:
        st.header("📝 Text to Sign Language Video")
        st.info(f"**Available words:** {', '.join(PSL_VOCABULARY.keys())}")
        
        text_input = st.text_input("Enter text:", placeholder="e.g., apple good world")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            gen_btn = st.button("🚀 Generate Digital Human Output")
        with col2:
            if st.button("🎤 Use Voice Input"):
                with st.spinner("🎙️ Listening..."):
                    voice_text = core.speech_to_text()
                    if voice_text:
                        st.info(f"🎤 Heard: **{voice_text}**")
                        text_input = voice_text

        if gen_btn and text_input:
            with st.spinner("🧪 Transforming to Digital Avatar..."):
                words = text_input.lower().split()
                v_clips = []
                dna_list = []
                
                for w in words:
                    if w in PSL_VOCABULARY:
                        try:
                            clip = translator.translate(PSL_VOCABULARY[w])
                            v_clips.append(clip)
                            dna = core.get_word_dna(w)
                            if dna is not None: dna_list.append(dna)
                        except: pass
                
                if v_clips:
                    col_orig, col_av = st.columns(2)
                    with col_orig:
                        st.markdown("### 📽️ Source Benchmark")
                        f_orig = v_clips[0]
                        for c in v_clips[1:]: f_orig = f_orig + c
                        p_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        f_orig.save(p_orig, overwrite=True)
                        st.video(p_orig)
                        
                    with col_av:
                        st.markdown("### 🤖 Seamless Digital Avatar")
                        if dna_list:
                            out_p = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                            renderer.stitch_and_render(dna_list, out_p)
                            st.video(out_p)
                    st.success("✅ Performance Complete")
                else:
                    st.error("❌ Words not in Benchmark.")

    # TAB 2: VIDEO TO TEXT
    with tab2:
        st.header("🎥 Sign Language Video to Text")
        uploaded_file = st.file_uploader("Upload video (.mp4)", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            st.video(temp_path)
            
            if st.button("🔍 Recognize Sign"):
                with st.spinner("Analyzing landmarks..."):
                    label, confidence = core.predict_sign(temp_path)
                    if label:
                        st.success(f"🏆 Detected: {label} ({confidence:.1f}%)")
                    else:
                        st.error("❌ Detection failed.")

    st.markdown("---")
    st.markdown("Designed by **Ahmed Eltaweel** | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    main()
