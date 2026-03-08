import os
import builtins
import tempfile
import streamlit as st
import shutil
import cv2

# ==============================================================================
# --- CRITICAL: GLOBAL MONKEYPATCH FOR STREAMLIT CLOUD PERMISSIONS ---
# ==============================================================================
WRITABLE_BASE = os.path.join(tempfile.gettempdir(), "slt_persistent_storage")
os.makedirs(WRITABLE_BASE, exist_ok=True)

_orig_makedirs = os.makedirs
_orig_mkdir = os.mkdir
_orig_open = builtins.open
_orig_rename = os.rename
_orig_replace = os.replace
_orig_exists = os.path.exists
_orig_isfile = os.path.isfile
_orig_listdir = os.listdir

def _get_shadow_path(path):
    if not path: return path
    p = str(path).replace("\\", "/")
    if "site-packages/sign_language_translator" in p:
        parts = p.split("site-packages/sign_language_translator/")
        rel = parts[1] if len(parts) > 1 else ""
        return os.path.join(WRITABLE_BASE, rel)
    return None

def _redirect_read_write(path, is_write=False):
    shadow = _get_shadow_path(path)
    if shadow:
        if is_write or _orig_exists(shadow):
            parent = os.path.dirname(shadow)
            if parent and not _orig_exists(parent):
                _orig_makedirs(parent, exist_ok=True)
            return shadow
    return path

def _patched_makedirs(name, mode=0o777, exist_ok=False):
    return _orig_makedirs(_redirect_read_write(name, is_write=True), mode, exist_ok)

def _patched_mkdir(path, mode=0o777, *args, **kwargs):
    return _orig_mkdir(_redirect_read_write(path, is_write=True), mode, *args, **kwargs)

def _patched_open(file, *args, **kwargs):
    mode = args[0] if args else kwargs.get('mode', 'r')
    is_write = any(m in mode for m in ('w', 'a', '+', 'x'))
    return _orig_open(_redirect_read_write(file, is_write=is_write), *args, **kwargs)

def _patched_rename(src, dst, *args, **kwargs):
    return _orig_rename(_redirect_read_write(src), _redirect_read_write(dst, is_write=True), *args, **kwargs)

def _patched_replace(src, dst, *args, **kwargs):
    return _orig_replace(_redirect_read_write(src), _redirect_read_write(dst, is_write=True), *args, **kwargs)

def _patched_exists(path):
    shadow = _get_shadow_path(path)
    if shadow and _orig_exists(shadow): return True
    return _orig_exists(path)

def _patched_isfile(path):
    shadow = _get_shadow_path(path)
    if shadow and _orig_isfile(shadow): return True
    return _orig_isfile(path)

def _patched_listdir(path):
    return _orig_listdir(_redirect_read_write(path))

# Apply global patches
os.makedirs = _patched_makedirs
os.mkdir = _patched_mkdir
builtins.open = _patched_open
os.rename = _patched_rename
os.replace = _patched_replace
os.path.exists = _patched_exists
os.path.isfile = _patched_isfile
os.listdir = _patched_listdir

# --- ENCODING FIX: Standard Linux/Streamlit lack H.264 encoder ---
import subprocess

def _optimize_video_for_web(path):
    """Converts a video to H.264 using ffmpeg to ensure browser compatibility."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    
    # We use a temp file for output
    temp_out = path + ".optimized.mp4"
    try:
        # ffmpeg command to convert to H.264 (libx264)
        # -y: overwrite
        # -preset ultrafast: for speed
        # -vcodec libx264: the standard for web
        cmd = [
            "ffmpeg", "-y", "-i", path, 
            "-vcodec", "libx264", 
            "-pix_fmt", "yuv420p", 
            "-preset", "ultrafast", 
            temp_out
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
            os.replace(temp_out, path)
            print(f"🎬 Video Optimized: {path}")
    except Exception as e:
        print(f"⚠️ Video Optimization Failed for {path}: {e}")
        if os.path.exists(temp_out): os.remove(temp_out)

class _VideoWriterWrapper:
    def __init__(self, writer, filename):
        self.writer = writer
        self.filename = filename
    def write(self, frame):
        self.writer.write(frame)
    def release(self):
        self.writer.release()
        _optimize_video_for_web(self.filename)
    def __getattr__(self, name):
        return getattr(self.writer, name)

_orig_VideoWriter = cv2.VideoWriter
def _patched_VideoWriter(filename, fourcc, *args, **kwargs):
    # Force mp4v for high-compatibility saving on Linux
    # Then we optimize it to H.264 on release
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = _orig_VideoWriter(filename, fourcc, *args, **kwargs)
    return _VideoWriterWrapper(writer, filename)

cv2.VideoWriter = _patched_VideoWriter
# -----------------------------------------------------------------

# Initialize Assets ROOT_DIR immediately
try:
    import sign_language_translator as slt
    slt.Assets.ROOT_DIR = WRITABLE_BASE
except:
    pass
# ==============================================================================

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
    # Explicitly ensure the library uses our writable path (Shadow Persistence)
    slt.Assets.ROOT_DIR = WRITABLE_BASE
    # Sync core data directory to the same base to avoid redundant downloads
    core_data_dir = os.path.join(WRITABLE_BASE, "app_internal_data")
    os.makedirs(core_data_dir, exist_ok=True)
    
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
    st.info("🧠 Training CLR Core Brain (v2 Matrix)...")
    if core.train_core():
        print("🚀 SLT Core Brain Training Complete!")
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
                
                status_placeholder = st.empty() # Added for dynamic status updates
                for w in words:
                    if w in PSL_VOCABULARY:
                        try:
                            status_placeholder.info(f"🔍 Processing: **{w}**") # Changed to use placeholder
                            clip = translator.translate(PSL_VOCABULARY[w])
                            v_clips.append(clip)
                            dna = core.get_word_dna(w)
                            if dna is not None:
                                dna_list.append(dna)
                                status_placeholder.success(f"✅ DNA for **{w}** ready.") # Changed to use placeholder
                            else:
                                status_placeholder.warning(f"⚠️ DNA for **{w}** missing in dictionary.") # Changed to use placeholder
                        except Exception as e:
                            status_placeholder.error(f"❌ Error synthesizing **{w}**: {e}") # Changed to use placeholder
                    else:
                        status_placeholder.warning(f"📖 Word **{w}** not in vocabulary.") # Changed to use placeholder
                
                if v_clips:
                    full_dna = renderer.stitch_landmarks(dna_list)
                    
                    col_orig, col_av, col_neo = st.columns(3)
                    with col_orig:
                        st.markdown("### 📽️ Source Benchmark")
                        f_orig = v_clips[0]
                        for c in v_clips[1:]: f_orig = f_orig + c
                        p_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        f_orig.save(p_orig, overwrite=True)
                        st.video(p_orig)
                        st.caption("Standard library stitching.")
                        
                    with col_av:
                        st.markdown("### 🤖 Seamless Skeletal")
                        if full_dna is not None:
                            out_p = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                            renderer.render_landmark_dna(full_dna, out_p)
                            st.video(out_p)
                            st.caption("Internal 'Takhyeet' engine.")
                            
                    with col_neo:
                        st.markdown("### 🦾 Neo-Avatar 3D")
                        if full_dna is not None:
                            out_neo = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                            renderer.render_neo_avatar(full_dna, out_neo)
                            st.video(out_neo)
                            st.caption("Premium volumetric animation.")
                            
                    st.success("✅ Multi-Phase Synthesis Complete")
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
