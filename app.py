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
    page_title="SLT Translator",
    page_icon="🤟",
    layout="wide"
)

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

DATA_DIR = "./psl_cv_assets"
VIDEOS_DIR = "./psl_cv_assets/videos"
MODEL_PATH = "./psl_classifier.pkl"

# ===================== INITIALIZATION =====================
@st.cache_resource
def load_slt_engine():
    import sign_language_translator as slt
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    slt.Assets.set_root_dir(DATA_DIR)
    
    # Force download necessary assets for PSL
    with st.spinner("📥 Downloading sign language assets (first time only)..."):
        try:
            # Download psl videos and required models
            slt.Assets.download(".*psl.*")
        except Exception as e:
            st.warning(f"⚠️ Asset download notice: {e}")
    
    translator = slt.models.ConcatenativeSynthesis(
        text_language="urdu",
        sign_language="psl",
        sign_format="vid"
    )
    return translator, slt

@st.cache_resource
def load_mediapipe():
    """Load MediaPipe Holistic"""
    import mediapipe as mp
    return mp.solutions.holistic

def download_training_videos(translator, progress_bar):
    """Download all vocabulary videos for training"""
    total = len(PSL_VOCABULARY)
    downloaded = 0
    
    for i, (english_word, urdu_word) in enumerate(PSL_VOCABULARY.items()):
        try:
            sign_video = translator.translate(urdu_word)
            if len(sign_video) > 0:
                downloaded += 1
        except:
            pass
        progress_bar.progress((i + 1) / total, f"Downloading: {english_word}")
    
    return downloaded

def extract_landmarks(video_path, mp_holistic, max_frames=30):
    """Extract normalized landmarks from video"""
    cap = cv2.VideoCapture(video_path)
    all_features = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            frame_features = []
            
            # Reference point (nose)
            ref_x, ref_y, ref_z = 0, 0, 0
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[0]
                ref_x, ref_y, ref_z = nose.x, nose.y, nose.z
            
            def normalize(lm):
                return [lm.x - ref_x, lm.y - ref_y, lm.z - ref_z]
            
            # Left Hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_features.extend(normalize(lm))
            else:
                frame_features.extend([0.0] * 63)
            
            # Right Hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_features.extend(normalize(lm))
            else:
                frame_features.extend([0.0] * 63)
            
            # Pose
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_features.extend(normalize(lm))
            else:
                frame_features.extend([0.0] * 99)
            
            all_features.append(frame_features)
            frame_count += 1
    
    cap.release()
    
    if all_features:
        return np.mean(all_features, axis=0)
    return None

def train_classifier(mp_holistic, progress_bar):
    """Train Random Forest classifier on downloaded videos"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Find all videos
    files = glob.glob(f"{VIDEOS_DIR}/**/*.mp4", recursive=True)
    
    if not files:
        return None, None, False
    
    X_train, y_train = [], []
    
    for i, path in enumerate(files):
        filename = os.path.basename(path)
        
        # Match to vocabulary
        word = None
        for known in PSL_VOCABULARY.keys():
            if known.lower() in filename.lower():
                word = known
                break
        if not word:
            word = filename.split('.')[0]
        
        progress_bar.progress((i + 1) / len(files), f"Training: {word}")
        
        features = extract_landmarks(path, mp_holistic)
        if features is not None:
            X_train.append(features)
            y_train.append(word)
    
    if len(X_train) == 0:
        return None, None, False
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_encoded)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((classifier, label_encoder), f)
    
    return classifier, label_encoder, True

def load_or_train_classifier(mp_holistic, translator):
    """Load existing classifier or train new one"""
    
    # Try to load existing
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                classifier, label_encoder = pickle.load(f)
            return classifier, label_encoder, True
        except:
            pass
    
    # Need to train
    st.info("🔧 First run detected. Setting up the system...")
    
    # Step 1: Download videos
    st.write("📥 **Step 1/2:** Downloading training videos...")
    progress1 = st.progress(0)
    download_training_videos(translator, progress1)
    
    # Step 2: Train classifier
    st.write("🧠 **Step 2/2:** Training AI classifier...")
    progress2 = st.progress(0)
    classifier, label_encoder, success = train_classifier(mp_holistic, progress2)
    
    if success:
        st.success("✅ Setup complete! The system is ready.")
        return classifier, label_encoder, True
    else:
        st.error("❌ Setup failed. Please check your internet connection.")
        return None, None, False

# ===================== CORE FUNCTIONS =====================
def text_to_video(text: str, translator, slt):
    """Convert text to stitched sign language video"""
    words = text.lower().strip().split()
    video_clips = []
    
    for word in words:
        if word in PSL_VOCABULARY:
            urdu_word = PSL_VOCABULARY[word]
            try:
                sign_video = translator.translate(urdu_word)
                if len(sign_video) > 0:
                    video_clips.append(sign_video)
            except Exception as e:
                st.warning(f"⚠️ Could not find video for: {word}")
    
    if video_clips:
        # Stitch videos
        final_video = video_clips[0]
        for clip in video_clips[1:]:
            final_video = final_video + clip
        
        # Save to temp file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.save(output_path, overwrite=True)
        return output_path
    return None

def video_to_text(video_path, classifier, label_encoder, mp_holistic):
    """Recognize sign from video using trained classifier"""
    features = extract_landmarks(video_path, mp_holistic)
    
    if features is None:
        return None, 0
    
    features = features.reshape(1, -1)
    probabilities = classifier.predict_proba(features)[0]
    max_prob = np.max(probabilities)
    prediction = classifier.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_label.upper(), max_prob * 100

# ===================== STREAMLIT UI =====================
def main():
    # Header
    st.title("🤟 Sign Language Translator")
    st.markdown("**Bidirectional Translation:** Text ↔ Pakistani Sign Language (PSL)")
    st.markdown("---")
    
    # Load engines
    with st.spinner("⏳ Loading translation engine..."):
        translator, slt = load_slt_engine()
        mp_holistic = load_mediapipe()
    
    # Load or train classifier (AUTO)
    classifier, label_encoder, model_ready = load_or_train_classifier(mp_holistic, translator)
    
    if not model_ready:
        st.stop()
    
    st.success(f"✅ System Ready | Vocabulary: {list(label_encoder.classes_)}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📝 Text → Video", "🎥 Video → Text"])
    
    # ==================== TAB 1: TEXT TO VIDEO ====================
    with tab1:
        st.header("📝 Text to Sign Language Video")
        st.info(f"**Available words:** {', '.join(PSL_VOCABULARY.keys())}")
        
        text_input = st.text_input(
            "Enter text (English):",
            placeholder="e.g., apple good world"
        )
        
        if st.button("🎬 Generate Video", key="gen"):
            if text_input:
                with st.spinner("Generating sign language video..."):
                    video_path = text_to_video(text_input, translator, slt)
                    
                    if video_path and os.path.exists(video_path):
                        st.success("✅ Video generated!")
                        st.video(video_path)
                        
                        # Download button
                        with open(video_path, "rb") as f:
                            st.download_button(
                                "💾 Download Video",
                                f,
                                file_name="sign_language_output.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("❌ Could not generate video. Try different words.")
            else:
                st.warning("⚠️ Please enter some text first.")
    
    # ==================== TAB 2: VIDEO TO TEXT ====================
    with tab2:
        st.header("🎥 Sign Language Video to Text")
        
        uploaded_file = st.file_uploader(
            "Upload a sign language video (.mp4)",
            type=["mp4", "avi", "mov"]
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Preview
            st.video(temp_path)
            
            if st.button("🔍 Recognize Sign", key="rec"):
                with st.spinner("Analyzing video with Computer Vision..."):
                    label, confidence = video_to_text(temp_path, classifier, label_encoder, mp_holistic)
                    
                    if label:
                        # Color based on confidence
                        color = "#22c55e" if confidence > 70 else "#f59e0b"
                        
                        st.markdown(f"""
                        <div style='border: 3px solid {color}; border-radius: 15px; padding: 20px; text-align: center; background-color: #f9fafb;'>
                            <h1 style='color: {color};'>🏆 {label}</h1>
                            <div style='background: #e5e7eb; border-radius: 10px; height: 30px; margin: 10px 0;'>
                                <div style='width: {confidence}%; background: {color}; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                                    {confidence:.1f}%
                                </div>
                            </div>
                            <p style='color: #6b7280;'><b>Confidence Score:</b> {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Could not detect any landmarks in the video.")
    
    # Footer
    st.markdown("---")
    st.markdown("Designed by **Ahmed Eltaweel** | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    main()
