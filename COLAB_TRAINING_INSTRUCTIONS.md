# ☁️ Train AI Model on Google Colab (Save Local Space)

Since your local device is full, we will train the model on **Google Colab** (Free Cloud) and just download the small model file.

## 1️⃣ Open Google Colab
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click **"Upload"** -> select your notebook `Sign_Language_Real_Demo.ipynb`.
    *   *If you don't have it handy, I can generate a quick training script for you below.*

## 2️⃣ Run the Training Script
Copy and paste this code into a code cell in Colab and run it (`Shift + Enter`). It will do everything automatically.

```python

Sign_Language_Real_Demo.ipynb
Sign_Language_Real_Demo.ipynb_
🔬 SLT Real Computer Vision Classifier (PRO Version)
Engine: MediaPipe Holistic + Random Forest | Dataset: Pakistani (PSL)
⚠️ Colab Compatibility Notice
Resource	Colab Status	Notes
PSL Dataset	✅ Works	Auto-downloads via SLT library
MediaPipe	✅ Works	CPU-based, no GPU required
KArSL (Saudi)	❌ Requires Manual Upload	100GB+ dataset
Egyptian Corpus	❌ Requires Manual Upload	Academic access only
🎯 Vocabulary (8 Words)
apple, world, pakistan, good, red, is, the, that

[ ]
# ⚙️ Step 1: Install Dependencies
print("⏳ Installing Real CV Pipeline...")
!pip uninstall -y numpy pandas scipy mediapipe sign-language-translator --quiet
!pip install "numpy<2.0" "pandas<2.2" "scipy<1.14" mediapipe==0.10.14 sign-language-translator scikit-learn opencv-python-headless ipywidgets --quiet
print("✅ Installation Complete.")
print("⚠️ MANDATORY: Go to 'Runtime' -> 'Restart Session' then run Step 2.")
⏳ Installing Real CV Pipeline...
WARNING: Skipping mediapipe as it is not installed.
WARNING: Skipping sign-language-translator as it is not installed.
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.0/61.0 kB 2.8 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.6/60.6 kB 4.9 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.8/55.8 kB 5.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.7/35.7 MB 56.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.0/18.0 MB 90.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 103.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.2/38.2 MB 12.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.5/181.5 kB 13.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 755.5/755.5 MB 1.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 410.6/410.6 MB 4.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 73.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 29.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 823.6/823.6 kB 51.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.7/731.7 MB 2.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 MB 7.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 13.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 7.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 6.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 166.0/166.0 MB 6.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 9.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 13.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 69.1/69.1 MB 11.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 25.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 104.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.2/81.2 MB 10.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 78.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.7/39.7 MB 18.4 MB/s eta 0:00:00
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-colab 1.0.0 requires pandas==2.2.2, but you have pandas 2.1.4 which is incompatible.
grpcio-status 1.71.2 requires protobuf<6.0dev,>=5.26.1, but you have protobuf 4.25.8 which is incompatible.
pytensor 2.36.3 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
opentelemetry-proto 1.37.0 requires protobuf<7.0,>=5.0, but you have protobuf 4.25.8 which is incompatible.
rasterio 1.5.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
xarray 2025.12.0 requires pandas>=2.2, but you have pandas 2.1.4 which is incompatible.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
access 1.1.10.post3 requires scipy>=1.14.1, but you have scipy 1.13.1 which is incompatible.
shap 0.50.0 requires numpy>=2, but you have numpy 1.26.4 which is incompatible.
torchaudio 2.9.0+cpu requires torch==2.9.0, but you have torch 2.2.2 which is incompatible.
plotnine 0.14.5 requires pandas>=2.2.0, but you have pandas 2.1.4 which is incompatible.
tsfresh 0.21.1 requires scipy>=1.14.0; python_version >= "3.10", but you have scipy 1.13.1 which is incompatible.
mizani 0.13.5 requires pandas>=2.2.0, but you have pandas 2.1.4 which is incompatible.
ydf 0.13.0 requires protobuf<7.0.0,>=5.29.1, but you have protobuf 4.25.8 which is incompatible.
tobler 0.13.0 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.
tobler 0.13.0 requires pandas>=2.2, but you have pandas 2.1.4 which is incompatible.
torchvision 0.24.0+cpu requires torch==2.9.0, but you have torch 2.2.2 which is incompatible.
✅ Installation Complete.
⚠️ MANDATORY: Go to 'Runtime' -> 'Restart Session' then run Step 2.

[1]
20s
# ⚙️ Step 2: Initialize All Engines
import os
import cv2
import glob
import shutil
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



📥 Step 3: Download Videos (Auto-Cache Mode)
Downloads are automatically handled by the library and stored in ./psl_cv_assets/videos.


[2]
3s
# 📥 Step 3: Video Downloader
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

print(f"📥 Retrieving videos...\n")

for english_word, urdu_word in PSL_VOCABULARY.items():
    try:
        print(f"   🔄 Fetching asset for: {english_word}...")
        # Translate triggers the download internally to VIDEOS_DIR
        sign_video = translator.translate(urdu_word)
        if len(sign_video) > 0:
            print(f"      ✔️ Asset acquired.")
    except Exception as e:
        print(f"      ⚠️ Warning: {e}")

print("\n✅ All reachable assets cached.")

🎬 Step 4: Video Gallery (Verified Path)

[3]
0s
# 🎬 Gallery - Scans ./psl_cv_assets/videos
VIDEOS_DIR = "./psl_cv_assets/videos"
print(f"📂 Scanning directory: {VIDEOS_DIR} ...")

if not os.path.exists(VIDEOS_DIR):
    print("❌ Directory missing. Run Step 3.")
else:
    # Recursively find all mp4 files
    files = glob.glob(f"{VIDEOS_DIR}/**/*.mp4", recursive=True)
    files = sorted(files)
    
    if not files:
        print("❌ No .mp4 files found.")
    else:
        print(f"✅ Found {len(files)} videos.\n")
        print("="*40)
        
        for path in files:
            filename = os.path.basename(path)
            # Simple visual label
            label = filename.split('_')[0] if '_' in filename else filename
            
            print(f"📌 {filename}")
            display(Video(path, embed=True, html_attributes="controls", width=300))
            display(FileLink(path, result_html_prefix=f"💾 Download: "))
            print("-" * 40)

🧬 Phase A: PRO Landmark Extraction (Normalized + Bug Fixed)
Includes normalization to handle distance/camera shifts and fixed right-hand logic.


[4]
0s
def extract_landmarks_from_video(video_path, max_frames=30):
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
            
            # 1. Normalization Reference (Nose or Center of Shoulders)
            ref_x, ref_y, ref_z = 0, 0, 0
            if results.pose_landmarks:
                # Use Nose (landmark 0) as reference point
                nose = results.pose_landmarks.landmark[0]
                ref_x, ref_y, ref_z = nose.x, nose.y, nose.z

            # Helper to normalize
            def normalize(lm):
                return [lm.x - ref_x, lm.y - ref_y, lm.z - ref_z]

            # 2. Extract & Normalize Left Hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_features.extend(normalize(lm))
            else:
                frame_features.extend([0.0] * 63)
            
            # 3. Extract & Normalize Right Hand (BUG FIXED HERE)
            if results.right_hand_landmarks:
                # CORRECTED LOGIC: Use results.right_hand_landmarks, NOT left
                for lm in results.right_hand_landmarks.landmark:
                    frame_features.extend(normalize(lm))
            else:
                frame_features.extend([0.0] * 63)
            
            # 4. Extract & Normalize Pose
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_features.extend(normalize(lm))
            else:
                frame_features.extend([0.0] * 99)
            
            all_features.append(frame_features)
            frame_count += 1
    
    cap.release()
    
    # Averaging features (can be improved to flattening for temporal logic)
    if all_features:
        return np.mean(all_features, axis=0)
    return None

print("✅ PRO Landmark Extractor Ready (Normalized + Fixed).")
✅ PRO Landmark Extractor Ready (Normalized + Fixed).
🎓 Phase B: Train Classifier (Random Forest)

[5]
24s
# Generate Training Data from ./psl_cv_assets/videos
VIDEOS_DIR = "./psl_cv_assets/videos"
X_train, y_train = [], []

print(f"🔹 Scanning {VIDEOS_DIR} for training data...\n")

if not os.path.exists(VIDEOS_DIR):
    print("❌ Directory missing. Run Step 3.")
else:
    files = glob.glob(f"{VIDEOS_DIR}/**/*.mp4", recursive=True)
    
    if not files:
        print("❌ No videos found to train on!")
    else:
        for path in files:
            # Map filename to English word
            filename = os.path.basename(path)
            word = None
            for known in PSL_VOCABULARY.keys():
                if known.lower() in filename.lower():
                    word = known
                    break
            if not word:
                word = filename.split('.')[0]

            print(f"   📊 Processing: {word} (File: {filename})")
            features = extract_landmarks_from_video(path)
            
            if features is not None:
                X_train.append(features)
                y_train.append(word)
                print(f"   ✔️ Done.")
            else:
                print(f"   ⚠️ No features extracted.")

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(f"\n📊 Training Data: {len(X_train)} samples, {len(set(y_train))} classes")
🔹 Scanning ./psl_cv_assets/videos for training data...

   📊 Processing: apple (File: pk-hfad-1_apple.mp4)
/usr/local/lib/python3.12/dist-packages/google/protobuf/symbol_database.py:55: UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.
  warnings.warn('SymbolDatabase.GetPrototype() is deprecated. Please '
   ✔️ Done.
   📊 Processing: is (File: pk-hfad-1_is.mp4)
   ✔️ Done.
   📊 Processing: that (File: pk-hfad-1_that.mp4)
   ✔️ Done.
   📊 Processing: red (File: pk-hfad-1_red.mp4)
   ✔️ Done.
   📊 Processing: pk-hfad-1_it (File: pk-hfad-1_it.mp4)
   ✔️ Done.
   📊 Processing: world (File: pk-hfad-1_world.mp4)
   ✔️ Done.
   📊 Processing: good (File: pk-hfad-1_good.mp4)
   ✔️ Done.
   📊 Processing: pk-hfad-1_p(double-handed-letter) (File: pk-hfad-1_p(double-handed-letter).mp4)
   ✔️ Done.

📊 Training Data: 8 samples, 8 classes

[6]
0s
# Train Classifier - Using Random Forest for Stability
if len(X_train) > 0:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    # Random Forest is more robust to noise and high dimensions than KNN
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_encoded)
    
    with open('psl_classifier.pkl', 'wb') as f:
        pickle.dump((classifier, label_encoder), f)
    
    print("✅ Random Forest Classifier Trained!")
    print(f"🎯 Can recognize: {list(label_encoder.classes_)}")
else:
    print("❌ No training data. Run Step 3 first.")
✅ Random Forest Classifier Trained!
🎯 Can recognize: ['apple', 'good', 'is', 'pk-hfad-1_it', 'pk-hfad-1_p(double-handed-letter)', 'red', 'that', 'world']
🧠 Phase 2: Real Computer Vision Recognition (Pro UI)

[7]
0s
# Load Classifier
try:
    with open('psl_classifier.pkl', 'rb') as f:
        classifier, label_encoder = pickle.load(f)
    MODEL_READY = True
    print("✅ Classifier Ready.")
    print(f"🎯 Knows: {list(label_encoder.classes_)}")
except:
    MODEL_READY = False
    print("⚠️ Train the model first (Phase B).")



[8]
0s
4.  Drag and drop `psl_classifier.pkl`.
5.  Click **"Commit changes"**.

Next steps:
Colab paid products - Cancel contracts here


## 4️⃣ Deploy
Go back to Streamlit Cloud and click "Reboot" or "Deploy". It will now work instantly!
