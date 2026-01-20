# ☁️ Train AI Model on Google Colab (Save Local Space)

Since your local device is full, we will train the model on **Google Colab** (Free Cloud) and just download the small model file.

## 1️⃣ Open Google Colab
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click **"Upload"** -> select your notebook `Sign_Language_Real_Demo.ipynb`.
    *   *If you don't have it handy, I can generate a quick training script for you below.*

## 2️⃣ Run the Training Script
Copy and paste this code into a code cell in Colab and run it (`Shift + Enter`). It will do everything automatically.

```python
# 1. Install Dependencies
!pip install sign-language-translator mediapipe scikit-learn

# 2. Setup & Download Data
import os
import sign_language_translator as slt
from sign_language_translator.models import ConcatenativeSynthesis
import mediapipe as mp
import cv2
import numpy as np
import pickle
import glob

# Setup paths
DATA_DIR = "./psl_cv_assets"
os.makedirs(f"{DATA_DIR}/videos", exist_ok=True)
slt.Assets.set_root_dir(DATA_DIR)

# Initialize
translator = ConcatenativeSynthesis(text_language="urdu", sign_language="psl", sign_format="vid")
mp_holistic = mp.solutions.holistic

# Vocabulary
PSL_VOCABULARY = {
    "apple": "سیب", "world": "دنیا", "pakistan": "پاکستان",
    "good": "اچھا", "red": "لال", "is": "ہے", "the": "یہ", "that": "وہ"
}

# Download Videos
print("📥 Downloading videos...")
for eng, urdu in PSL_VOCABULARY.items():
    try:
        translator.translate(urdu)
    except:
        pass

# 3. Extract Features
print("👀 Extracting features (MediaPipe)...")
def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    all_features = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret: break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Normalization (Simple)
            features = []
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[0]
                ref_x, ref_y, ref_z = nose.x, nose.y, nose.z
                def norm(lm): return [lm.x - ref_x, lm.y - ref_y, lm.z - ref_z]
                
                # Add Hands & Pose
                for lms in [results.left_hand_landmarks, results.right_hand_landmarks]:
                    if lms: features.extend([c for lm in lms.landmark for c in norm(lm)])
                    else: features.extend([0.0]*63)
                if results.pose_landmarks:
                    features.extend([c for lm in results.pose_landmarks.landmark for c in norm(lm)])
                else: features.extend([0.0]*99)
                all_features.append(features)
            frame_count += 1
    cap.release()
    return np.mean(all_features, axis=0) if all_features else None

X, y = [], []
videos = glob.glob(f"{DATA_DIR}/videos/*.mp4")
for v in videos:
    feat = extract_landmarks(v)
    if feat is not None:
        name = [k for k in PSL_VOCABULARY if k in v.lower()]
        label = name[0] if name else "unknown"
        X.append(feat)
        y.append(label)

# 4. Train & Save
print("🧠 Training Model...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y_enc)

# Save
with open('psl_classifier.pkl', 'wb') as f:
    pickle.dump((clf, le), f)

print("✅ DONE! Downloading 'psl_classifier.pkl'...")
from google.colab import files
files.download('psl_classifier.pkl')
```

## 3️⃣ Upload to GitHub
1.  After the file `psl_classifier.pkl` downloads to your computer.
2.  Go to your GitHub Repository: [https://github.com/eltaweelactuary/streamlit_slt_app](https://github.com/eltaweelactuary/streamlit_slt_app)
3.  Click **"Add file"** -> **"Upload files"**.
4.  Drag and drop `psl_classifier.pkl`.
5.  Click **"Commit changes"**.

## 4️⃣ Deploy
Go back to Streamlit Cloud and click "Reboot" or "Deploy". It will now work instantly!
