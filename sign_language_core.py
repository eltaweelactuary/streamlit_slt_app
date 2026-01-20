import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
import speech_recognition as sr
from pathlib import Path

class SignLanguageCore:
    """
    The central engine for Next-Gen Sign Language Translation.
    Unifies Landmarks, Text, and Video into a single logical core.
    """
    def __init__(self, data_dir="./slt_core_assets"):
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.landmarks_dir = self.data_dir / "landmarks"
        self.models_dir = self.data_dir / "models" # Added models directory
        self.model_path = self.models_dir / "clr_model_v2.pkl" # Renamed model path
        
        # Create directories
        try:
            self.videos_dir.mkdir(parents=True, exist_ok=True)
            self.landmarks_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True) # Create models directory
        except Exception as e:
            print(f"Warning: Could not create directories in {self.data_dir}: {e}")
        
        # MediaPipe Setup
        self.mp_holistic = mp.solutions.holistic
        
        # Dictionary and Model
        self.landmark_dict = {}
        self.classifier = None
        self.label_encoder = None
        
        # Vocabulary (Extendable)
        self.vocabulary = {
            "apple": "سیب", "world": "دنيا", "pakistan": "پاکستان",
            "good": "اچھا", "red": "لال", "is": "ہے",
            "the": "یہ", "that": "وہ"
        }

    def extract_landmarks_from_video(self, video_path, max_frames=60, return_sequence=True):
        """Extract skeletal landmarks sequence using MediaPipe Holistic"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"🚫 Could not open video file: {video_path}")
            return None
            
        features_sequence = []
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret: break
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                ref_x, ref_y = 0.5, 0.5
                if results.pose_landmarks:
                    nose = results.pose_landmarks.landmark[0]
                    ref_x, ref_y = nose.x, nose.y
                
                def get_coords(res_attr, num_pts=21):
                    if not res_attr: return [0.0] * (num_pts * 3)
                    pts = res_attr.landmark[:num_pts]
                    return [c for lm in pts for c in [lm.x - ref_x, lm.y - ref_y, lm.z]]

                frame_features = []
                frame_features.extend(get_coords(results.left_hand_landmarks, 21))
                frame_features.extend(get_coords(results.right_hand_landmarks, 21))
                frame_features.extend(get_coords(results.pose_landmarks, 33))
                
                features_sequence.append(frame_features)
                count += 1
        
        cap.release()
        return np.array(features_sequence) if features_sequence else None

    def build_landmark_dictionary(self, translator):
        """Build the CLR Dictionary with Full Temporal Sequences"""
        print(f"🏗️ Building Temporal Landmark Dictionary for {len(self.vocabulary)} words...")
        for word, urdu in self.vocabulary.items():
            temp_v = self.videos_dir / f"{word}.mp4"
            if not temp_v.exists():
                print(f"📥 Downloading video for: {word}...")
                try:
                    clip = translator.translate(urdu)
                    clip.save(str(temp_v), overwrite=True)
                    print(f"💾 Saved video to {temp_v}")
                except Exception as e:
                    print(f"❌ Failed to translate/save '{word}': {e}")
                    continue
            
            print(f"🎥 Extracting landmarks for: {word}...")
            sequence = self.extract_landmarks_from_video(temp_v, return_sequence=True)
            if sequence is not None:
                print(f"✨ Detected {len(sequence)} frames of landmarks for '{word}'.")
                self.landmark_dict[word] = sequence
                np.save(self.landmarks_dir / f"{word}.npy", sequence)
            else:
                print(f"⚠️ No landmarks detected for '{word}'.")
        
        print(f"✅ Temporal Dictionary built with {len(self.landmark_dict)} word sequences.")

    def train_core(self):
        """Train the classifier using the Mean of Landmark Dictionary"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        if not self.landmark_dict:
            # Try loading from disk
            for npy in self.landmarks_dir.glob("*.npy"):
                word = npy.stem
                self.landmark_dict[word] = np.load(npy)
        
        if not self.landmark_dict: return False
        
        # --- BEST PRACTICE: DATA AUGMENTATION ---
        # With only 1 video per word, we must create synthetic variants
        X_aug = []
        y_aug = []
        
        print(f"🧬 Generating Synthetic Data Augmentation for {len(self.landmark_dict)} words...")
        for word, seq in self.landmark_dict.items():
            mean_vec = np.mean(seq, axis=0)
            # 1. Original
            X_aug.append(mean_vec)
            y_aug.append(word)
            
            # 2. Add variants (Noise + Scaling)
            for _ in range(50): # Create 50 variations per word
                noise = np.random.normal(0, 0.015, mean_vec.shape)
                scale = np.random.uniform(0.95, 1.05)
                variant = (mean_vec + noise) * scale
                X_aug.append(variant)
                y_aug.append(word)
        
        X = np.array(X_aug)
        y = np.array(y_aug)
        
        print(f"🧠 Training SLT Brain on {len(X)} samples (Augmented) with {X.shape[1]} features...")
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)
        self.classifier.fit(X, y_encoded)
        
        print(f"💾 Saving SLT Brain to {self.model_path}...")
        # Save Model
        with open(self.model_path, 'wb') as f:
            pickle.dump((self.classifier, self.label_encoder), f)
        print("✅ SLT Brain Saved and Ready!")
        return True

    def load_core(self):
        """Load the trained model"""
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                self.classifier, self.label_encoder = pickle.load(f)
            return True
        return False

    def predict_sign(self, video_path):
        """Predict sign from video using CLR Core"""
        if not self.classifier: return None, 0
        landmarks = self.extract_landmarks_from_video(video_path)
        return self.predict_from_landmarks(landmarks)

    def predict_from_landmarks(self, landmarks_vector):
        """Predict sign directly from a landmark vector (Real-time Best Practice)"""
        if self.classifier is None or landmarks_vector is None:
            return None, 0
            
        # Landmarks arrive as a temporal sequence (frames, features)
        # We must average them to match the training vector (1, features)
        if len(landmarks_vector.shape) > 1:
            features = np.mean(landmarks_vector, axis=0).reshape(1, -1)
        else:
            features = landmarks_vector.reshape(1, -1)
        
        prob = self.classifier.predict_proba(features)[0]
        max_idx = np.argmax(prob)
        label = self.label_encoder.inverse_transform([max_idx])[0]
        return label, prob[max_idx] * 100

    def get_word_dna(self, word):
        """Retrieve the Skeletal DNA (landmarks) for a specific word"""
        word = word.lower()
        if word in self.landmark_dict:
            return self.landmark_dict[word]
        
        # Try loading from disk
        path = self.landmarks_dir / f"{word}.npy"
        if path.exists():
            return np.load(path)
        return None

    def speech_to_text(self):
        """Convert live speech to text for translation input (Graceful handles server lack of mic)"""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("🎤 Listening...")
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio)
                return text
        except (OSError, ImportError) as e:
            print(f"🎙️ Mic/Speech Error: {e}")
            return "ERROR: Microphone not available on server. Please type your text."
        except Exception as e:
            print(f"🎙️ Speech Error: {e}")
            return None

class DigitalHumanRenderer:
    """
    Synthesizes a high-fidelity Digital Human Avatar from skeletal landmarks.
    This is the 'Best Practice' for generating clean, focused sign language output.
    """
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
        # Professional Styling (Cyan Neon Theme)
        self.node_spec = self.mp_drawing.DrawingSpec(color=(248, 189, 56), thickness=2, circle_radius=2)
        self.link_spec = self.mp_drawing.DrawingSpec(color=(56, 189, 248), thickness=3)

    def render_landmark_dna(self, landmark_sequence, output_path, width=640, height=480, fps=30):
        """Renders raw landmarks into a stylized digital human video clip"""
        if landmark_sequence is None or len(landmark_sequence) == 0: return None
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Smooth interpolation could be added here for SOTA transitions
        
        for frame_vec in landmark_sequence:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (15, 23, 42) # Slate-900 Background
            
            cx, cy = width // 2, height // 2
            
            def draw_points(points, start_idx, num_points, color):
                for i in range(num_points):
                    idx = start_idx + (i * 3)
                    if idx + 2 >= len(points): break
                    # Scale and offset DNA to screen coordinates
                    px = int(cx + (points[idx] * width * 0.8))
                    py = int(cy + (points[idx+1] * height * 0.8))
                    # Glow effect
                    cv2.circle(canvas, (px, py), 4, color, -1)
                    cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)

            # Hand DNA (Cyan) - 21 pts each
            draw_points(frame_vec, 0, 21, (56, 189, 248)) 
            draw_points(frame_vec, 63, 21, (56, 189, 248)) 
            # Pose DNA (Green) - 33 pts
            draw_points(frame_vec, 126, 33, (34, 197, 94))
            # Face landmarks removed for performance & stability in v2
            
            # HUD/UI Overlay for 'Digital Human' Look
            cv2.line(canvas, (0, 40), (width, 40), (56, 189, 248), 1)
            cv2.putText(canvas, "STATE-OF-THE-ART DIGITAL HUMAN AVATAR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)
            cv2.putText(canvas, "LATEST ACCESSIBILITY SYNTHESIS ACTIVE", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (34, 197, 94), 1)
            
            out.write(canvas)
        
        out.release()
        return output_path

    def stitch_and_render(self, dna_list, output_path):
        """Concatenates multiple DNA sequences for a seamless full-sentence performance"""
        if not dna_list: return None
        full_sequence = np.concatenate(dna_list, axis=0)
        return self.render_landmark_dna(full_sequence, output_path)
