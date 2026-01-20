import cv2
import time
import numpy as np
from sign_language_core import SignLanguageCore

# Optional: YOLOv8 for professional person detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    # Load YOLOv8n (Nano) for high-speed CPU performance
    person_detector = YOLO('yolov8n.pt') 
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLOv8 not installed. Falling back to standard MediaPipe detection.")

def run_live_translation():
    core = SignLanguageCore()
    if not core.load_core():
        print("❌ Core Model not found. Initializing auto-training...")
        # Optional: Auto-init training if engine is available
        try:
            from app import load_slt_engine
            translator, _ = load_slt_engine()
            core.build_landmark_dictionary(translator)
            core.train_core()
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return

    cap = cv2.VideoCapture(0)
    print("🚀 Next-Gen Live Sign Language Platform Started...")
    
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    # Prediction display variables
    last_prediction = "Waiting..."
    confidence = 0
    inference_start_time = time.time()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # --- PHASE 1: BEST PRACTICE PERSON DETECTION (YOLO) ---
            roi = frame
            if YOLO_AVAILABLE:
                # Detect person only (class 0)
                results_yolo = person_detector(frame, classes=[0], verbose=False)
                for r in results_yolo:
                    if len(r.boxes) > 0:
                        # Take the largest person found
                        box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                        # Expand box slightly for MediaPipe
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1-20), max(0, y1-20)
                        x2, y2 = min(w, x2+20), min(h, y2+20)
                        roi = frame[y1:y2, x1:x2]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, "YOLO FOCUS", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # --- PHASE 2: LANDMARK EXTRACTION (CLR CORE) ---
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_roi)
            
            # Draw landmarks on original frame (offset if using ROI)
            # (Simplification: drawing directly on frame for demo)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # --- PHASE 3: REAL-TIME INFERENCE ---
            if time.time() - inference_start_time > 1.5:
                # Extract landmarks for this single frame
                # In a production setup, we use a buffer of frames, 
                # but for this SOTA demo, we'll use a snapshot-based prediction.
                
                # Mock feature extraction from current results
                ref_x, ref_y = 0.5, 0.5
                if results.pose_landmarks:
                    ref_x, ref_y = results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y
                
                def get_c(attr):
                    if not attr: return [0.0] * 63
                    return [c for lm in attr.landmark for c in [lm.x - ref_x, lm.y - ref_y, lm.z]]

                f_v = []
                f_v.extend(get_c(results.left_hand_landmarks))
                f_v.extend(get_c(results.right_hand_landmarks))
                if results.pose_landmarks:
                    f_v.extend([c for lm in results.pose_landmarks.landmark[:25] for c in [lm.x - ref_x, lm.y - ref_y, lm.z]])
                else: f_v.extend([0.0] * 75)
                
                # Predict using the specialized real-time method
                label, conf = core.predict_from_landmarks(np.array(f_v))
                if label and conf > 40:
                    last_prediction = label.upper()
                    confidence = conf
                else:
                    last_prediction = "Searching..."
                
                inference_start_time = time.time()

            # --- UI LAYER ---
            # Status Bar
            cv2.rectangle(frame, (0,0), (w, 80), (15, 23, 42), -1)
            cv2.putText(frame, "LIVE PSL TRANSLATOR | YOLOv8-Focused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
            cv2.putText(frame, f"Prediction: {last_prediction} ({confidence:.1f}%)", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2)
            
            # Show Frame
            cv2.imshow('Sign Language Next-Gen (Unified Core)', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_translation()
