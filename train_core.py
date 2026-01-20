import sys
import os
from sign_language_core import SignLanguageCore

def main():
    print("🤟 Sign Language Core - Training Routine")
    print("========================================")
    
    core = SignLanguageCore()
    
    # Import engine from app context
    try:
        from app import load_slt_engine
        print("📥 Initializing SLT Engine...")
        translator, _ = load_slt_engine()
    except ImportError:
        print("❌ Error: Could not find slt engine in app.py. Ensure you are running from project root.")
        return

    print("🧬 Building Landmark Dictionary (DNA Extraction)...")
    core.build_landmark_dictionary(translator)
    
    print("🧠 Training Core Classifier...")
    if core.train_core():
        print(f"✅ Success! Model saved to: {core.model_path}")
        print(f"📚 Vocabulary: {list(core.landmark_dict.keys())}")
    else:
        print("❌ Training failed. Insufficient data.")

if __name__ == "__main__":
    main()
