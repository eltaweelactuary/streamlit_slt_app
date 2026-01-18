# 🤟 Sign Language Translator - Streamlit App

A bidirectional sign language translation web application.

## Features

| Direction | Description |
|-----------|-------------|
| **Text → Video** | Enter English text → Get stitched PSL sign language video |
| **Video → Text** | Upload sign video → Get predicted text with confidence score |

## Quick Start

### Windows
```bash
# Double-click to run:
run_app.bat
```

### Manual
```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Then open: **http://localhost:8501**

## Requirements

- Python 3.9+
- Webcam (optional, for future live mode)
- `psl_classifier.pkl` (trained model from Colab notebook)

## Available Vocabulary

```
apple, world, pakistan, good, red, is, the, that
```

## Screenshots

| Text to Video | Video to Text |
|---------------|---------------|
| Enter text, get sign video | Upload video, get prediction |

---

**Powered by:** MediaPipe Holistic + Random Forest Classifier
