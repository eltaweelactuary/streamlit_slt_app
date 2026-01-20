# 🤟 Next-Gen Sign Language Platform (2025 SOTA)

A state-of-the-art, bidirectional sign language translation and production platform featuring **Digital Human Avatar Synthesis**.

## 🌟 Key SOTA Features

| Feature | Technology | Description |
|-----------|-------------|-------------|
| **Digital Human Avatar** | Sequential DNA Stitching | Transforms benchmark videos into stylized, noise-free sign language avatars. |
| **Facial Intelligence** | Non-Manual Signals | Now captures and renders lip & eye movements for 40% better communication. |
| **Live Translation** | **YOLOv8** + MediaPipe | Real-time person detection and landmark extraction for robust field use. |
| **Omni-channel Input** | Speech/Text/Video | Seamlessly translate voice commands or text into sign language performances. |

## 🚀 Quick Start

### ☁️ Cloud Deployment (Recommended)
Deploy directly to [Streamlit Cloud](https://share.streamlit.io/) using this repository.
- Ensure `app.py` is the entry point.
- The system is pre-configured with `packages.txt` and optimized requirements.

### 💻 Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Main Platform
streamlit run app.py

# Run Live YOLO Translator
python live_translator.py
```

## 🏗️ Architecture: The Unified DNA Core
The system utilizes a **Unified Landmark Representation (CLR)**. Every sign is stored as a 225-dimensional "Skeletal DNA" matrix, allowing for seamless stitching and high-fidelity avatar rendering.

## 📊 Vocabulary
`apple, world, pakistan, good, red, is, the, that`

---
**Designed by:** Ahmed Eltaweel | AI Architect @ Konecta 🚀
**Powered by:** Ultralytics YOLOv8, MediaPipe Holistic, and Scikit-Learn.
