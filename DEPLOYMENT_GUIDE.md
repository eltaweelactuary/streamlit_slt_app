# 🚀 Deployment Guide: GitHub & Streamlit Cloud

This guide explains how to upload your **Sign Language Translator** to GitHub and deploy it live on Streamlit Cloud.

---

## ✅ Phase 1: Prepare for GitHub

We have already auto-generated the necessary configuration files for you:
1.  **`packages.txt`**: Tells Streamlit to install `libgl1` (required for OpenCV).
2.  **`requirements.txt`**: Lists all Python libraries (updated for Cloud compatibility).
3.  **`.gitignore`**: Prevents uploading junk files (like `venv` or `__pycache__`).

---

## 🔼 Phase 2: Upload to GitHub

1.  **Create a New Repository** on GitHub (e.g., `slt-app`).
2.  **Open Terminal** in the `streamlit_slt_app` folder:
    ```bash
    cd streamlit_slt_app
    git init
    git add .
    git commit -m "Initial commit for Signal Language App"
    git branch -M main
    git remote add origin https://github.com/[YOUR_USERNAME]/slt-app.git
    git push -u origin main
    ```

---

## ☁️ Phase 3: Deploy on Streamlit Cloud

1.  Go to **[share.streamlit.io](https://share.streamlit.io/)**.
2.  Click **"New App"**.
3.  Select your repository (`slt-app`).
4.  **Main file path**: `app.py`.
5.  Click **"Deploy!"**.

---

## ⚡ Important Notes

*   **First Run:** The app will take 2-3 minutes to start because it needs to **download the 8 vocabulary videos** and **train the AI model** (Random Forest) from scratch on the cloud server.
*   **Logs:** You can see the progress (Downloading... Training...) in the "Manage App" -> "Logs" section on Streamlit Cloud.
*   **Persistence:** Once trained, the app will run instantly for subsequent users.

---

## 🛠️ Troubleshooting (Cloud)

| Error | Solution |
|-------|----------|
| `GLIBCXX_3.4.29 not found` | Ensure `packages.txt` exists and contains `libgl1-mesa-glx`. |
| `MemoryLimitExceeded` | The app is lightweight, but if this happens, restart the app via dashboard. |
| `ModuleNotFoundError` | Check `requirements.txt` names. |
