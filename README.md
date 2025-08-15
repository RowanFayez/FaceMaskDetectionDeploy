# Face Mask Detection (Streamlit)

A simple Streamlit app that detects if a face is wearing a mask using a TensorFlow/Keras model.

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   streamlit run main.py

## Model file

Large model artifacts are excluded from git (see `.gitignore`). The app will
auto-download the model at runtime if you provide a URL (no need to commit the
binary file).

Expected filename: `face_mask_model_fn.keras`

Provide a URL via:
- Streamlit secrets (recommended for Cloud): set `MODEL_URL` in the app’s Secrets.
- Environment variable (local/dev): set `MODEL_URL` to a direct link to the `.keras` file.

The first run will download the file and cache it on disk.

## Deploy on Streamlit Cloud

1. Push this repo (you already did).
2. On Streamlit Cloud, create a new app from this repo.
3. In the app’s Settings → Secrets, add:

   MODEL_URL = https://your-storage/face_mask_model_fn.keras

4. Save and deploy. The app will download the model on first run.

## Notes
- The repo excludes virtual environments and common large ML files to keep the repository small.
- If you need to add data samples, prefer smaller images and avoid binary blobs in git.
