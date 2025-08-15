# Face Mask Detection (Streamlit)

A simple Streamlit app that detects if a face is wearing a mask using a TensorFlow/Keras model.

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   streamlit run main.py

## Model file

By default, large model artifacts are excluded from git (see `.gitignore`).
This app expects a file named `face_mask_model_fn.keras` in the project root.

Options:
- Place the model file locally next to `main.py` (not checked into git).
- Or modify `main.py` to download the model at runtime from cloud storage.

## Notes
- The repo excludes virtual environments and common large ML files to keep the repository small.
- If you need to add data samples, prefer smaller images and avoid binary blobs in git.
