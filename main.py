import os
from pathlib import Path
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import applications

# Define custom objects
@tf.keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    class_weights = {0: 1.0176827214550355, 1: 0.9829212752114509}
    weights = tf.where(y_true == 1, class_weights[1], class_weights[0])
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights)

def _get_model_url() -> str | None:
    """Resolve the model URL from Streamlit secrets or environment variable."""
    # Prefer Streamlit secrets in Cloud
    try:
        if hasattr(st, "secrets") and "MODEL_URL" in st.secrets:
            return st.secrets["MODEL_URL"]
    except Exception:
        pass
    # Fallback to environment variable for local/dev
    return os.getenv("MODEL_URL")


def ensure_model_file(path: str = "face_mask_model_fn.keras") -> str | None:
    """Ensure the model file exists locally; download it if a URL is provided.

    Returns the local path if available or downloaded, else None.
    """
    path = str(Path(path))
    if os.path.exists(path):
        return path

    url = _get_model_url()
    if not url:
        st.error(
            "Model file not found and no MODEL_URL provided. "
            "Add MODEL_URL to Streamlit secrets (or env) pointing to the .keras file."
        )
        return None

    try:
        st.info("Downloading model… this happens only once per deployment.")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp_path = f"{path}.part"
            dl = 0
            progress = st.progress(0, text="Downloading model")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if total:
                        dl += len(chunk)
                        progress.progress(min(100, int(dl * 100 / total)))
            progress.empty()
            os.replace(tmp_path, path)
        st.success("✅ Model downloaded.")
        return path
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        return None


# Load Model
@st.cache_resource
def load_mask_model():
    try:
        model_path = ensure_model_file("face_mask_model_fn.keras")
        if not model_path:

            return None
        model = load_model(
            model_path,
            custom_objects={'weighted_loss': weighted_loss}
        )
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

model = load_mask_model()

# Streamlit app layout
st.title("Face Mask Detection 😷")
st.write("Upload an image to check if the person is wearing a mask")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Max file size: 5MB"
)

use_camera = st.checkbox("Use camera instead")
if use_camera:
    uploaded_file = st.camera_input("Take a picture")

# Prediction logic
if model and uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_resized = image.resize((160, 160))
        img_array = img_to_array(img_resized)
        img_array = applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        mask_prob = 1 - prediction
        confidence = mask_prob * 100
        
        st.subheader("Prediction Results")
        progress_bar = st.progress(0)
        
        if mask_prob > 0.3:
            progress_bar.progress(int(confidence))
            st.success(f"✅ Mask Detected ({confidence:.1f}%)")
            st.balloons()
        else:
            progress_bar.progress(int(confidence))
            st.error(f"🚨 No Mask Detected ({confidence:.1f}%)")
        
        with st.expander("Advanced Metrics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mask Confidence", f"{confidence:.2f}%")
            with col2:
                st.metric("No Mask Confidence", f"{prediction*100:.2f}%")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.sidebar.markdown("""
### Model Performance
- Test Accuracy: 99.15%
- Precision: 99.11%
- Recall: 99.22%
- AUC: 0.999
""")
