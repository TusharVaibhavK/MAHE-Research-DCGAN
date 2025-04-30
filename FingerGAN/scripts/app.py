import streamlit as st
from PIL import Image

# Streamlit app
st.title("FingerGAN: Synthetic Fingerprint Generation")

st.header("Real Fingerprint Sample")
# Load and display the real fingerprint
real_image_path = "data/SOCOFing/Real/sample_fingerprint.png"
try:
    real_image = Image.open(real_image_path)
    st.image(real_image, caption="Real Fingerprint", use_column_width=True)
except FileNotFoundError:
    st.error(f"Real fingerprint image not found at {real_image_path}")

st.header("Generated Fingerprints (DCGAN)")
# Load and display the generated fingerprints
generated_image_path = "generated_images/generated_fingerprints.png"
try:
    generated_image = Image.open(generated_image_path)
    st.image(generated_image,
             caption="Generated Fingerprints (4x4 Grid)", use_column_width=True)
except FileNotFoundError:
    st.error(f"Generated image not found at {generated_image_path}")
