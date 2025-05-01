import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from gtts import gTTS
import tempfile
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Image Captioning using ML", layout="centered")

# Load BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Title
st.markdown("<h1 style='text-align: center; color: #2973B2;'>📷 Image Captioning using ML</h1>", unsafe_allow_html=True)
st.write("Upload an image to generate a caption. You can also listen to or download the caption.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

caption = ""

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image (Click Generate)", use_column_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            st.success("✅ Caption Generated!")
            st.text_area("📝 Generated Caption", value=caption, height=100)

        # Speak the caption
        if caption:
            if st.button("🔊 Speak Caption"):
                with st.spinner("Generating speech..."):
                    tts = gTTS(text=caption)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        tts.save(fp.name)
                        audio_path = fp.name
                    st.audio(audio_path, format="audio/mp3")

            # Download caption as a .txt file
            if st.button("📥 Download Caption File"):
                st.download_button(
                    label="📄 Download Caption Text File",
                    data=caption,
                    file_name="caption.txt",
                    mime="text/plain"
                )
