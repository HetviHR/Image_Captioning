import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from gtts import gTTS
import tempfile

# Set Streamlit page configuration
st.set_page_config(page_title="Image Captioning using ML", layout="centered")

# Load BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Initialize session state for caption
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "image" not in st.session_state:
    st.session_state.image = None

# Title
st.markdown("<h1 style='text-align: center; color: #2973B2;'>📷 Image Captioning using ML</h1>", unsafe_allow_html=True)
st.write("Upload an image to generate a caption. You can also listen to or download the caption.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")
    st.image(st.session_state.image, caption="📷 Uploaded Image", use_column_width=True)

    # "Click image to caption" functionality (simulated)
    if st.button("🖱️ Click to Generate Caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(images=st.session_state.image, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs)
            st.session_state.caption = processor.decode(outputs[0], skip_special_tokens=True)
        st.success("✅ Caption Generated!")

# Show caption if generated
if st.session_state.caption:
    st.text_area("📝 Generated Caption", value=st.session_state.caption, height=100)

    # Speak the caption
    if st.button("🔊 Speak Caption"):
        with st.spinner("Generating speech..."):
            tts = gTTS(text=st.session_state.caption)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_path = fp.name
            st.audio(audio_path, format="audio/mp3")

    # Download caption as .txt file
    st.download_button(
        label="📥 Download Caption Text File",
        data=st.session_state.caption,
        file_name="caption.txt",
        mime="text/plain"
    )
