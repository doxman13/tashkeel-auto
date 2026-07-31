import streamlit as st
from streamlit_cropper import st_cropper
import easyocr

# Patch sqlite3 to allow Mishkal's internal databases to be shared across Streamlit threads
import sqlite3
_original_connect = sqlite3.connect
def patched_connect(*args, **kwargs):
    kwargs['check_same_thread'] = False
    return _original_connect(*args, **kwargs)
sqlite3.connect = patched_connect

import mishkal.tashkeel
from farasa.diacratizer import FarasaDiacritizer
from deep_translator import GoogleTranslator
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image
import numpy as np
import os
from google import genai

# Setup Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except:
        GEMINI_API_KEY = None
if not GEMINI_API_KEY:
    # YOUR_GEMINI_API_KEY_HERE - Paste your free API key here if not using env vars or secrets
    GEMINI_API_KEY = ""

def gemini_tashkeel(raw_arabic_text: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is missing. Please set it in the code, secrets, or environment variables.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = "You are an expert Arabic grammarian and linguist. Add full, grammatically precise Tashkeel (diacritical marks / vowels) to the provided Arabic text. Automatically fix minor OCR typos (such as truncated prepositions like 'إل' -> 'إلى' or missing letter tails). Return ONLY the corrected, fully diacritized Arabic text without any preamble, markdown, or explanations.\n\nText: " + raw_arabic_text
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    return response.text.strip()

# Page config
st.set_page_config(layout="wide", page_title="Manga Arabic Diacritizer")
st.title("Manga Arabic Diacritizer (Self-Study)")
st.markdown("Upload your Arabic manga pages, crop the speech bubbles, and instantly extract and diacritize the text to help with your self-study!")

# Cache the OCR reader to avoid reloading on every interaction
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['ar'])

# Cache tashkeel object
@st.cache_resource
def load_mishkal_model():
    return mishkal.tashkeel.TashkeelClass()

reader = load_ocr_model()
mishkal_voweler = load_mishkal_model()

@st.cache_resource
def load_farasa_model():
    return FarasaDiacritizer(interactive=True)
    
farasa_voweler = load_farasa_model()

def process_text_for_display(text):
    """Reshape and apply bidi algorithm for proper RTL display in Streamlit widgets."""
    if not text.strip():
        return ""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# Sidebar: Engine Selection
st.sidebar.title("Settings")
engine_choice = st.sidebar.selectbox(
    "Select Diacritization Engine",
    [
        "Gemini 1.5 Flash (AI - Highest Accuracy)",
        "Farasa (Statistical NLP)", 
        "Mishkal (Rule-Based)"
    ]
)
st.sidebar.info("💡 **Tip:** Gemini 1.5 Flash provides state-of-the-art accuracy by understanding context and automatically correcting OCR errors before adding vowels.")

# 1. File Upload Limit (max 10 files)
uploaded_files = st.file_uploader("Upload Manga Pages (Max 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Enforce limit of 10 files
    if len(uploaded_files) > 10:
        st.warning(f"⚠️ You have uploaded {len(uploaded_files)} files. Only the first 10 will be processed to stay within limits.")
        uploaded_files = uploaded_files[:10]
        
    # 2. Image Selection & Cropping
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.selectbox("Select Image to Crop", file_names)
    
    # Find the selected file object
    selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
    
    # Load image using PIL
    img = Image.open(selected_file)
    
    # Inject custom CSS to make the left column sticky
    st.markdown("""
        <style>
            /* Target the first column specifically (supports older and newer Streamlit versions) */
            div[data-testid="column"]:nth-of-type(1),
            div[data-testid="stColumn"]:nth-of-type(1) {
                position: -webkit-sticky !important;
                position: sticky !important;
                top: 4rem !important; /* Adjust based on Streamlit's header height */
                align-self: flex-start !important; /* Important for flex items to stick properly */
                z-index: 999 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # UI Layout: Left Column (Cropper), Right Column (Results)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Crop Speech Bubble")
        st.markdown("Draw a rectangle over the speech bubble you want to extract text from.")
        
        # Interactive cropping tool
        cropped_img = st_cropper(
            img, 
            realtime_update=True, 
            box_color='#FF0000', 
            aspect_ratio=None,
            key=f"cropper_{selected_file_name}"
        )
        
    with col2:
        st.subheader("2. Extracted & Diacritized Text")
        
        if cropped_img:
            st.image(cropped_img, caption="Cropped Area", width=150)
            
            if st.button("Process & Add Vowels"):
                # Convert PIL image to numpy array for EasyOCR
                cropped_array = np.array(cropped_img)
                
                with st.spinner("Extracting text and applying Tashkeel..."):
                    # Read text using EasyOCR (ar)
                    results = reader.readtext(cropped_array, detail=0, paragraph=True)
                    extracted_text = " ".join(results)
                    
                    if not extracted_text.strip():
                        st.error("No text detected in the selected area. Please try cropping a clearer area or a different bubble.")
                    else:
                        # Diacritize dynamically based on selected engine
                        if "Gemini" in engine_choice:
                            try:
                                diacritized_text = gemini_tashkeel(extracted_text)
                            except Exception as e:
                                st.error(f"Gemini API Error: {e}")
                                diacritized_text = extracted_text
                        elif "Farasa" in engine_choice:
                            diacritized_text = farasa_voweler.diacritize(extracted_text)
                        else:
                            diacritized_text = mishkal_voweler.tashkeel(extracted_text)
                        
                        # Apply bidi/reshaper for text area (which doesn't natively support HTML RTL styling)
                        bidi_raw = process_text_for_display(extracted_text)
                        bidi_diacritized = process_text_for_display(diacritized_text)
                        
                        st.markdown("### Raw Extracted Text")
                        st.text_area(
                            "Raw Text (Editable / Copyable)", 
                            value=bidi_raw, 
                            height=100, 
                            key="raw_text"
                        )
                        
                        st.markdown("### Diacritized Text (Tashkeel)")
                        st.text_area(
                            "Tashkeel Text (Editable / Copyable)", 
                            value=bidi_diacritized, 
                            height=150, 
                            key="diacritized_text"
                        )
                        
                        st.markdown("### English Translation")
                        try:
                            translated_text = GoogleTranslator(source='ar', target='en').translate(extracted_text)
                            st.success(translated_text)
                        except Exception as e:
                            st.warning(f"Translation failed: {e}")
                        
                        # Extra: Render with native HTML for larger, cleaner reading
                        st.markdown("---")
                        st.markdown("### Native Render (Large Font)")
                        
                        # We render the original non-bidi'ed text because browsers handle standard Arabic 
                        # perfectly when dir="rtl" is specified. This provides the most natural look.
                        html_str = f"""
                        <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                            <div style="font-size: 24px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px; background-color: rgba(200, 200, 200, 0.1);">
                                <strong>Raw:</strong> {extracted_text}
                            </div>
                            <div style="font-size: 32px; padding: 15px; border: 2px solid #4CAF50; border-radius: 8px; color: #2E7D32; background-color: rgba(76, 175, 80, 0.05); line-height: 1.8;">
                                <strong>Tashkeel:</strong> {diacritized_text}
                            </div>
                        </div>
                        """
                        st.markdown(html_str, unsafe_allow_html=True)
else:
    st.info("Please upload one or more manga pages to begin.")
