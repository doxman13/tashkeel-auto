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
from farasa.stemmer import FarasaStemmer
from deep_translator import GoogleTranslator
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image
import numpy as np
import os
import re
import json
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Configuration variable - paste your API key here or use env vars / secrets
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

def get_gemini_api_key():
    key = GEMINI_API_KEY
    if not key or key == "YOUR_GEMINI_API_KEY_HERE":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            try:
                key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                key = None
    return key

def gemini_tashkeel(raw_arabic_text: str) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    system_prompt = (
        "You are an expert Arabic grammarian. Add full, grammatically precise Tashkeel (diacritical marks / vowels) "
        "to the provided Arabic text. Fix minor OCR typos if present (such as truncated prepositions like 'إل' -> 'إلى'). "
        "Return ONLY the corrected, fully diacritized Arabic text without preamble, markdown, or conversational filler."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=raw_arabic_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    return response.text.strip()

# Structured output schemas for Sarf analysis
class SarfTable(BaseModel):
    wazn: str = Field(description="Pattern/Form, e.g., 'Form IV / أَفْعَلَ'")
    madi: str = Field(description="Past tense 3rd person, fully diacritized")
    mudari: str = Field(description="Present tense 3rd person, fully diacritized")
    masdar: str = Field(description="Verbal noun, fully diacritized")
    ism_faail: str = Field(description="Active participle, fully diacritized")
    ism_mafool: str = Field(description="Passive participle, fully diacritized")

def get_sarf_table(word: str, root: str) -> dict:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    system_prompt = (
        "You are an expert Arabic grammarian and morphologist. Analyze the provided word and root, "
        "and output its morphological analysis table as a JSON object matching the requested schema. "
        "Ensure all Arabic words in the response are fully and precisely diacritized."
    )
    
    prompt = (
        f"Analyze the morphological structure (Sarf) of the Arabic word '{word}' "
        f"which has the root '{root}'.\n"
        f"Provide the pattern/form (wazn), past tense (madi), present tense (mudari), "
        f"verbal noun (masdar), active participle (ism_faail), and passive participle (ism_mafool) "
        f"for this word's paradigm."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SarfTable,
            system_instruction=system_prompt,
        )
    )
    
    data = json.loads(response.text.strip())
    return data

def translate_words(words_list: list) -> list:
    """Translate a list of Arabic words to English in a single batch call to deep_translator."""
    if not words_list:
        return []
    # Replace empty values with placeholder to keep alignment
    cleaned_words = [w if w and w.strip() else "..." for w in words_list]
    batch_str = " | ".join(cleaned_words)
    try:
        translated_str = GoogleTranslator(source='ar', target='en').translate(batch_str)
        translated_list = [t.strip() for t in translated_str.split('|')]
        # Pad list if length doesn't match
        while len(translated_list) < len(words_list):
            translated_list.append("N/A")
        # Replace placeholder back with original or empty
        return [t if t != "..." and t != "N/A" else "N/A" for t in translated_list]
    except Exception:
        return ["N/A"] * len(words_list)


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

@st.cache_resource
def load_farasa_stemmer():
    return FarasaStemmer(interactive=True)

farasa_stemmer = load_farasa_stemmer()

# Initialize session state variables
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'diacritized_text' not in st.session_state:
    st.session_state.diacritized_text = ""
if 'current_file' not in st.session_state:
    st.session_state.current_file = ""
if 'sarf_results' not in st.session_state:
    st.session_state.sarf_results = None
if 'sarf_word' not in st.session_state:
    st.session_state.sarf_word = ""

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
    
    # Reset session state if the selected file name changes
    if st.session_state.current_file != selected_file_name:
        st.session_state.current_file = selected_file_name
        st.session_state.extracted_text = ""
        st.session_state.diacritized_text = ""
        st.session_state.sarf_results = None
        st.session_state.sarf_word = ""

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
        st.subheader("2. Extracted and diacritized text")
        
        if cropped_img:
            st.image(cropped_img, caption="Cropped Area", width=150)
            
            if st.button("Process & Add Vowels", type="primary"):
                # Convert PIL image to numpy array for EasyOCR
                cropped_array = np.array(cropped_img)
                
                with st.spinner("Extracting text and applying Tashkeel..."):
                    # Read text using EasyOCR (ar)
                    results = reader.readtext(cropped_array, detail=0, paragraph=True)
                    extracted_text = " ".join(results)
                    
                    if not extracted_text.strip():
                        st.error("No text detected in the selected area. Please try cropping a clearer area or a different bubble.")
                        st.session_state.extracted_text = ""
                        st.session_state.diacritized_text = ""
                        st.session_state.sarf_results = None
                        st.session_state.sarf_word = ""
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
                        
                        st.session_state.extracted_text = extracted_text
                        st.session_state.diacritized_text = diacritized_text
                        # Clear old sarf analysis when processing new text
                        st.session_state.sarf_results = None
                        st.session_state.sarf_word = ""

            # Render results if we have diacritized text in session state
            if st.session_state.diacritized_text:
                extracted_text = st.session_state.extracted_text
                diacritized_text = st.session_state.diacritized_text
                
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
                
                # Morphological and sarf analysis section
                st.markdown("---")
                with st.container(border=True):
                    st.subheader("Morphological and sarf analysis")
                    
                    # Clean and extract Arabic words from the diacritized text
                    arabic_word_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
                    words = arabic_word_pattern.findall(diacritized_text)
                    
                    if words:
                        # Deduplicate words while preserving order
                        seen = set()
                        unique_words = [w for w in words if not (w in seen or seen.add(w))]
                        
                        selected_word = st.selectbox(
                            "Select a word to analyze from diacritized text",
                            options=unique_words
                        )
                        
                        custom_word = st.text_input("Or type a custom Arabic word to analyze")
                        word_to_analyze = custom_word.strip() if custom_word.strip() else selected_word
                    else:
                        word_to_analyze = st.text_input("Type an Arabic word to analyze")
                    
                    if word_to_analyze:
                        # If the selected word has changed, clear past analysis to avoid stale results
                        if st.session_state.sarf_word != word_to_analyze:
                            st.session_state.sarf_word = word_to_analyze
                            st.session_state.sarf_results = None
                        
                        # Perform Step 3 LOCALLY
                        try:
                            # farasa_stemmer.stem returns the stem
                            stemmed_word = farasa_stemmer.stem(word_to_analyze)
                            # Clean the stem to letters only for root display
                            clean_stem = re.sub(r'[^\u0621-\u064A]', '', stemmed_word)
                            root_display = " - ".join(list(clean_stem))
                        except Exception as e:
                            st.error(f"Farasa Stemmer Error: {e}")
                            clean_stem = word_to_analyze
                            root_display = "Unknown"
                        
                        # Translate the word and root using deep_translator
                        try:
                            word_trans = GoogleTranslator(source='ar', target='en').translate(word_to_analyze)
                            root_trans = GoogleTranslator(source='ar', target='en').translate(clean_stem)
                        except Exception:
                            word_trans = "N/A"
                            root_trans = "N/A"
                        
                        st.markdown(f"Analyzing word: **{word_to_analyze}** (Translation: *{word_trans}*)")
                        st.info(f"🌱 **Extracted local root/stem:** {root_display} (Translation: *{root_trans}*)")
                        
                        # Perform Step 4 via Gemini
                        api_key = get_gemini_api_key()
                        if not api_key:
                            st.warning("⚠️ Gemini API key is missing. Please configure it to enable Sarf table generation.")
                        
                        if st.button("Generate sarf analysis", disabled=(not api_key)):
                            with st.spinner("Generating Sarf table via Gemini..."):
                                try:
                                    sarf_data = get_sarf_table(word_to_analyze, clean_stem)
                                    st.session_state.sarf_results = sarf_data
                                except Exception as e:
                                    st.error(f"Gemini Sarf Analysis failed: {e}")
                                    st.session_state.sarf_results = None
                        
                        # Display Sarf results if available
                        if st.session_state.sarf_results and st.session_state.sarf_word == word_to_analyze:
                            sarf_data = st.session_state.sarf_results
                            
                            # Layout metrics and table
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric(label="Pattern / form (الوزن)", value=sarf_data.get('wazn', 'N/A'))
                            with col_m2:
                                st.metric(label="Past tense (الماضي)", value=sarf_data.get('madi', 'N/A'))
                            with col_m3:
                                st.metric(label="Present tense (المضارع)", value=sarf_data.get('mudari', 'N/A'))
                                
                            # Complete paradigm table with English translations
                            st.markdown("#### Morphological paradigm table")
                            
                            # Translate the paradigm words (excluding wazn, which already includes English description)
                            sarf_words_to_translate = [
                                sarf_data.get('madi', ''),
                                sarf_data.get('mudari', ''),
                                sarf_data.get('masdar', ''),
                                sarf_data.get('ism_faail', ''),
                                sarf_data.get('ism_mafool', '')
                            ]
                            paradigm_translations = translate_words(sarf_words_to_translate)
                            
                            sarf_details = {
                                "Grammatical element": [
                                    "Pattern / form (الوزن)",
                                    "Past tense 3rd person (الماضي)",
                                    "Present tense 3rd person (المضارع)",
                                    "Verbal noun (المصدر)",
                                    "Active participle (اسم الفاعل)",
                                    "Passive participle (اسم المفعول)"
                                ],
                                "Arabic word (with tashkeel)": [
                                    sarf_data.get('wazn', 'N/A'),
                                    sarf_data.get('madi', 'N/A'),
                                    sarf_data.get('mudari', 'N/A'),
                                    sarf_data.get('masdar', 'N/A'),
                                    sarf_data.get('ism_faail', 'N/A'),
                                    sarf_data.get('ism_mafool', 'N/A')
                                ],
                                "English translation": [
                                    "N/A",  # wazn is descriptive
                                    paradigm_translations[0],
                                    paradigm_translations[1],
                                    paradigm_translations[2],
                                    paradigm_translations[3],
                                    paradigm_translations[4]
                                ]
                            }
                            df_sarf = pd.DataFrame(sarf_details)
                            st.table(df_sarf)
else:
    st.info("Please upload one or more manga pages to begin.")
