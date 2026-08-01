import streamlit as st
from streamlit_cropper import st_cropper
import easyocr
import pypdf
import pdf2image
import io

import base64
from datetime import datetime

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

# SQLite Database Initialization & Helper Functions
def init_db():
    """Initialize SQLite database for storing Arabic study logs."""
    conn = sqlite3.connect("arabic_study_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_filename TEXT,
            image_base64 TEXT,
            tashkeel_text TEXT,
            full_translation TEXT,
            verbs_json TEXT,
            nouns_json TEXT,
            particles_json TEXT,
            deep_sarf_json TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on load
init_db()

def save_study_entry(source_filename, image_base64, tashkeel_text, full_translation, verbs, nouns, particles, deep_sarf):
    """Insert a study entry into SQLite database."""
    conn = sqlite3.connect("arabic_study_history.db")
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO study_logs 
        (timestamp, source_filename, image_base64, tashkeel_text, full_translation, verbs_json, nouns_json, particles_json, deep_sarf_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        source_filename,
        image_base64,
        tashkeel_text,
        full_translation,
        json.dumps(verbs, ensure_ascii=False),
        json.dumps(nouns, ensure_ascii=False),
        json.dumps(particles, ensure_ascii=False),
        json.dumps(deep_sarf, ensure_ascii=False)
    ))
    conn.commit()
    conn.close()

def get_all_study_entries():
    """Retrieve all study entries from SQLite database."""
    conn = sqlite3.connect("arabic_study_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, source_filename, image_base64, tashkeel_text, full_translation, verbs_json, nouns_json, particles_json, deep_sarf_json
        FROM study_logs
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_study_entry(entry_id: int):
    """Delete a study log entry by ID from SQLite database."""
    conn = sqlite3.connect("arabic_study_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM study_logs WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()

def image_to_base64(pil_img: Image.Image) -> str:
    """Convert a PIL Image to a Base64 PNG string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

# Pydantic Schemas for Master Gemini JSON Response
class WordMeaningItem(BaseModel):
    word: str = Field(description="The diacritized Arabic word (with full Tashkeel)")
    meaning: str = Field(description="Contextual English translation/meaning of this word in the sentence")

class ParticleItem(BaseModel):
    word: str = Field(description="The diacritized particle (with full Tashkeel, e.g. 'فِي', 'إِنَّ')")
    type: str = Field(description="Particle category in Arabic & English (e.g. 'حَرْف جَرّ (Genitive Particle)')")
    meaning: str = Field(description="Contextual English meaning (e.g. 'In / At')")
    effect: str = Field(description="Grammatical effect / الأَثَر الإِعْرَابِي (e.g. 'Forces following noun into Majroor state')")

class OCRTashkeelResult(BaseModel):
    tashkeel_text: str = Field(description="Full diacritized Arabic text with complete Tashkeel")
    full_translation: str = Field(description="Complete English translation of the cropped sentence/passage")
    verbs: list[WordMeaningItem] = Field(description="List of verbs with meanings (including verbs with attached pronouns like 'خُذْنِي')")
    nouns: list[WordMeaningItem] = Field(description="List of nouns, pronouns, and adjectives with meanings")
    particles: list[ParticleItem] = Field(description="List of particles with type, meaning, and effect")

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

MASTER_GEMINI_SYSTEM_PROMPT = (
    "You are an expert Arabic grammarian, OCR, and computational linguist tool.\n"
    "Your task is to transcribe/process raw Arabic text, fix minor typos, add full, grammatically precise Tashkeel (diacritics), "
    "provide a complete, fluent English translation of the sentence/passage, and analyze all words into the 3 distinct pillars of Arabic grammar:\n"
    "Verbs (أَفْعَال), Nouns (أَسْمَاء), and Particles (حُرُوف).\n\n"
    "CRITICAL CLASSIFICATION & STRUCTURE RULES:\n"
    "1. 'tashkeel_text': The full diacritized Arabic text with complete vowels.\n"
    "2. 'full_translation': Complete, accurate English translation of the entire cropped sentence/passage.\n"
    "3. 'verbs': List of objects {\"word\": \"<diacritized_verb>\", \"meaning\": \"<english_meaning>\"}. EXPLICIT RULE: Verbs with attached object pronouns (e.g., 'خُذْنِي', 'لَسْتُ', 'أَقُولُهُ', 'سَأَلْتُكَ') MUST be placed under 'verbs'.\n"
    "4. 'nouns': List of objects {\"word\": \"<diacritized_noun>\", \"meaning\": \"<english_meaning>\"} for all nouns, pronouns, and adjectives.\n"
    "5. 'particles': List of objects {\"word\": \"<diacritized_particle>\", \"type\": \"<category>\", \"meaning\": \"<meaning>\", \"effect\": \"<grammatical_effect>\"} for all prepositions, conjunctions, and particles (حُرُوف)."
)

def gemini_tashkeel(raw_arabic_text: str) -> dict:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=raw_arabic_text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OCRTashkeelResult,
            system_instruction=MASTER_GEMINI_SYSTEM_PROMPT,
        )
    )
    try:
        data = json.loads(response.text.strip())
        return data
    except Exception as e:
        return {
            "tashkeel_text": raw_arabic_text,
            "full_translation": "",
            "verbs": [],
            "nouns": [],
            "particles": []
        }

def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel) from text."""
    pattern = re.compile(r'[\u064B-\u0652]')
    return pattern.sub('', text)

def gemini_vision_ocr_tashkeel(pil_image: Image.Image) -> dict:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=[pil_image, "Perform OCR, add full Tashkeel, provide full translation, and classify all verbs, nouns, and particles in this image."],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OCRTashkeelResult,
            system_instruction=MASTER_GEMINI_SYSTEM_PROMPT,
        )
    )
    try:
        data = json.loads(response.text.strip())
        return data
    except Exception as e:
        return {
            "tashkeel_text": "",
            "full_translation": "",
            "verbs": [],
            "nouns": [],
            "particles": []
        }

def fit_image_to_max_width(image: Image.Image, max_width: int = 700) -> Image.Image:
    """
    If the image width exceeds max_width (e.g. container size in #root),
    scale it down proportionally so it fits cleanly on screen.
    """
    W, H = image.size
    if W > max_width:
        new_w = max_width
        new_h = int(max_width * (H / W))
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        return image.resize((new_w, new_h), resample=resample_filter)
    return image

def get_zoomed_viewport(image: Image.Image, zoom_factor: float, offset_x: float = 0.0, offset_y: float = 0.0) -> Image.Image:
    """
    Crop the image based on zoom factor and normalized offsets (-1.0 to 1.0),
    then resize it to target dimensions (W * zoom_factor, H * zoom_factor).
    Allows navigating every region of the page when zoomed in.
    """
    if zoom_factor == 1.0 and offset_x == 0.0 and offset_y == 0.0:
        return image
        
    W, H = image.size
    
    # Calculate viewport sub-region size
    sub_w = W / zoom_factor
    sub_h = H / zoom_factor
    
    # Max shift allowed in X and Y
    max_shift_x = (W - sub_w) / 2
    max_shift_y = (H - sub_h) / 2
    
    # Center coordinates
    center_x = W / 2 + offset_x * max_shift_x
    center_y = H / 2 + offset_y * max_shift_y
    
    # Viewport boundaries
    left = max(0.0, center_x - sub_w / 2)
    top = max(0.0, center_y - sub_h / 2)
    right = min(float(W), left + sub_w)
    bottom = min(float(H), top + sub_h)
    
    cropped = image.crop((int(left), int(top), int(right), int(bottom)))
    
    target_w = int(W * zoom_factor)
    target_h = int(H * zoom_factor)
    
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
        
    return cropped.resize((target_w, target_h), resample=resample_filter)

@st.cache_data
def render_pdf_page(pdf_bytes: bytes, page: int) -> Image.Image:
    """Render a single page of a PDF bytes to a PIL Image."""
    images = pdf2image.convert_from_bytes(pdf_bytes, first_page=page, last_page=page)
    if images:
        return images[0]
    raise ValueError(f"Could not render page {page} of PDF.")

# Structured output schemas for Sarf analysis
class VerbSarfTable(BaseModel):
    wazn: str = Field(description="Triliteral Form/Pattern, e.g., 'Form IV / أَفْعَلَ'")
    madi: str = Field(description="3rd person past tense (الْمَاضِي), fully diacritized")
    mudari: str = Field(description="3rd person present tense (الْمُضَارِع), fully diacritized")
    amr: str = Field(description="Imperative command form (الأَمْر), fully diacritized")
    masdar: str = Field(description="Verbal Noun (الْمَصْدَر), fully diacritized")
    ism_faail: str = Field(description="Active Participle (اسْم الْفَاعِل), fully diacritized")
    ism_mafool: str = Field(description="Passive Participle (اسْم الْمَفْعُول), fully diacritized")

class NounSarfTable(BaseModel):
    noun_type: str = Field(description="Classification, either 'Derived / مُشْتَق' or 'Primary / جَامِد'")
    category: str = Field(description="Noun class, e.g., 'Place/Ism Makan', 'Tool/Ism Aalah', 'Agent', 'Concept', etc.")
    wazn: str = Field(description="Pattern/Weight, e.g., 'مَفْعَل' or 'فَاعِل', fully diacritized")
    singular: str = Field(description="Singular form (المُفْرَد), fully diacritized")
    dual: str = Field(description="Dual form (المُثَنَّى), fully diacritized")
    plural: str = Field(description="Broken or Sound Plural (الجَمْع), fully diacritized")
    root_verb: str = Field(description="Associated 3rd-person past verb (if derived), fully diacritized. If not derived or not applicable, write N/A")

def get_verb_analysis(word: str, root: str) -> dict:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    system_prompt = (
        "You are an expert Arabic grammarian and morphologist. Analyze the provided verb and its root, "
        "and output its morphological analysis table as a JSON object matching the requested schema. "
        "Ensure all Arabic words in the response are fully and precisely diacritized."
    )
    
    prompt = (
        f"Analyze the morphological structure (Sarf) of the Arabic verb '{word}' "
        f"which has the root '{root}'.\n"
        f"Provide the pattern/form (wazn), past tense (madi), present tense (mudari), "
        f"imperative (amr), verbal noun (masdar), active participle (ism_faail), and passive participle (ism_mafool) "
        f"for this verb's paradigm."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerbSarfTable,
            system_instruction=system_prompt,
        )
    )
    
    data = json.loads(response.text.strip())
    return data

def get_noun_analysis(word: str, root: str) -> dict:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    system_prompt = (
        "You are an expert Arabic grammarian and morphologist. Analyze the provided noun and its root, "
        "and output its morphological analysis as a JSON object matching the requested schema. "
        "Ensure all Arabic words in the response are fully and precisely diacritized."
    )
    
    prompt = (
        f"Analyze the morphological structure of the Arabic noun '{word}' "
        f"which has the root '{root}'.\n"
        f"Provide the noun type (noun_type), category, pattern/weight (wazn), "
        f"singular, dual, plural, and associated root verb (root_verb)."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=NounSarfTable,
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

# Inject WebFont definitions for KFGQPC Uthman Taha Naskh & HAFS and custom CSS for Arabic rendering
st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Scheherazade+New:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* WebFont definitions for official King Fahd Complex Uthmani fonts with punctuation fallback */
        @font-face {
            font-family: 'KFGQPC Uthman Taha Naskh';
            src: url('https://cdn.jsdelivr.net/npm/kfgqpc-uthmanic-script-hafs-regular@1.0.0/arabic.otf') format('opentype');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
            unicode-range: U+0000-060B, U+060D-FFFF; /* Exclude U+060C (Arabic comma ،) to avoid missing glyph black dot */
        }
        @font-face {
            font-family: 'KFGQPC Uthmanic Script HAFS';
            src: url('https://cdn.jsdelivr.net/npm/kfgqpc-uthmanic-script-hafs-regular@1.0.0/arabic.otf') format('opentype');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
            unicode-range: U+0000-060B, U+060D-FFFF; /* Exclude U+060C (Arabic comma ،) to avoid missing glyph black dot */
        }

        /* Define Arabic text class using requested KFGQPC Uthman Taha Naskh font stack with punctuation fallbacks */
        .arabic-text {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
        }
        .arabic-large {
            font-size: 32px !important;
            line-height: 1.6 !important;
        }
        .arabic-medium {
            font-size: 24px !important;
            line-height: 1.5 !important;
        }
        .arabic-root {
            font-size: 30px !important;
            font-weight: bold !important;
            letter-spacing: 4px !important;
        }
        
        /* 1. Global Table & TabPanel Paradigm Tables Styling */
        div[role="tabpanel"] table,
        div[data-testid="stTable"] table,
        table {
            width: 100% !important;
        }

        div[role="tabpanel"] table th,
        div[data-testid="stTable"] th,
        table th {
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 8px 12px !important;
            background-color: rgba(200, 200, 200, 0.15) !important;
        }

        div[role="tabpanel"] table td,
        div[data-testid="stTable"] td,
        div[data-testid="stDataFrame"] td,
        .stTable td,
        table td,
        div[role="gridcell"] {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 36px !important;
            font-weight: 700 !important;
            line-height: 1.35 !important;
            padding: 6px 12px !important;
            direction: rtl !important;
        }

        /* 2. Text areas & inputs using requested font stack */
        textarea, input {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 22px !important;
            line-height: 1.5 !important;
        }

        /* 3. Streamlit native metric values */
        div[data-testid="stMetricValue"] {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 28px !important;
            font-weight: bold !important;
            line-height: 1.3 !important;
            direction: rtl !important;
            text-align: right !important;
            color: #1E88E5 !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 14px !important;
            font-weight: 600 !important;
        }

        /* 4. Selectbox option dropdown font size */
        div[role="listbox"] div, div[data-baseweb="select"] {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 20px !important;
        }
    </style>
""")

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

# Initialize session state variables
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'diacritized_text' not in st.session_state:
    st.session_state.diacritized_text = ""
if 'full_translation' not in st.session_state:
    st.session_state.full_translation = ""
if 'current_file' not in st.session_state:
    st.session_state.current_file = ""
if 'sarf_results' not in st.session_state:
    st.session_state.sarf_results = None
if 'sarf_word' not in st.session_state:
    st.session_state.sarf_word = ""
if 'verb_results' not in st.session_state:
    st.session_state.verb_results = None
if 'last_analyzed_verb' not in st.session_state:
    st.session_state.last_analyzed_verb = ""
if 'noun_results' not in st.session_state:
    st.session_state.noun_results = None
if 'last_analyzed_noun' not in st.session_state:
    st.session_state.last_analyzed_noun = ""
if 'processed_by' not in st.session_state:
    st.session_state.processed_by = ""
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 1.0
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 1.0
if 'pan_x' not in st.session_state:
    st.session_state.pan_x = 0.0
if 'pan_y' not in st.session_state:
    st.session_state.pan_y = 0.0
if 'verbs' not in st.session_state:
    st.session_state.verbs = []
if 'nouns' not in st.session_state:
    st.session_state.nouns = []
if 'particles' not in st.session_state:
    st.session_state.particles = []

def process_text_for_display(text):
    """Reshape and apply bidi algorithm for proper RTL display in Streamlit widgets."""
    if not text.strip():
        return ""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# Sidebar: Engine Selection
st.sidebar.title("Settings")

ocr_mode = st.sidebar.radio(
    "⚙️ OCR & Diacritization Engine",
    [
        "🟢 Hybrid Mode (EasyOCR + Gemini Text)",
        "🟣 Full Gemini Vision"
    ]
)

if ocr_mode == "🟢 Hybrid Mode (EasyOCR + Gemini Text)":
    st.sidebar.info("💡 **Tip:** Hybrid Mode uses local EasyOCR first ($0 cost), then sends raw text to Gemini 2.5 Flash-Lite to fix typos, add Tashkeel, and classify verbs/nouns.")
else:
    st.sidebar.info("💡 **Tip:** Full Gemini Vision mode sends the image crop directly to Gemini, allowing it to perform OCR, Tashkeel, and word classification in a single step.")

# 1. File Upload Limit (max 10 files)
uploaded_files = st.file_uploader("Upload Manga Pages/PDFs (Max 10)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

if uploaded_files:
    # Enforce limit of 10 files
    if len(uploaded_files) > 10:
        st.warning(f"⚠️ You have uploaded {len(uploaded_files)} files. Only the first 10 will be processed to stay within limits.")
        uploaded_files = uploaded_files[:10]
        
    # 2. Image Selection & Cropping
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.selectbox("Select File to Crop", file_names)
    
    # Find the selected file object
    selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
    
    is_pdf = selected_file_name.lower().endswith('.pdf')
    
    # Initialize pdf_page in session state if not present
    if 'pdf_page' not in st.session_state:
        st.session_state.pdf_page = 1
    if 'last_pdf_page' not in st.session_state:
        st.session_state.last_pdf_page = 1

    # Reset session state if the selected file name changes
    if st.session_state.current_file != selected_file_name:
        st.session_state.current_file = selected_file_name
        st.session_state.extracted_text = ""
        st.session_state.diacritized_text = ""
        st.session_state.sarf_results = None
        st.session_state.sarf_word = ""
        st.session_state.verb_results = None
        st.session_state.last_analyzed_verb = ""
        st.session_state.noun_results = None
        st.session_state.last_analyzed_noun = ""
        st.session_state.processed_by = ""
        st.session_state.zoom_factor = 1.0
        st.session_state.zoom_level = 1.0
        st.session_state.pan_x = 0.0
        st.session_state.pan_y = 0.0
        st.session_state.verbs = []
        st.session_state.nouns = []
        st.session_state.particles = []
        st.session_state.pdf_page = 1
        st.session_state.last_pdf_page = 1

    # PDF Navigation in Sidebar
    if is_pdf:
        try:
            pdf_bytes = selected_file.getvalue()
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(pdf_reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            num_pages = 1
            
        if st.session_state.pdf_page > num_pages:
            st.session_state.pdf_page = num_pages
        if st.session_state.pdf_page < 1:
            st.session_state.pdf_page = 1

        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 PDF Page Navigation")
        
        col_prev, col_next = st.sidebar.columns(2)
        
        if col_prev.button("⬅️ Previous", disabled=(st.session_state.pdf_page <= 1)):
            st.session_state.pdf_page -= 1
            
        if col_next.button("Next ➡️", disabled=(st.session_state.pdf_page >= num_pages)):
            st.session_state.pdf_page += 1
            
        st.sidebar.number_input(
            f"Page (1 of {num_pages})",
            min_value=1,
            max_value=num_pages,
            step=1,
            key="pdf_page"
        )
        
        if st.session_state.pdf_page != st.session_state.last_pdf_page:
            st.session_state.extracted_text = ""
            st.session_state.diacritized_text = ""
            st.session_state.sarf_results = None
            st.session_state.sarf_word = ""
            st.session_state.verb_results = None
            st.session_state.last_analyzed_verb = ""
            st.session_state.noun_results = None
            st.session_state.last_analyzed_noun = ""
            st.session_state.processed_by = ""
            st.session_state.zoom_factor = 1.0
            st.session_state.zoom_level = 1.0
            st.session_state.pan_x = 0.0
            st.session_state.pan_y = 0.0
            st.session_state.verbs = []
            st.session_state.nouns = []
            st.session_state.particles = []
            st.session_state.last_pdf_page = st.session_state.pdf_page

    # Load image (from PIL directly or render from PDF)
    img = None
    if is_pdf:
        try:
            img = render_pdf_page(selected_file.getvalue(), st.session_state.pdf_page)
        except Exception as e:
            if "poppler" in str(e).lower() or "pdfinfo" in str(e).lower():
                st.error(
                    "⚠️ **System dependency error:** `poppler` is required to render PDF pages but was not found on your system.\n\n"
                    "**How to fix:**\n"
                    "1. Download Poppler for Windows (e.g., from [conda-forge](https://anaconda.org/conda-forge/poppler) or [GitHub Releases](https://github.com/oschwartz10612/poppler-windows/releases)).\n"
                    "2. Extract the archive and add the `bin` folder to your system PATH.\n"
                    "3. Restart the Streamlit app."
                )
            else:
                st.error(f"Error rendering PDF page: {e}")
    else:
        try:
            img = Image.open(selected_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    if img is None:
        st.stop()
    
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
            /* Enable native scrollbars (horizontal and vertical) when cropper canvas expands */
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
    
    def reset_view_callback():
        st.session_state.zoom_level = 1.0
        st.session_state.pan_x = 0.0
        st.session_state.pan_y = 0.0

    with col1:
        st.subheader("1. Crop Speech Bubble")
        st.markdown("Draw a rectangle over the speech bubble you want to extract text from.")
        
        # Zoom and Pan controls
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            zoom_val = st.slider("🔍 Zoom Level", min_value=1.0, max_value=3.0, value=st.session_state.zoom_level, step=0.25, key="zoom_level")
        with col_ctrl2:
            st.write("")  # spacing
            st.write("")
            st.button("🔄 Reset View", use_container_width=True, on_click=reset_view_callback)
                
        col_pan1, col_pan2 = st.columns(2)
        disabled_pan = (zoom_val == 1.0)
        with col_pan1:
            pan_x_val = st.slider("↔️ Horizontal Pan", min_value=-1.0, max_value=1.0, value=st.session_state.pan_x, step=0.02, disabled=disabled_pan, key="pan_x")
        with col_pan2:
            pan_y_val = st.slider("↕️ Vertical Pan", min_value=-1.0, max_value=1.0, value=st.session_state.pan_y, step=0.02, disabled=disabled_pan, key="pan_y")

        # Fit image to container max width (700px) so giant images/PDFs fit cleanly in #root at 1.0x
        base_img = fit_image_to_max_width(img, max_width=700)

        # Obtain zoomed image, scaled up proportionally to magnify details
        zoomed_img = get_zoomed_viewport(base_img, zoom_val, pan_x_val, pan_y_val)

        # Inject dynamic CSS: unclip parent wrappers & set explicit iframe dimensions
        st.markdown(f"""
            <style>
                /* Unclip Streamlit column wrappers so scrollbars can render */
                div[data-testid="column"]:nth-of-type(1) {{
                    overflow: visible !important;
                }}
                div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"] {{
                    overflow: visible !important;
                }}
                div[data-testid="column"]:nth-of-type(1) div[data-testid="stElementContainer"],
                div[data-testid="column"]:nth-of-type(1) div.stElementContainer,
                div[data-testid="stCustomComponentV1"],
                div.stCropper {{
                    overflow: auto !important;
                    max-height: 700px !important;
                    max-width: 100% !important;
                    border: 2px solid #e0e0e0 !important;
                    border-radius: 8px !important;
                    background-color: #fafafa !important;
                }}
                /* Force component iframe to expand to zoomed_img dimensions */
                div[data-testid="column"]:nth-of-type(1) iframe {{
                    width: {zoomed_img.width}px !important;
                    height: {zoomed_img.height}px !important;
                    min-width: {zoomed_img.width}px !important;
                    min-height: {zoomed_img.height}px !important;
                }}
            </style>
        """, unsafe_allow_html=True)

        # Interactive cropping tool on the zoomed image (key includes zoom and pan values to force remount)
        cropper_key = f"cropper_{selected_file_name}_z{zoom_val}_px{pan_x_val}_py{pan_y_val}"
        if is_pdf:
            cropper_key += f"_p{st.session_state.pdf_page}"
            
        cropped_img = st_cropper(
            zoomed_img, 
            realtime_update=True, 
            box_color='#FF0000', 
            aspect_ratio=None,
            return_type='image',
            should_resize_image=False,
            key=cropper_key
        )
        
        # Map cropped area back to full native image resolution for Step 2
        if cropped_img is not None and zoomed_img is not None and zoomed_img.width > 0:
            scale_ratio = img.width / zoomed_img.width
            orig_w = max(1, int(cropped_img.width * scale_ratio))
            orig_h = max(1, int(cropped_img.height * scale_ratio))
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            cropped_img = cropped_img.resize((orig_w, orig_h), resample=resample_filter)
        
    with col2:
        st.subheader("2. Extracted and diacritized text")
        
        if cropped_img:
            st.image(cropped_img, caption="Cropped Area", width=150)
            
            # User feedback badge
            if st.session_state.processed_by:
                st.info(f"⚡ **Last processed by:** {st.session_state.processed_by}")
            else:
                st.caption(f"Ready to process using: **{ocr_mode}**")
            
            if st.button("Process & Add Vowels", type="primary"):
                with st.spinner("Extracting text and applying Tashkeel..."):
                    try:
                        if ocr_mode == "🟢 Hybrid Mode (EasyOCR + Gemini Text)":
                            # Convert PIL image to numpy array for EasyOCR
                            cropped_array = np.array(cropped_img)
                            results = reader.readtext(cropped_array, detail=0, paragraph=True)
                            extracted_text = " ".join(results)
                            
                            if not extracted_text.strip():
                                st.error("No text detected in the selected area. Please try cropping a clearer area or a different bubble.")
                                st.session_state.extracted_text = ""
                                st.session_state.diacritized_text = ""
                                st.session_state.processed_by = ""
                                st.session_state.verbs = []
                                st.session_state.nouns = []
                                st.session_state.particles = []
                                st.session_state.sarf_results = None
                                st.session_state.sarf_word = ""
                                st.session_state.verb_results = None
                                st.session_state.last_analyzed_verb = ""
                                st.session_state.noun_results = None
                                st.session_state.last_analyzed_noun = ""
                            else:
                                # Call Gemini 2.5 Flash-Lite
                                gemini_res = gemini_tashkeel(extracted_text)
                                diacritized_text = gemini_res.get("tashkeel_text", extracted_text)
                                full_trans = gemini_res.get("full_translation", "")
                                verbs_list = gemini_res.get("verbs", [])
                                nouns_list = gemini_res.get("nouns", [])
                                particles_list = gemini_res.get("particles", [])
                                
                                st.session_state.extracted_text = extracted_text
                                st.session_state.diacritized_text = diacritized_text
                                st.session_state.full_translation = full_trans
                                st.session_state.processed_by = ocr_mode
                                st.session_state.verbs = verbs_list
                                st.session_state.nouns = nouns_list
                                st.session_state.particles = particles_list
                                st.session_state.sarf_results = None
                                st.session_state.sarf_word = ""
                                st.session_state.verb_results = None
                                st.session_state.last_analyzed_verb = ""
                                st.session_state.noun_results = None
                                st.session_state.last_analyzed_noun = ""
                                st.rerun()
                        else:
                            # Option 2: Full Gemini Vision
                            gemini_res = gemini_vision_ocr_tashkeel(cropped_img)
                            diacritized_text = gemini_res.get("tashkeel_text", "")
                            full_trans = gemini_res.get("full_translation", "")
                            verbs_list = gemini_res.get("verbs", [])
                            nouns_list = gemini_res.get("nouns", [])
                            particles_list = gemini_res.get("particles", [])
                            
                            if not diacritized_text.strip():
                                st.error("Gemini Vision was unable to read any text from the cropped area. Please try a different area.")
                                st.session_state.extracted_text = ""
                                st.session_state.diacritized_text = ""
                                st.session_state.full_translation = ""
                                st.session_state.processed_by = ""
                                st.session_state.verbs = []
                                st.session_state.nouns = []
                                st.session_state.particles = []
                                st.session_state.sarf_results = None
                                st.session_state.sarf_word = ""
                                st.session_state.verb_results = None
                                st.session_state.last_analyzed_verb = ""
                                st.session_state.noun_results = None
                                st.session_state.last_analyzed_noun = ""
                            else:
                                # Strip diacritics to get raw extracted text
                                extracted_text = strip_tashkeel(diacritized_text)
                                
                                st.session_state.extracted_text = extracted_text
                                st.session_state.diacritized_text = diacritized_text
                                st.session_state.full_translation = full_trans
                                st.session_state.processed_by = ocr_mode
                                st.session_state.verbs = verbs_list
                                st.session_state.nouns = nouns_list
                                st.session_state.particles = particles_list
                                st.session_state.sarf_results = None
                                st.session_state.sarf_word = ""
                                st.session_state.verb_results = None
                                st.session_state.last_analyzed_verb = ""
                                st.session_state.noun_results = None
                                st.session_state.last_analyzed_noun = ""
                                st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

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
                if st.session_state.full_translation:
                    st.success(st.session_state.full_translation)
                else:
                    try:
                        translated_text = GoogleTranslator(source='ar', target='en').translate(diacritized_text)
                        st.success(translated_text)
                    except Exception as e:
                        st.warning(f"Translation failed: {e}")
                
                # Extra: Render with native HTML for larger, cleaner reading
                st.markdown("---")
                st.markdown("### Native Render (Large Font)")
                
                # We render the original non-bidi'ed text because browsers handle standard Arabic 
                # perfectly when dir="rtl" is specified. This provides the most natural look.
                html_str = f"""
                <div dir="rtl" class="arabic-text">
                    <div style="font-size: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px; background-color: rgba(200, 200, 200, 0.1); text-align: right;">
                        <strong>Raw:</strong> {extracted_text}
                    </div>
                    <div class="arabic-large" style="padding: 15px; border: 2px solid #4CAF50; border-radius: 8px; color: #2E7D32; background-color: rgba(76, 175, 80, 0.05); text-align: right;">
                        <strong>Tashkeel:</strong> {diacritized_text}
                    </div>
                </div>
                """
                st.markdown(html_str, unsafe_allow_html=True)
                
                # Morphological and sarf analysis section
                st.markdown("---")
                with st.container(border=True):
                    st.subheader("Morphological and sarf analysis")
                    
                    detected_verbs = st.session_state.verbs
                    detected_nouns = st.session_state.nouns
                    detected_particles = st.session_state.particles

                    # Setup tabs for all 3 pillars of Arabic grammar
                    tab1, tab2, tab3 = st.tabs(["⚙️ Verbs (فِعْل)", "🏷️ Nouns (اسْم)", "📌 Particles (حُرُوف)"])

                    with tab1:
                        st.subheader("Verb Analysis")
                        
                        # Populate verb selectbox with fallback to custom input
                        if detected_verbs:
                            verb_options = []
                            verb_word_map = {}
                            for v in detected_verbs:
                                if isinstance(v, dict):
                                    w = v.get("word", "")
                                    m = v.get("meaning", "")
                                elif hasattr(v, 'word'):
                                    w = getattr(v, 'word', "")
                                    m = getattr(v, 'meaning', "")
                                else:
                                    w = str(v)
                                    m = ""
                                disp = f"{w} ({m})" if m else w
                                verb_options.append(disp)
                                verb_word_map[disp] = w

                            selected_verb_disp = st.selectbox(
                                "Select a word to analyze as a Verb",
                                options=verb_options,
                                key="verb_select"
                            )
                            custom_verb = st.text_input("Or type a custom Arabic verb to analyze", key="verb_custom")
                            verb_to_analyze = custom_verb.strip() if custom_verb.strip() else verb_word_map.get(selected_verb_disp, selected_verb_disp)
                        else:
                            st.info("No verbs detected in this crop. You can enter one manually below.")
                            verb_to_analyze = st.text_input("Type an Arabic verb to analyze", key="verb_custom_only")
                            
                        if verb_to_analyze:
                            # 1. Run local Farasa Stemmer first to get the root
                            try:
                                stemmed_verb = farasa_stemmer.stem(verb_to_analyze)
                                clean_verb_root = re.sub(r'[^\u0621-\u064A]', '', stemmed_verb)
                                verb_root_display = " - ".join(list(clean_verb_root))
                            except Exception as e:
                                st.error(f"Farasa Stemmer Error: {e}")
                                clean_verb_root = verb_to_analyze
                                verb_root_display = "Unknown"
                                
                            try:
                                verb_trans = GoogleTranslator(source='ar', target='en').translate(verb_to_analyze)
                                verb_root_trans = GoogleTranslator(source='ar', target='en').translate(clean_verb_root)
                            except Exception:
                                verb_trans = "N/A"
                                verb_root_trans = "N/A"
                                
                            st.markdown(f"""
                            <div style="font-size: 18px; margin-bottom: 10px;">
                                Analyzing verb: <span class="arabic-text arabic-medium" style="font-weight: bold; color: #1E88E5;">{verb_to_analyze}</span> (Translation: <em>{verb_trans}</em>)
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.info(f"🌱 Extracted Local Root: {verb_root_display} (Translation: {verb_root_trans})")
                            
                            # Clear results if the verb changes
                            if st.session_state.last_analyzed_verb != verb_to_analyze:
                                st.session_state.verb_results = None
                                
                            api_key = get_gemini_api_key()
                            if not api_key:
                                st.warning("⚠️ Gemini API key is missing. Please configure it to enable Verb analysis.")
                                
                            if st.button("Analyze Verb via Gemini", key="btn_analyze_verb", disabled=(not api_key)):
                                with st.spinner("Analyzing verb paradigm..."):
                                    try:
                                        verb_data = get_verb_analysis(verb_to_analyze, clean_verb_root)
                                        st.session_state.verb_results = verb_data
                                        st.session_state.last_analyzed_verb = verb_to_analyze
                                    except Exception as e:
                                        st.error(f"Gemini Verb Analysis failed: {e}")
                                        st.session_state.verb_results = None
                                        
                            # Render results
                            if st.session_state.verb_results and st.session_state.last_analyzed_verb == verb_to_analyze:
                                v_data = st.session_state.verb_results
                                
                                # Layout metrics
                                col_v1, col_v2, col_v3 = st.columns(3)
                                with col_v1:
                                    st.metric(label="Pattern / Form (الوزن)", value=v_data.get('wazn', 'N/A'))
                                with col_v2:
                                    st.metric(label="Past Tense (الماضي)", value=v_data.get('madi', 'N/A'))
                                with col_v3:
                                    st.metric(label="Present Tense (المضارع)", value=v_data.get('mudari', 'N/A'))
                                    
                                st.markdown("#### Morphological paradigm table")
                                words_to_translate = [
                                    v_data.get('madi', ''),
                                    v_data.get('mudari', ''),
                                    v_data.get('amr', ''),
                                    v_data.get('masdar', ''),
                                    v_data.get('ism_faail', ''),
                                    v_data.get('ism_mafool', '')
                                ]
                                translations = translate_words(words_to_translate)
                                
                                verb_df = pd.DataFrame({
                                    "Grammatical Element": [
                                        "Pattern / Form (الوزن)",
                                        "Past Tense (الْمَاضِي)",
                                        "Present Tense (الْمُضَارِع)",
                                        "Imperative Form (الأَمْر)",
                                        "Verbal Noun (الْمَصْدَر)",
                                        "Active Participle (اسْم الْفَاعِل)",
                                        "Passive Participle (اسْم الْمَفْعُول)"
                                    ],
                                    "Arabic Word (with Tashkeel)": [
                                        v_data.get('wazn', 'N/A'),
                                        v_data.get('madi', 'N/A'),
                                        v_data.get('mudari', 'N/A'),
                                        v_data.get('amr', 'N/A'),
                                        v_data.get('masdar', 'N/A'),
                                        v_data.get('ism_faail', 'N/A'),
                                        v_data.get('ism_mafool', 'N/A')
                                    ],
                                    "English Translation": [
                                        "N/A",
                                        translations[0],
                                        translations[1],
                                        translations[2],
                                        translations[3],
                                        translations[4],
                                        translations[5]
                                    ]
                                })
                                st.table(verb_df)
                                
                    with tab2:
                        st.subheader("Noun Analysis")
                        
                        # Populate noun selectbox with fallback to custom input
                        if detected_nouns:
                            noun_options = []
                            noun_word_map = {}
                            for n in detected_nouns:
                                if isinstance(n, dict):
                                    w = n.get("word", "")
                                    m = n.get("meaning", "")
                                elif hasattr(n, 'word'):
                                    w = getattr(n, 'word', "")
                                    m = getattr(n, 'meaning', "")
                                else:
                                    w = str(n)
                                    m = ""
                                disp = f"{w} ({m})" if m else w
                                noun_options.append(disp)
                                noun_word_map[disp] = w

                            selected_noun_disp = st.selectbox(
                                "Select a word to analyze as a Noun",
                                options=noun_options,
                                key="noun_select"
                            )
                            custom_noun = st.text_input("Or type a custom Arabic noun to analyze", key="noun_custom")
                            noun_to_analyze = custom_noun.strip() if custom_noun.strip() else noun_word_map.get(selected_noun_disp, selected_noun_disp)
                        else:
                            st.info("No nouns detected in this crop. You can enter one manually below.")
                            noun_to_analyze = st.text_input("Type an Arabic noun to analyze", key="noun_custom_only")
                            
                        if noun_to_analyze:
                            # 1. Run local Farasa Stemmer first to get the root
                            try:
                                stemmed_noun = farasa_stemmer.stem(noun_to_analyze)
                                clean_noun_root = re.sub(r'[^\u0621-\u064A]', '', stemmed_noun)
                                noun_root_display = " - ".join(list(clean_noun_root))
                            except Exception as e:
                                st.error(f"Farasa Stemmer Error: {e}")
                                clean_noun_root = noun_to_analyze
                                noun_root_display = "Unknown"
                                
                            try:
                                noun_trans = GoogleTranslator(source='ar', target='en').translate(noun_to_analyze)
                                noun_root_trans = GoogleTranslator(source='ar', target='en').translate(clean_noun_root)
                            except Exception:
                                noun_trans = "N/A"
                                noun_root_trans = "N/A"
                                
                            st.markdown(f"""
                            <div style="font-size: 18px; margin-bottom: 10px;">
                                Analyzing noun: <span class="arabic-text arabic-medium" style="font-weight: bold; color: #1E88E5;">{noun_to_analyze}</span> (Translation: <em>{noun_trans}</em>)
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.info(f"🌱 Extracted Local Root: {noun_root_display} (Translation: {noun_root_trans})")
                            
                            # Clear results if the noun changes
                            if st.session_state.last_analyzed_noun != noun_to_analyze:
                                st.session_state.noun_results = None
                                
                            api_key = get_gemini_api_key()
                            if not api_key:
                                st.warning("⚠️ Gemini API key is missing. Please configure it to enable Noun analysis.")
                                
                            if st.button("Analyze Noun via Gemini", key="btn_analyze_noun", disabled=(not api_key)):
                                with st.spinner("Analyzing noun forms..."):
                                    try:
                                        noun_data = get_noun_analysis(noun_to_analyze, clean_noun_root)
                                        st.session_state.noun_results = noun_data
                                        st.session_state.last_analyzed_noun = noun_to_analyze
                                    except Exception as e:
                                        st.error(f"Gemini Noun Analysis failed: {e}")
                                        st.session_state.noun_results = None
                                        
                            # Render results
                            if st.session_state.noun_results and st.session_state.last_analyzed_noun == noun_to_analyze:
                                n_data = st.session_state.noun_results
                                
                                # Layout metrics
                                col_n1, col_n2, col_n3 = st.columns(3)
                                with col_n1:
                                    st.metric(label="Classification (النوع)", value=n_data.get('noun_type', 'N/A'))
                                with col_n2:
                                    st.metric(label="Category (الفئة)", value=n_data.get('category', 'N/A'))
                                with col_n3:
                                    st.metric(label="Pattern / Weight (الوزن)", value=n_data.get('wazn', 'N/A'))
                                     
                                st.markdown("#### Noun paradigm table")
                                words_to_translate = [
                                    n_data.get('singular', ''),
                                    n_data.get('dual', ''),
                                    n_data.get('plural', ''),
                                    n_data.get('root_verb', '')
                                ]
                                translations = translate_words(words_to_translate)
                                
                                noun_df = pd.DataFrame({
                                    "Grammatical Element": [
                                        "Classification (النوع)",
                                        "Category (الفئة)",
                                        "Singular Form (المُفْرَد)",
                                        "Dual Form (المُثَنَّى)",
                                        "Plural Form (الجَمْع)",
                                        "Associated Root Verb (الفعل الأصلي)"
                                    ],
                                    "Arabic Word (with Tashkeel)": [
                                        n_data.get('noun_type', 'N/A'),
                                        n_data.get('category', 'N/A'),
                                        n_data.get('singular', 'N/A'),
                                        n_data.get('dual', 'N/A'),
                                        n_data.get('plural', 'N/A'),
                                        n_data.get('root_verb', 'N/A')
                                    ],
                                    "English Translation": [
                                        "N/A",
                                        "N/A",
                                        translations[0],
                                        translations[1],
                                        translations[2],
                                        translations[3]
                                    ]
                                })
                                st.table(noun_df)
                                
                    with tab3:
                        st.subheader("Particle Analysis (الأَدَوَات وَالْحُرُوف)")
                        st.markdown("Particles (حُرُوف) do not have 3-letter roots. Below are the grammatical categories, contextual meanings, and grammatical effects of all particles detected in this text.")
                        
                        if detected_particles:
                            particle_rows = []
                            for p in detected_particles:
                                if isinstance(p, dict):
                                    p_item = p
                                elif hasattr(p, 'model_dump'):
                                    p_item = p.model_dump()
                                elif hasattr(p, 'dict'):
                                    p_item = p.dict()
                                else:
                                    p_item = dict(p)
                                    
                                particle_rows.append({
                                    "Particle (الْحَرْف)": p_item.get("word") or p_item.get("particle", "N/A"),
                                    "Category (النَّوْع)": p_item.get("type", "N/A"),
                                    "Meaning (الْمَعْنَى)": p_item.get("meaning", "N/A"),
                                    "Grammatical Effect (الأَثَر الإِعْرَابِي)": p_item.get("effect", "N/A")
                                })
                                
                            p_df = pd.DataFrame(particle_rows)
                            st.table(p_df)
                        else:
                            st.info("No particles detected in the current text crop. Try processing a text passage containing prepositions (حُرُوف الْجَرّ) or conjunctions (حُرُوف الْعَطْف).")

                    # Prominent Save to Database Button
                    st.markdown("---")
                    if st.button("💾 Save Entry to Database", type="primary", use_container_width=True):
                        if cropped_img and diacritized_text:
                            img_b64 = image_to_base64(cropped_img)
                            deep_sarf = {
                                "last_verb": st.session_state.last_analyzed_verb,
                                "verb_sarf": st.session_state.verb_results,
                                "last_noun": st.session_state.last_analyzed_noun,
                                "noun_sarf": st.session_state.noun_results
                            }
                            save_study_entry(
                                source_filename=selected_file_name,
                                image_base64=img_b64,
                                tashkeel_text=diacritized_text,
                                full_translation=st.session_state.full_translation,
                                verbs=st.session_state.verbs,
                                nouns=st.session_state.nouns,
                                particles=st.session_state.particles,
                                deep_sarf=deep_sarf
                            )
                            st.success("Saved entry to study database! 🎉")
                        else:
                            st.warning("Please process a crop first before saving to database.")

    # Expandable Section: Saved History & Anki Export
    st.markdown("---")
    with st.expander("📚 Saved History & Anki Export", expanded=False):
        entries = get_all_study_entries()
        if entries:
            st.subheader(f"Saved Study Entries ({len(entries)})")
            
            # 1. Saved Entry Selection & Delete Management
            entry_map = {}
            dropdown_options = []
            
            for row in entries:
                entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = row
                snippet = tashkeel[:35] + "..." if len(tashkeel) > 35 else tashkeel
                opt_str = f"[ID: {entry_id}] {timestamp} | {fname} | {snippet}"
                dropdown_options.append(opt_str)
                entry_map[opt_str] = row
                
            col_sel, col_del = st.columns([4, 1])
            with col_sel:
                selected_option = st.selectbox("🔍 Select Saved Entry to Inspect", options=dropdown_options, key="inspect_entry_select")
            with col_del:
                st.write("")
                st.write("")
                selected_row = entry_map[selected_option]
                sel_id = selected_row[0]
                if st.button("🗑️ Delete Entry", key=f"btn_del_{sel_id}", use_container_width=True):
                    delete_study_entry(sel_id)
                    st.success(f"Deleted entry #{sel_id}!")
                    st.rerun()

            # 2. Full Entry Inspector View
            entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = selected_row
            
            st.markdown("---")
            st.markdown(f"### Inspector View: Entry #{entry_id} (`{fname}` — *{timestamp}*)")
            
            col_img, col_details = st.columns([1, 2])
            
            with col_img:
                if img_b64:
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        saved_pil_img = Image.open(io.BytesIO(img_bytes))
                        st.image(saved_pil_img, caption=f"Saved Crop ({fname})", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load image: {e}")
                else:
                    st.caption("No image data stored.")
                    
            with col_details:
                st.markdown(f"""
                <div dir="rtl" class="arabic-text arabic-large" style="padding: 12px; border: 2px solid #4CAF50; border-radius: 8px; background-color: rgba(76, 175, 80, 0.05); margin-bottom: 12px;">
                    <strong>Tashkeel:</strong> {tashkeel}
                </div>
                """, unsafe_allow_html=True)
                
                if translation:
                    st.info(f"💡 **English Translation:** {translation}")
                else:
                    st.caption("No translation recorded.")

            # Parse saved JSON fields
            try:
                saved_verbs = json.loads(verbs_str) if verbs_str else []
            except Exception:
                saved_verbs = []
            try:
                saved_nouns = json.loads(nouns_str) if nouns_str else []
            except Exception:
                saved_nouns = []
            try:
                saved_particles = json.loads(particles_str) if particles_str else []
            except Exception:
                saved_particles = []
            try:
                saved_deep_sarf = json.loads(deep_sarf_str) if deep_sarf_str else {}
            except Exception:
                saved_deep_sarf = {}

            # 3. Categorized Word Tabs for Saved Entry
            st.markdown("#### Categorized Vocabulary Breakdown")
            tab_v, tab_n, tab_p = st.tabs(["⚙️ Saved Verbs", "🏷️ Saved Nouns", "📌 Saved Particles"])
            
            with tab_v:
                if saved_verbs:
                    v_rows = []
                    for v in saved_verbs:
                        if isinstance(v, dict):
                            v_rows.append({"Word (الْكَلِمَة)": v.get("word", "N/A"), "Meaning (الْمَعْنَى)": v.get("meaning", "N/A")})
                        else:
                            v_rows.append({"Word (الْكَلِمَة)": str(v), "Meaning (الْمَعْنَى)": "N/A"})
                    st.table(pd.DataFrame(v_rows))
                else:
                    st.caption("No verbs recorded for this entry.")
                    
            with tab_n:
                if saved_nouns:
                    n_rows = []
                    for n in saved_nouns:
                        if isinstance(n, dict):
                            n_rows.append({"Word (الْكَلِمَة)": n.get("word", "N/A"), "Meaning (الْمَعْنَى)": n.get("meaning", "N/A")})
                        else:
                            n_rows.append({"Word (الْكَلِمَة)": str(n), "Meaning (الْمَعْنَى)": "N/A"})
                    st.table(pd.DataFrame(n_rows))
                else:
                    st.caption("No nouns recorded for this entry.")
                    
            with tab_p:
                if saved_particles:
                    p_rows = []
                    for p in saved_particles:
                        if isinstance(p, dict):
                            p_rows.append({
                                "Particle (الْحَرْف)": p.get("word") or p.get("particle", "N/A"),
                                "Category (النَّوْع)": p.get("type", "N/A"),
                                "Meaning (الْمَعْنَى)": p.get("meaning", "N/A"),
                                "Grammatical Effect (الأَثَر الإِعْرَابِي)": p.get("effect", "N/A")
                            })
                        else:
                            p_rows.append({"Particle (الْحَرْف)": str(p), "Category (النَّوْع)": "N/A", "Meaning (الْمَعْنَى)": "N/A", "Grammatical Effect (الأَثَر الإِعْرَابِي)": "N/A"})
                    st.table(pd.DataFrame(p_rows))
                else:
                    st.caption("No particles recorded for this entry.")

            # 4. Deep Sarf Cache Viewer
            if saved_deep_sarf and (saved_deep_sarf.get("verb_sarf") or saved_deep_sarf.get("noun_sarf")):
                with st.expander("📖 Saved Deep Sarf Analysis", expanded=False):
                    if saved_deep_sarf.get("verb_sarf"):
                        st.markdown(f"**Verb Sarf for '{saved_deep_sarf.get('last_verb', '')}':**")
                        st.json(saved_deep_sarf["verb_sarf"])
                    if saved_deep_sarf.get("noun_sarf"):
                        st.markdown(f"**Noun Sarf for '{saved_deep_sarf.get('last_noun', '')}':**")
                        st.json(saved_deep_sarf["noun_sarf"])

            # 5. History Summary Table & Anki Export Button
            st.markdown("---")
            st.subheader("Summary Table & Anki Deck Export")
            
            history_data = []
            anki_rows = []
            
            for row in entries:
                e_id, t_stamp, f_name, i_b64, t_text, f_trans, v_str, n_str, p_str, d_str = row
                try:
                    v_list = json.loads(v_str) if v_str else []
                except Exception:
                    v_list = []
                try:
                    n_list = json.loads(n_str) if n_str else []
                except Exception:
                    n_list = []
                try:
                    p_list = json.loads(p_str) if p_str else []
                except Exception:
                    p_list = []

                verbs_formatted = ", ".join([f"{v.get('word','')} ({v.get('meaning','')})" if isinstance(v, dict) else str(v) for v in v_list])
                nouns_formatted = ", ".join([f"{n.get('word','')} ({n.get('meaning','')})" if isinstance(n, dict) else str(n) for n in n_list])
                particles_formatted = ", ".join([f"{p.get('word','') or p.get('particle','')} ({p.get('meaning','')})" if isinstance(p, dict) else str(p) for p in p_list])

                back_card = f"{f_trans}\n\n[Verbs]: {verbs_formatted}\n[Nouns]: {nouns_formatted}\n[Particles]: {particles_formatted}"
                
                anki_rows.append({
                    "Front (Arabic)": t_text,
                    "Back (English)": back_card,
                    "Source File": f_name,
                    "Timestamp": t_stamp
                })

                history_data.append({
                    "ID": e_id,
                    "Timestamp": t_stamp,
                    "Source File": f_name,
                    "Arabic Text (Tashkeel)": t_text,
                    "Translation": f_trans,
                    "Verbs": len(v_list),
                    "Nouns": len(n_list),
                    "Particles": len(p_list)
                })

            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)

            df_anki = pd.DataFrame(anki_rows)
            csv_buffer = df_anki.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Anki CSV",
                data=csv_buffer,
                file_name="arabic_manga_anki_deck.csv",
                mime="text/csv",
                key="download_anki_csv"
            )
        else:
            st.info("No entries saved in the database yet. Click '💾 Save Entry to Database' after analyzing a crop!")
else:
    st.info("Please upload one or more manga pages to begin.")
