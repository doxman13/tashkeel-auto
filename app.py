import streamlit as st
from streamlit_cropper import st_cropper
import easyocr
import pypdf
import pdf2image
import io

import base64
from datetime import datetime
import asyncio
import edge_tts
from gtts import gTTS
import streamlit.components.v1 as components

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
    """Insert a study entry into SQLite database and return new ID."""
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
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def update_study_entry_sarf(entry_id: int, deep_sarf: dict):
    """Update deep_sarf_json for an existing study log entry in SQLite database."""
    conn = sqlite3.connect("arabic_study_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE study_logs
        SET deep_sarf_json = ?
        WHERE id = ?
    """, (json.dumps(deep_sarf, ensure_ascii=False), entry_id))
    conn.commit()
    conn.close()

def auto_save_or_update_current_entry(source_filename, cropped_img, diacritized_text, full_translation, verbs, nouns, particles, verb_results, last_verb, noun_results, last_noun):
    """Auto-save or update the current study entry in SQLite database whenever Sarf or Noun analysis is generated."""
    if not diacritized_text:
        return None
        
    img_b64 = image_to_base64(cropped_img) if cropped_img else ""
    deep_sarf = {
        "last_verb": last_verb,
        "verb_sarf": verb_results,
        "last_noun": last_noun,
        "noun_sarf": noun_results
    }
    
    current_id = st.session_state.get("current_db_entry_id")
    if current_id:
        conn = sqlite3.connect("arabic_study_history.db")
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE study_logs
            SET verbs_json = ?, nouns_json = ?, particles_json = ?, deep_sarf_json = ?
            WHERE id = ?
        """, (
            json.dumps(verbs, ensure_ascii=False),
            json.dumps(nouns, ensure_ascii=False),
            json.dumps(particles, ensure_ascii=False),
            json.dumps(deep_sarf, ensure_ascii=False),
            current_id
        ))
        conn.commit()
        conn.close()
        return current_id
    else:
        new_id = save_study_entry(
            source_filename=source_filename,
            image_base64=img_b64,
            tashkeel_text=diacritized_text,
            full_translation=full_translation,
            verbs=verbs,
            nouns=nouns,
            particles=particles,
            deep_sarf=deep_sarf
        )
        st.session_state.current_db_entry_id = new_id
        return new_id

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

def build_word_meaning_map(verbs: list, nouns: list, particles: list) -> dict:
    """Build a lookup dictionary mapping diacritized & raw words to their grammar category, sub_type, derived status, base_verb, root, and meaning."""
    lookup = {}

    for v in verbs:
        if isinstance(v, dict):
            w = v.get("word", "")
            m = v.get("meaning", "")
            derived = v.get("derived", True)
            sub_type = v.get("sub_type", "Verb")
            base_verb = v.get("base_verb")
            root = v.get("root")
        elif hasattr(v, 'word'):
            w = getattr(v, 'word', "")
            m = getattr(v, 'meaning', "")
            derived = getattr(v, 'derived', True)
            sub_type = getattr(v, 'sub_type', "Verb")
            base_verb = getattr(v, 'base_verb', None)
            root = getattr(v, 'root', None)
        else:
            w, m = str(v), ""
            derived, sub_type, base_verb, root = True, "Verb", None, None

        if w:
            entry = {
                "word": w,
                "role": f"⚙️ {sub_type} (فِعْل)" if sub_type == "Verb" else f"⚙️ Verb ({sub_type})",
                "meaning": m,
                "derived": derived,
                "sub_type": sub_type,
                "base_verb": base_verb,
                "root": root
            }
            lookup[w] = entry
            lookup[strip_tashkeel(w)] = entry

    for n in nouns:
        if isinstance(n, dict):
            w = n.get("word", "")
            m = n.get("meaning", "")
            derived = n.get("derived", False)
            sub_type = n.get("sub_type", "Solid Noun")
            base_verb = n.get("base_verb")
            root = n.get("root")
        elif hasattr(n, 'word'):
            w = getattr(n, 'word', "")
            m = getattr(n, 'meaning', "")
            derived = getattr(n, 'derived', False)
            sub_type = getattr(n, 'sub_type', "Solid Noun")
            base_verb = getattr(n, 'base_verb', None)
            root = getattr(n, 'root', None)
        else:
            w, m = str(n), ""
            derived, sub_type, base_verb, root = False, "Solid Noun", None, None

        if w:
            entry = {
                "word": w,
                "role": f"🏷️ Noun ({sub_type})" if sub_type != "Solid Noun" else "🏷️ Noun (اسْم)",
                "meaning": m,
                "derived": derived,
                "sub_type": sub_type,
                "base_verb": base_verb,
                "root": root
            }
            lookup[w] = entry
            lookup[strip_tashkeel(w)] = entry

    for p in particles:
        if isinstance(p, dict):
            w = p.get("word") or p.get("particle", "")
            m = p.get("meaning", "")
            p_type = p.get("type", "Particle (حَرْف)")
            effect = p.get("effect", "")
        elif hasattr(p, 'word'):
            w = getattr(p, 'word', "")
            m = getattr(p, 'meaning', "")
            p_type = getattr(p, 'type', "Particle (حَرْف)")
            effect = getattr(p, 'effect', "")
        else:
            w = str(p)
            m = ""
            p_type = "Particle (حَرْف)"
            effect = ""
        if w:
            entry = {
                "word": w,
                "role": f"📌 {p_type}",
                "meaning": m,
                "effect": effect,
                "derived": False,
                "sub_type": "Particle",
                "base_verb": None,
                "root": None
            }
            lookup[w] = entry
            lookup[strip_tashkeel(w)] = entry

    return lookup


def aggregate_vocabulary_across_entries(entries):
    """Aggregate & deduplicate verbs, nouns, and particles across all saved study entries.

    Words differing only in diacritics are merged into a single row; distinct
    meanings are preserved when present. Returns (all_verbs, all_nouns, all_particles),
    each a sorted list of dicts with keys: word, meaning, sub_type/category, derived,
    base_verb, root/effect, count, sources.
    """
    verbs_index, nouns_index, particles_index = {}, {}, {}

    def _ensure_record(item, defaults):
        if isinstance(item, dict):
            word = item.get("word", "")
            record = {
                "word": word,
                "meanings": [m for m in [item.get("meaning", "")] if m],
                "sub_type": item.get("sub_type", defaults["sub_type"]),
                "derived": item.get("derived", defaults["derived"]),
                "base_verb": item.get("base_verb"),
                "root": item.get("root"),
            }
        else:
            word = str(item)
            record = {
                "word": word,
                "meanings": [],
                "sub_type": defaults["sub_type"],
                "derived": defaults["derived"],
                "base_verb": None,
                "root": None,
            }
        return word, record

    def _ensure_particle_record(item):
        if isinstance(item, dict):
            word = item.get("word") or item.get("particle", "")
            return {
                "word": word,
                "category": item.get("type", ""),
                "effect": item.get("effect", ""),
                "meanings": [m for m in [item.get("meaning", "")] if m],
            }
        else:
            word = str(item)
            return {
                "word": word,
                "category": "",
                "effect": "",
                "meanings": [],
            }

    for row in entries:
        entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = row
        source_info = {"id": entry_id, "timestamp": timestamp, "source": fname}

        for idx, defaults, raw_json in (
            (verbs_index, {"sub_type": "Verb", "derived": True}, verbs_str),
            (nouns_index, {"sub_type": "Solid Noun", "derived": False}, nouns_str),
        ):
            try:
                items = json.loads(raw_json) if raw_json else []
            except Exception:
                items = []
            for item in items:
                word, record = _ensure_record(item, defaults)
                if not word:
                    continue
                key = strip_tashkeel(word)
                if not key:
                    continue
                if key in idx:
                    existing = idx[key]
                    existing["count"] += 1
                    existing["sources"].append(source_info)
                    for m in record["meanings"]:
                        if m not in existing["meanings"]:
                            existing["meanings"].append(m)
                    if not existing["base_verb"] and record["base_verb"]:
                        existing["base_verb"] = record["base_verb"]
                    if not existing["root"] and record["root"]:
                        existing["root"] = record["root"]
                else:
                    record["count"] = 1
                    record["sources"] = [source_info]
                    idx[key] = record

        try:
            p_items = json.loads(particles_str) if particles_str else []
        except Exception:
            p_items = []
        for item in p_items:
            record = _ensure_particle_record(item)
            word = record["word"]
            if not word:
                continue
            key = strip_tashkeel(word)
            if not key:
                continue
            if key in particles_index:
                existing = particles_index[key]
                existing["count"] += 1
                existing["sources"].append(source_info)
                for m in record["meanings"]:
                    if m not in existing["meanings"]:
                        existing["meanings"].append(m)
                if not existing["category"] and record["category"]:
                    existing["category"] = record["category"]
                if not existing["effect"] and record["effect"]:
                    existing["effect"] = record["effect"]
            else:
                record["count"] = 1
                record["sources"] = [source_info]
                particles_index[key] = record

    def _finalize(records):
        result = []
        for r in records.values():
            r["meaning"] = "; ".join(r["meanings"]) if r["meanings"] else ""
            del r["meanings"]
            seen_ids = set()
            unique_sources = []
            for s in r["sources"]:
                if s["id"] not in seen_ids:
                    seen_ids.add(s["id"])
                    unique_sources.append(s)
            r["sources"] = unique_sources
            result.append(r)
        return sorted(result, key=lambda x: (-x["count"], x["word"]))

    return _finalize(verbs_index), _finalize(nouns_index), _finalize(particles_index)


def _escape_html_attr(text: str) -> str:
    """Escape characters that would break HTML attribute values."""
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


@st.cache_data
def build_interactive_tashkeel_html(diacritized_text: str, verbs_json: str, nouns_json: str, particles_json: str):
    """Pre-compute interactive tashkeel HTML for both vowel-showing and vowel-hiding states.

    The expensive Google Translate fallback calls only happen once per text version,
    making the Show Vowels toggle feel instant. Returns (vowel_html, no_vowel_html, sentence_tokens).
    """
    verbs = json.loads(verbs_json) if verbs_json else []
    nouns = json.loads(nouns_json) if nouns_json else []
    particles = json.loads(particles_json) if particles_json else []

    word_map = build_word_meaning_map(verbs, nouns, particles)
    sentence_tokens = [w.strip() for w in re.split(r'[\s،؛؟\.\!\:\-"\']+', diacritized_text) if w.strip()]

    vowel_spans = []
    novowel_spans = []

    for token in sentence_tokens:
        raw_token = strip_tashkeel(token)
        match_info = word_map.get(token) or word_map.get(raw_token)
        if match_info:
            tooltip_txt = f"{match_info.get('role', '')} | Meaning: {match_info.get('meaning', '')}"
        else:
            try:
                unclass_trans = GoogleTranslator(source='ar', target='en').translate(raw_token or token)
            except Exception:
                unclass_trans = "N/A"
            tooltip_txt = f"Unclassified Word | Meaning: {unclass_trans}"

        safe_tooltip = _escape_html_attr(tooltip_txt)
        vowel_spans.append(f'<span class="tashkeel-word" title="{safe_tooltip}">{token}</span>')
        novowel_spans.append(f'<span class="tashkeel-word" title="{safe_tooltip}">{raw_token}</span>')

    return " ".join(vowel_spans), " ".join(novowel_spans), sentence_tokens


def render_sub_type_badge(sub_type: str, derived: bool, base_verb: str | None = None) -> str:
    """Generate HTML badge for sub_type and derived status."""
    badge_colors = {
        "Verb": ("#1E88E5", "rgba(30, 136, 229, 0.15)"),
        "Ism Fa'il": ("#2E7D32", "rgba(46, 125, 50, 0.15)"),
        "Ism Maf'ul": ("#7B1FA2", "rgba(123, 31, 162, 0.15)"),
        "Masdar": ("#E65100", "rgba(230, 81, 0, 0.15)"),
        "Sifah Mushabbahah": ("#00838F", "rgba(0, 131, 143, 0.15)"),
        "Solid Noun": ("#616161", "rgba(97, 97, 97, 0.15)"),
        "Particle": ("#D81B60", "rgba(216, 27, 96, 0.15)")
    }
    fg, bg = badge_colors.get(sub_type, ("#1E88E5", "rgba(30, 136, 229, 0.15)"))
    
    badges = [f'<span style="background-color: {bg}; color: {fg}; font-weight: bold; padding: 2px 8px; border-radius: 10px; font-size: 13px; border: 1px solid {fg}; margin-right: 4px;">{sub_type}</span>']
    if derived:
        badges.append('<span style="background-color: rgba(76, 175, 80, 0.15); color: #2E7D32; font-weight: bold; padding: 2px 8px; border-radius: 10px; font-size: 13px; border: 1px solid #2E7D32; margin-right: 4px;">⚡ Derived (مشتق)</span>')
    if base_verb:
        badges.append(f'<span style="background-color: rgba(30, 136, 229, 0.1); color: #1E88E5; padding: 2px 8px; border-radius: 10px; font-size: 13px; border: 1px solid #1E88E5;">🌱 Base: {base_verb}</span>')
    
    return " ".join(badges)

def render_custom_table(df: pd.DataFrame):
    """Render DataFrame as an HTML table where Arabic cells are 34px and English cells are 16px."""
    if df.empty:
        return
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda val: f'<span class="arabic-cell">{val}</span>' if isinstance(val, str) and re.search(r'[\u0600-\u06FF]', val) else str(val)
        )
    html_table = formatted_df.to_html(escape=False, index=False, classes="custom-styled-table")
    st.markdown(html_table, unsafe_allow_html=True)

async def _edge_tts_async(text: str, voice: str = "ar-SA-HamedNeural") -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def generate_edge_audio(text: str, voice: str = "ar-SA-HamedNeural") -> bytes:
    """Generate audio using Microsoft Edge Neural TTS in memory."""
    try:
        return asyncio.run(_edge_tts_async(text, voice))
    except Exception:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(_edge_tts_async(text, voice))

def generate_gtts_audio(text: str) -> bytes:
    """Generate audio using Google TTS in memory."""
    tts = gTTS(text=text, lang='ar', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.getvalue()

def render_arabic_tts(text: str, engine_choice: str, key_suffix: str = "main"):
    """Render audio controls for the provided Arabic text based on chosen engine."""
    if not text or not text.strip():
        return

    if "Microsoft Edge" in engine_choice:
        if st.button(f"🔊 Play (Edge Neural)", key=f"btn_tts_edge_{key_suffix}"):
            with st.spinner("Generating lifelike Edge Neural audio..."):
                try:
                    audio_bytes = generate_edge_audio(text)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.error(f"Edge TTS error: {e}")
    elif "Google Voice" in engine_choice:
        if st.button(f"🔊 Play (gTTS)", key=f"btn_tts_gtts_{key_suffix}"):
            with st.spinner("Generating Google Voice audio..."):
                try:
                    audio_bytes = generate_gtts_audio(text)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.error(f"gTTS error: {e}")
    else:
        # Browser Native Web Speech API with Chrome Voice Resolution
        escaped_txt = text.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
        html_code = f"""
        <div style="margin-top: 5px;">
            <button onclick="speakText()" style="
                background-color: #1E88E5; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 6px; 
                font-size: 15px; 
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 6px;">
                ⚡ 🔊 Listen via Browser (Instant)
            </button>
        </div>
        <script>
        let arVoice = null;
        function loadVoices() {{
            if ('speechSynthesis' in window) {{
                const voices = window.speechSynthesis.getVoices();
                arVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('ar'));
            }}
        }}
        if ('speechSynthesis' in window) {{
            loadVoices();
            if (window.speechSynthesis.onvoiceschanged !== undefined) {{
                window.speechSynthesis.onvoiceschanged = loadVoices;
            }}
        }}
        function speakText() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const msg = new SpeechSynthesisUtterance('{escaped_txt}');
                msg.lang = 'ar-SA';
                msg.rate = 0.85;
                if (!arVoice) {{
                    loadVoices();
                }}
                if (arVoice) {{
                    msg.voice = arVoice;
                }}
                window.speechSynthesis.speak(msg);
            }} else {{
                alert('Browser Web Speech API not supported.');
            }}
        }}
        </script>
        """
        components.html(html_code, height=45)

# Pydantic Schemas for Master Gemini JSON Response
class WordMeaningItem(BaseModel):
    word: str = Field(description="The diacritized Arabic word (with full Tashkeel)")
    meaning: str = Field(description="Contextual English translation/meaning of this word in the sentence")
    derived: bool = Field(default=False, description="True for Verbs AND Derived Nouns (Ism Fa'il, Ism Maf'ul, Masdar, Sifah Mushabbahah); False for Solid Nouns (الأسماء الجامدة) and Particles.")
    sub_type: str = Field(default="Noun", description="Sub-type classification: 'Verb', 'Solid Noun', 'Ism Fa'il', 'Ism Maf'ul', 'Masdar', 'Sifah Mushabbahah', or 'Particle'.")
    base_verb: str | None = Field(default=None, description="The past-tense base verb string (e.g., 'إِبْتَسَمَ' for 'مُبْتَسِمًا'), or null if solid noun.")
    root: str | None = Field(default=None, description="The 3-letter or 4-letter root string (e.g., 'ب-س-م'), or null if unknown.")

class ParticleItem(BaseModel):
    word: str = Field(description="The diacritized particle (with full Tashkeel, e.g. 'فِي', 'إِنَّ')")
    type: str = Field(description="Particle category in Arabic & English (e.g. 'حَرْف جَرّ (Genitive Particle)')")
    meaning: str = Field(description="Contextual English meaning (e.g. 'In / At')")
    effect: str = Field(description="Grammatical effect / الأَثَر الإِعْرَابِي (e.g. 'Forces following noun into Majroor state')")
    derived: bool = Field(default=False, description="Always False for particles.")
    sub_type: str = Field(default="Particle", description="Always 'Particle'.")
    base_verb: str | None = Field(default=None, description="Always null for particles.")
    root: str | None = Field(default=None, description="Always null for particles.")

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
    "You are an expert Arabic grammarian, OCR engine, and computational linguist tool.\n"
    "Your task is to transcribe/process raw Arabic text or images, fix minor typos, add full, grammatically precise Tashkeel (diacritics), "
    "provide a complete, fluent English translation of the sentence/passage, and exhaustively analyze EVERY single word into the 3 distinct pillars of Arabic grammar:\n"
    "Verbs (أَفْعَال), Nouns (أَسْمَاء), and Particles (حُرُوف).\n\n"
    "CRITICAL CLASSIFICATION & STRUCTURE RULES:\n"
    "1. 'tashkeel_text': The full diacritized Arabic text with complete vowels.\n"
    "2. 'full_translation': Complete, accurate English translation of the entire cropped sentence/passage.\n\n"
    "3. 'verbs': List of objects for all past, present, and imperative verbs. Verbs with attached object/subject pronouns MUST be included here.\n"
    "   - Schema: {\n"
    '       "word": "<diacritized_verb>",\n'
    '       "meaning": "<english_meaning>",\n'
    '       "derived": true,\n'
    '       "sub_type": "Verb",\n'
    '       "base_verb": "<past_tense_form_I_or_augmented_verb>",\n'
    '       "root": "<3_or_4_letter_root_separated_by_dashes>"\n'
    "     }\n\n"
    "4. 'nouns': List of objects under the Noun category (أَسْمَاء). In Arabic grammar, this includes:\n"
    "     a) Solid Nouns (أَسْمَاء جَامِدَة) & Adjectives.\n"
    "     b) Active Participles (اسْم فَاعِل) e.g., 'مُسْرِعًا', 'جَالِسٌ', 'مُبْتَسِمًا'.\n"
    "     c) Passive Participles (اسْم مَفْعُول) e.g., 'مَكْتُوب', 'مَفْتُوح'.\n"
    "     d) Verbal Nouns (مَصْدَر), Sifah Mushabbahah (صِفَة مُشَبَّهَة), and Adverbs of Time/Place (اسْم زَمَان/مَكَان).\n"
    "     e) Pronouns, Demonstratives (هَذَا), Relative Pronouns (الَّذِي), and Time/Place Adverbs (أَبَدًا, الْآنَ).\n"
    "   - Schema: {\n"
    '       "word": "<diacritized_noun>",\n'
    '       "meaning": "<english_meaning>",\n'
    '       "derived": <true_if_Ism_Fail_Ism_Mafoul_Masdar_SifahMushabbahah_else_false>,\n'
    '       "sub_type": "<Solid Noun | Ism Fa\'il | Ism Maf\'ul | Masdar | Sifah Mushabbahah>",\n'
    '       "base_verb": "<past_tense_base_verb_if_derived_is_true_else_null>",\n'
    '       "root": "<3_or_4_letter_root_separated_by_dashes_if_applicable_else_null>"\n'
    "     }\n\n"
    "5. 'particles': List of objects for prepositions (حُرُوف جَرّ), accusative/subjunctive particles (حُرُوف نَصْب like إِنَّ / أَنْ), "
    "jussive particles (حُرُوف جَزْم), and conjunctions (حُرُوف عَطْف like وَ, فَ, ثُمَّ).\n"
    "   - Schema: {\n"
    '       "word": "<diacritized_particle>",\n'
    '       "type": "<category>",\n'
    '       "meaning": "<meaning>",\n'
    '       "effect": "<grammatical_effect>",\n'
    '       "derived": false,\n'
    '       "sub_type": "Particle",\n'
    '       "base_verb": null,\n'
    '       "root": null\n'
    "     }\n\n"
    "6. DERIVATION & SARF RULES:\n"
    "   - Set 'derived': true for all Verbs AND for all Nouns that are morphologically derived from a verb (Ism Fa'il, Ism Maf'ul, Masdar, Sifah Mushabbahah).\n"
    "   - Set 'derived': false for solid nouns (الأسماء الجامدة like 'كِتَاب', 'مَدِينَة') and all particles.\n"
    "   - Whenever 'derived' is true, ALWAYS populate 'base_verb' with the unaugmented or augmented past-tense Form verb (e.g., for 'مُبْتَسِمًا', base_verb is 'إِبْتَسَمَ'; for 'جَالِسٌ', base_verb is 'جَلَسَ').\n\n"
    "7. EXHAUSTIVE COVERAGE: Every single word in 'tashkeel_text' MUST appear in exactly one of the three lists ('verbs', 'nouns', or 'particles'). Do NOT omit any word!"
)

def gemini_tashkeel(raw_arabic_text: str) -> dict:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key is missing. Please configure it in app.py, environment variables, or Streamlit secrets.")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
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
        model='gemini-3.1-flash-lite',
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
        model='gemini-3.1-flash-lite',
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
        model='gemini-3.1-flash-lite',
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
st.set_page_config(layout="wide", page_title="Arabic Diacritizer & Analyzer", page_icon=":material/translate:")

# Hero banner
st.html("""
<div class="app-hero">
    <div class="app-hero-icon">📖</div>
    <div>
        <p class="app-hero-title">Arabic Diacritizer & Analyzer</p>
        <p class="app-hero-sub">Upload manga pages or PDFs, crop speech bubbles, extract & diacritize Arabic text, and study grammar in depth.</p>
    </div>
</div>
""")

# Inject WebFont definitions for KFGQPC Uthman Taha Naskh & HAFS and custom CSS for Arabic rendering
st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Scheherazade+New:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* WebFont definitions for official King Fahd Complex Uthmani fonts with punctuation and digit fallback */
        @font-face {
            font-family: 'KFGQPC Uthman Taha Naskh';
            src: url('https://cdn.jsdelivr.net/npm/kfgqpc-uthmanic-script-hafs-regular@1.0.0/arabic.otf') format('opentype');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
            unicode-range: U+0000-060B, U+060D-065F, U+066A-FFFF; /* Exclude U+060C (comma) and U+0660-U+0669 (Arabic digits ٠-٩) to avoid verse marker rosettes */
        }
        @font-face {
            font-family: 'KFGQPC Uthmanic Script HAFS';
            src: url('https://cdn.jsdelivr.net/npm/kfgqpc-uthmanic-script-hafs-regular@1.0.0/arabic.otf') format('opentype');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
            unicode-range: U+0000-060B, U+060D-065F, U+066A-FFFF; /* Exclude U+060C (comma) and U+0660-U+0669 (Arabic digits ٠-٩) to avoid verse marker rosettes */
        }

        /* Define Arabic text class using requested KFGQPC Uthman Taha Naskh font stack with punctuation fallbacks */
        .arabic-text {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 24px !important;
            line-height: 1.6 !important;
            direction: rtl !important;
            text-align: right !important;
        }
        .arabic-large {
            font-size: 26px !important;
            font-weight: bold;
            line-height: 1.7 !important;
        }
        .arabic-medium {
            font-size: 20px !important;
            line-height: 1.5 !important;
        }
        .arabic-root {
            font-size: 20px !important;
            font-weight: bold !important;
            letter-spacing: 4px !important;
        }
        
        /* 1. Global Tables, DataFrames & Custom Styled Table */
        .custom-styled-table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin-top: 10px !important;
            margin-bottom: 15px !important;
        }

        .custom-styled-table th,
        table th,
        div[role="tabpanel"] table th,
        div[data-testid="stTable"] th,
        div[data-testid="stDataFrame"] th {
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 8px 12px !important;
            background-color: rgba(200, 200, 200, 0.15) !important;
            border: 1px solid #ddd !important;
            text-align: left !important;
        }

        .custom-styled-table td,
        table td {
            font-size: 16px !important;
            font-weight: normal !important;
            padding: 8px 12px !important;
            border: 1px solid #ddd !important;
            text-align: left !important;
            vertical-align: middle !important;
        }

        .arabic-cell,
        span.arabic-cell,
        td .arabic-cell {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-weight: 700 !important;
            line-height: 1.4 !important;
            direction: rtl !important;
            display: inline-block !important;
            text-align: right !important;
        }

        /* 2. Text areas & inputs using requested font stack & RTL text alignment */
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input,
        div[class*="st-key-diacritized_text"] textarea,
        div[class*="st-key-raw_text"] textarea,
        .stTextArea textarea {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 24px !important;
            line-height: 1.6 !important;
            direction: rtl !important;
            text-align: right !important;
        }

        /* Force LTR + normal font on non-Arabic inputs (number, select, etc.) */
        input[type="number"],
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div,
        div[data-testid="stRadio"] label,
        div[data-testid="stMultiSelect"] div {
            direction: ltr !important;
            text-align: left !important;
            font-family: inherit !important;
        }

        /* 3. Interactive Word Pills Font Size */
        div[data-testid="stPills"] button *,
        div[data-testid="stPills"] button span {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 26px !important;
            line-height: 1.5 !important;
        }

        /* 3. Streamlit native metric values */
        div[data-testid="stMetricValue"] {
            font-family: 'KFGQPC Uthman Taha Naskh', 'KFGQPC Uthmanic Script HAFS', 'Scheherazade New', 'Amiri', 'Trebuchet MS', Arial, Helvetica, sans-serif !important;
            font-size: 20px !important;
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

        /* Hero banner */
        .app-hero {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
            border: 1px solid #4338ca;
            border-radius: 12px;
            padding: 20px 28px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .app-hero-icon { font-size: 38px; line-height: 1; }
        .app-hero-title {
            font-size: 24px;
            font-weight: 700;
            color: #e0e7ff;
            margin: 0;
            line-height: 1.2;
        }
        .app-hero-sub {
            font-size: 13px;
            color: #a5b4fc;
            margin: 6px 0 0 0;
        }

        /* File status strip */
        .file-status-strip {
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 13px;
            color: #a5b4fc;
            margin-bottom: 10px;
        }

        /* Centered 1280px Main Container */
        section.main .block-container,
        div[data-testid="stAppViewBlockContainer"] {
            max-width: 1280px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* Sticky Left Column on Desktop Web View */
        @media (min-width: 769px) {
            div[data-testid="stColumn"]:nth-of-type(1),
            div[data-testid="column"]:nth-of-type(1) {
                position: sticky !important;
                top: 3.5rem !important;
                align-self: flex-start !important;
                z-index: 90 !important;
                max-height: calc(100vh - 4.5rem) !important;
                overflow: hidden !important;
                padding-right: 0.5rem !important;
            }
        }

        /* Scrollable Cropper Viewport Container */
        div[data-testid="stCustomComponentV1"] {
            width: 100% !important;
            max-width: 100% !important;
            max-height: 70vh !important;
            overflow: auto !important;
        }

        /* Mobile overrides */
        @media (max-width: 768px) {
            .arabic-text { font-size: 20px !important; }
            .arabic-large { font-size: 22px !important; }
            .app-hero { padding: 16px 18px; flex-direction: column; align-items: flex-start; }
            .app-hero-title { font-size: 18px; }
            section.main .block-container,
            div[data-testid="stAppViewBlockContainer"] {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }


        /* Hover highlight for interactive tashkeel words */
        .tashkeel-word {
            cursor: pointer;
            padding: 0 3px;
        }
        .tashkeel-word:hover {
            background-color: rgba(33, 150, 243, 0.15);
            border-radius: 4px;
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

farasa_stemmer = load_farasa_stemmer()

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
if 'current_db_entry_id' not in st.session_state:
    st.session_state.current_db_entry_id = None
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

# ── Top-Level Centered Modals for Text Inspection ──
@st.dialog("📄 Raw Extracted Text")
def show_raw_text_modal():
    raw_text = st.session_state.get("extracted_text", "")
    st.caption("Copy or edit the raw extracted text below:")
    bidi_raw = process_text_for_display(raw_text)
    st.text_area("Raw Text (Editable / Copyable)", value=bidi_raw, height=130, key="modal_raw_area")
    st.markdown("#### 🔍 Raw Text (Large Font)")
    st.markdown(f"""
    <div dir="rtl" class="arabic-text" style="font-size: 24px !important; font-weight: normal !important; padding: 12px; border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; text-align: right; line-height: 1.8;">
        {raw_text}
    </div>
    """, unsafe_allow_html=True)

@st.dialog("✨ Diacritized Text (Tashkeel)")
def show_diacritized_text_modal():
    diacritized_text = st.session_state.get("diacritized_text", "")
    st.caption("Copy or edit the diacritized text below:")
    bidi_diacritized = process_text_for_display(diacritized_text)
    st.text_area("Tashkeel Text (Editable / Copyable)", value=bidi_diacritized, height=130, key="modal_diacritized_area")
    st.markdown("#### ✨ Diacritized Text (Large Font)")
    st.markdown(f"""
    <div dir="rtl" class="arabic-text" style="font-size: 24px !important; font-weight: normal !important; padding: 12px; border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; text-align: right; line-height: 1.8;">
        {diacritized_text}
    </div>
    """, unsafe_allow_html=True)

# Sidebar Navigation Menu
st.sidebar.title("📌 Navigation")
nav_page = st.sidebar.radio(
    "Go to page:",
    [
        "📖 Diacritizer & Analyzer",
        "📚 Saved History & Anki Export",
        "🔍 Combined Vocabulary",
        "🖼️ Saved Entry Gallery"
    ],
    key="nav_page_selection"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎙️ Text-to-Speech (TTS)")
tts_engine_choice = st.sidebar.radio(
    "Arabic Voice Engine:",
    [
        "🎙️ Microsoft Edge Neural (Hamed)",
        "🌐 Google Voice (gTTS)",
        "⚡ Browser Native (Web Speech API)"
    ],
    key="tts_engine_selector"
)
st.sidebar.markdown("---")

if nav_page == "📖 Diacritizer & Analyzer":
    # Sidebar Engine Selection
    st.sidebar.title("Settings")

    ocr_mode = st.sidebar.radio(
        "⚙️ OCR & Diacritization Engine",
        [
            "🟢 Hybrid Mode (EasyOCR + Gemini Text)",
            "🟣 Full Gemini Vision"
        ]
    )

    if ocr_mode == "🟢 Hybrid Mode (EasyOCR + Gemini Text)":
        st.sidebar.info("💡 **Tip:** Hybrid Mode uses local EasyOCR first ($0 cost), then sends raw text to Gemini 3.5 Flash-Lite to fix typos, add Tashkeel, and classify verbs/nouns.")
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

            col_prev, col_page, col_next = st.sidebar.columns([1, 2, 1])

            if col_prev.button("⬅️", disabled=(st.session_state.pdf_page <= 1), width="stretch"):
                st.session_state.pdf_page -= 1

            col_page.number_input(
                f"Page (1 of {num_pages})",
                min_value=1,
                max_value=num_pages,
                step=1,
                key="pdf_page",
                label_visibility="collapsed",
            )

            if col_next.button("➡️", disabled=(st.session_state.pdf_page >= num_pages), width="stretch"):
                st.session_state.pdf_page += 1
        
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
                is_pan_mode = st.toggle("🖐️ Pan Mode (Click & Drag to Scroll)", value=False, help="Enable this to click and drag the zoomed image. Disable to draw a crop box.")
            with col_ctrl2:
                st.write("")  # spacing
                st.write("")
                st.button("🔄 Reset View", use_container_width=True, on_click=reset_view_callback)
                
            # Fit image to container max width (700px) so giant images/PDFs fit cleanly in #root at 1.0x
            base_img = fit_image_to_max_width(img, max_width=700)

            # Obtain zoomed image, scaled up proportionally to magnify details for native scrolling
            if zoom_val > 1.0:
                new_w = int(base_img.width * zoom_val)
                new_h = int(base_img.height * zoom_val)
                zoomed_img = base_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                zoomed_img = base_img

            pointer_events = "none" if is_pan_mode else "auto"
            cursor_style = "grab" if is_pan_mode else "crosshair"

            # Inject dynamic CSS and JS for drag-to-scroll feature
            st.markdown(f"""
                <style>
                    /* Target the immediate wrapper of the iframe */
                    div:has(> iframe) {{
                        cursor: {cursor_style} !important;
                        overflow: auto !important;
                        max-height: 70vh !important;
                        max-width: 100% !important;
                        width: 100% !important;
                    }}
                    div:has(> iframe):active {{
                        cursor: {'grabbing' if is_pan_mode else 'crosshair'} !important;
                    }}
                    iframe {{
                        width: {zoomed_img.width}px !important;
                        height: {zoomed_img.height}px !important;
                        min-width: {zoomed_img.width}px !important;
                        min-height: {zoomed_img.height}px !important;
                        max-width: none !important;
                        pointer-events: {pointer_events} !important;
                    }}
                </style>
            """, unsafe_allow_html=True)

            # Interactive cropping tool on the zoomed image (key includes zoom value to force remount)
            cropper_key = f"cropper_{selected_file_name}_z{zoom_val}"
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

            if is_pan_mode:
                st.components.v1.html("""
                <script>
                    const parentDoc = window.parent.document;
                    // Find the container that actually has the scrollbars (parent of the iframe)
                    const iframes = parentDoc.querySelectorAll('iframe');
                    let cropper = null;
                    for (let iframe of iframes) {
                        const computedStyle = parentDoc.defaultView.getComputedStyle(iframe);
                        if (computedStyle.pointerEvents === 'none') {
                            cropper = iframe.parentElement;
                            break;
                        }
                    }
                    
                    if (cropper) {
                        let isDown = false;
                        let startX, startY, scrollLeft, scrollTop;

                        cropper.onmousedown = (e) => {
                            isDown = true;
                            startX = e.pageX - cropper.offsetLeft;
                            startY = e.pageY - cropper.offsetTop;
                            scrollLeft = cropper.scrollLeft;
                            scrollTop = cropper.scrollTop;
                        };
                        cropper.onmouseleave = () => { isDown = false; };
                        cropper.onmouseup = () => { isDown = false; };
                        cropper.onmousemove = (e) => {
                            if (!isDown) return;
                            e.preventDefault();
                            const x = e.pageX - cropper.offsetLeft;
                            const y = e.pageY - cropper.offsetTop;
                            const walkX = (x - startX) * 1.5;
                            const walkY = (y - startY) * 1.5;
                            cropper.scrollLeft = scrollLeft - walkX;
                            cropper.scrollTop = scrollTop - walkY;
                        };
                    }
                </script>
                """, height=0)
        
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
                                    # Call Gemini 3.1 Flash-Lite
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
                
                    # Top Toolbar: Centered Action Buttons & TTS
                    col_m1, col_m2, col_tts = st.columns([2, 2, 3])
                    with col_m1:
                        if st.button("📄 Raw Text", key="btn_open_raw_modal", icon=":material/description:", use_container_width=True):
                            show_raw_text_modal()
                    with col_m2:
                        if st.button("✨ Diacritized Text", key="btn_open_diacritized_modal", icon=":material/edit_note:", use_container_width=True):
                            show_diacritized_text_modal()
                    with col_tts:
                        render_arabic_tts(diacritized_text, tts_engine_choice, key_suffix="full_sentence")

                    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

                    # Tashkeel Text Header & Vowel Toggle
                    col_thdr, col_ttog = st.columns([3, 2])
                    with col_thdr:
                        st.markdown("#### ✨ Tashkeel Text (Hover for Tooltip)")
                    with col_ttog:
                        st.markdown("""
                        <div class="showvowels">
                        """, unsafe_allow_html=True)
                        show_vowels = st.toggle("Show Vowels", value=True, key="main_vowel_toggle")
                        st.markdown("</div>", unsafe_allow_html=True)

                    v_html, nv_html, sentence_tokens = build_interactive_tashkeel_html(
                        diacritized_text,
                        json.dumps(st.session_state.verbs),
                        json.dumps(st.session_state.nouns),
                        json.dumps(st.session_state.particles),
                    )
                    interactive_tashkeel_html = v_html if show_vowels else nv_html

                    # 24px, not bold, no dotted line, no border box
                    st.markdown(f"""
                    <div dir="rtl" class="arabic-text" style="font-size: 24px !important; font-weight: normal !important; border: none !important; background: transparent !important; padding: 4px 0 !important; color: inherit !important; text-align: right !important; line-height: 1.8 !important; margin-bottom: 15px;">
                        {interactive_tashkeel_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                    st.markdown("### English Translation")
                    if st.session_state.full_translation:
                        st.success(st.session_state.full_translation)
                    else:
                        try:
                            translated_text = GoogleTranslator(source='ar', target='en').translate(diacritized_text)
                            st.success(translated_text)
                        except Exception as e:
                            st.warning(f"Translation failed: {e}")
                
                    # Interactive Word Lookup Reader
                    st.markdown("---")
                    st.markdown("### 👆 Interactive Word Lookup (Click or Hover)")
                    st.caption("Click any word in the sentence below to view its instant contextual translation, grammar role, and root.")

                    selected_word = st.pills(
                        "Select a word from the diacritized sentence:",
                        options=sentence_tokens,
                        selection_mode="single",
                        key="interactive_sentence_pills"
                    )

                    if selected_word:
                        raw_selected = strip_tashkeel(selected_word)
                        info = word_map.get(selected_word) or word_map.get(raw_selected)
                        
                        # 1. Grab Gemini's root from lookup object
                        gemini_root = info.get("root") if info else None

                        if gemini_root and gemini_root != "N/A":
                            # Formats Gemini's "ب-س-م" cleanly to "ب - س - م"
                            root_display = " - ".join(gemini_root.replace("-", " ").split())
                        else:
                            # 2. Fallback to Farasa ONLY if Gemini didn't return a root
                            try:
                                stemmed = farasa_stemmer.stem(selected_word)
                                clean_root = re.sub(r'[^\u0621-\u064A]', '', stemmed)
                                root_display = " - ".join(list(clean_root)) if clean_root else "N/A"
                            except Exception:
                                root_display = "N/A"

                        # 3. Always render the card container (OUTSIDE the if/else block)
                        with st.container(border=True):
                            col_w1, col_w2 = st.columns([1, 2])
                            with col_w1:
                                st.markdown(f"""
                                <div dir="rtl" class="arabic-text arabic-large" style="color: #1E88E5; font-weight: bold; padding: 12px; border-radius: 6px; background-color: rgba(30, 136, 229, 0.08); text-align: center; margin-bottom: 8px;">
                                    {selected_word}
                                </div>
                                """, unsafe_allow_html=True)

                                # Word Audio Player Controls
                                if "Microsoft Edge" in tts_engine_choice:
                                    if st.button("🔊 Pronounce Word", key=f"btn_word_edge_{selected_word}"):
                                        try:
                                            w_audio = generate_edge_audio(selected_word)
                                            st.audio(w_audio, format="audio/mp3", autoplay=True)
                                        except Exception as e:
                                            st.error(f"TTS Error: {e}")
                                elif "Google Voice" in tts_engine_choice:
                                    if st.button("🔊 Pronounce Word", key=f"btn_word_gtts_{selected_word}"):
                                        try:
                                            w_audio = generate_gtts_audio(selected_word)
                                            st.audio(w_audio, format="audio/mp3", autoplay=True)
                                        except Exception as e:
                                            st.error(f"TTS Error: {e}")
                                else:
                                    esc_w = selected_word.replace("'", "\\'").replace('"', '\\"')
                                    html_word = f"""
                                    <button onclick="speakWord('{esc_w}')" style="background:#1E88E5; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; width:100%;">
                                        ⚡ 🔊 Pronounce Word
                                    </button>
                                    <script>
                                    function speakWord(w) {{
                                        if ('speechSynthesis' in window) {{
                                            window.speechSynthesis.cancel();
                                            const msg = new SpeechSynthesisUtterance(w);
                                            msg.lang = 'ar-SA';
                                            msg.rate = 0.8;
                                            const voices = window.speechSynthesis.getVoices();
                                            const v = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('ar'));
                                            if (v) msg.voice = v;
                                            window.speechSynthesis.speak(msg);
                                        }}
                                    }}
                                    </script>
                                    """
                                    components.html(html_word, height=40)
                            with col_w2:
                                if info:
                                    sub_type_val = info.get('sub_type', 'Noun')
                                    derived_val = info.get('derived', False)
                                    base_verb_val = info.get('base_verb')
                                    root_val = info.get('root') or root_display

                                    st.markdown(f"**💡 Contextual Meaning:** {info.get('meaning', 'N/A')}")
                                    st.markdown(f"**🏷️ Category / Role:** {info.get('role', 'Unknown')}")
                                    st.markdown(f"**🏷️ Morphological Type:** {render_sub_type_badge(sub_type_val, derived_val, base_verb_val)}", unsafe_allow_html=True)
                                    if info.get('effect'):
                                        st.markdown(f"**⚡ Grammatical Effect:** {info.get('effect')}")

                                    can_generate_sarf = derived_val or sub_type_val == 'Verb' or (base_verb_val is not None) or (root_val is not None)
                                    if can_generate_sarf:
                                        sarf_target_verb = base_verb_val if base_verb_val else selected_word
                                        if st.button(f"⚡ Generate Sarf for '{sarf_target_verb}'", key=f"btn_card_sarf_{selected_word}"):
                                            with st.spinner(f"Generating Sarf paradigm for base verb '{sarf_target_verb}'..."):
                                                try:
                                                    target_r = root_val if root_val else root_display
                                                    s_data = get_verb_analysis(sarf_target_verb, target_r)
                                                    st.session_state.verb_results = s_data
                                                    st.session_state.last_analyzed_verb = sarf_target_verb
                                                    auto_save_or_update_current_entry(
                                                        selected_file_name, cropped_img, diacritized_text,
                                                        st.session_state.full_translation, st.session_state.verbs,
                                                        st.session_state.nouns, st.session_state.particles,
                                                        st.session_state.verb_results, st.session_state.last_analyzed_verb,
                                                        st.session_state.noun_results, st.session_state.last_analyzed_noun
                                                    )
                                                    st.success(f"Generated Sarf paradigm for '{sarf_target_verb}'! Auto-saved to study database 💾")
                                                except Exception as e:
                                                    st.error(f"Sarf generation failed: {e}")
                                else:
                                    try:
                                        quick_trans = GoogleTranslator(source='ar', target='en').translate(selected_word)
                                    except Exception:
                                        quick_trans = "N/A"
                                    st.markdown(f"**💡 Contextual Meaning:** {quick_trans}")
                                    st.markdown("**🏷️ Category / Role:** Unclassified Word")

                                if root_display and root_display != "N/A":
                                    st.markdown(f"**🌱 Extracted Root:** `{root_display}`")
                
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
                                            auto_save_or_update_current_entry(
                                                selected_file_name, cropped_img, diacritized_text,
                                                st.session_state.full_translation, st.session_state.verbs,
                                                st.session_state.nouns, st.session_state.particles,
                                                st.session_state.verb_results, st.session_state.last_analyzed_verb,
                                                st.session_state.noun_results, st.session_state.last_analyzed_noun
                                            )
                                            st.toast("Auto-saved Verb Sarf analysis to database! 💾")
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
                                    render_custom_table(verb_df)
                                
                        with tab2:
                            st.subheader("Noun Analysis")
                            word_map = build_word_meaning_map(detected_verbs, detected_nouns, detected_particles)

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

                                # Look up noun metadata in word_map
                                sel_noun_obj = word_map.get(noun_to_analyze) or word_map.get(strip_tashkeel(noun_to_analyze)) or {}
                                n_derived = sel_noun_obj.get("derived", False)
                                n_sub_type = sel_noun_obj.get("sub_type", "Solid Noun")
                                n_base_verb = sel_noun_obj.get("base_verb")
                                n_root = sel_noun_obj.get("root") or clean_noun_root
                                
                                try:
                                    noun_trans = GoogleTranslator(source='ar', target='en').translate(noun_to_analyze)
                                    noun_root_trans = GoogleTranslator(source='ar', target='en').translate(clean_noun_root)
                                except Exception:
                                    noun_trans = "N/A"
                                    noun_root_trans = "N/A"
                                
                                st.markdown(f"""
                                <div style="font-size: 18px; margin-bottom: 10px;">
                                    Analyzing noun: <span class="arabic-text arabic-medium" style="font-weight: bold; color: #1E88E5;">{noun_to_analyze}</span> (Translation: <em>{noun_trans}</em>) {render_sub_type_badge(n_sub_type, n_derived, n_base_verb)}
                                </div>
                                """, unsafe_allow_html=True)
                            
                                st.info(f"🌱 Extracted Local Root: {noun_root_display} (Translation: {noun_root_trans})")
                            
                                # Clear results if the noun changes
                                if st.session_state.last_analyzed_noun != noun_to_analyze:
                                    st.session_state.noun_results = None
                                
                                api_key = get_gemini_api_key()
                                if not api_key:
                                    st.warning("⚠️ Gemini API key is missing. Please configure it to enable Noun analysis.")
                                
                                # NEW LOGIC: Render Sarf button ADJACENT to Noun Analysis if derived == True or base_verb is present
                                is_noun_derived = n_derived or (n_sub_type in ["Ism Fa'il", "Ism Maf'ul", "Masdar", "Sifah Mushabbahah", "Verb"]) or (n_base_verb is not None)

                                if is_noun_derived:
                                    col_nbtn1, col_nbtn2 = st.columns(2)
                                    with col_nbtn1:
                                        if st.button("Analyze Noun via Gemini", key="btn_analyze_noun", disabled=(not api_key), use_container_width=True):
                                            with st.spinner("Analyzing noun forms..."):
                                                try:
                                                    noun_data = get_noun_analysis(noun_to_analyze, clean_noun_root)
                                                    st.session_state.noun_results = noun_data
                                                    st.session_state.last_analyzed_noun = noun_to_analyze
                                                    auto_save_or_update_current_entry(
                                                        selected_file_name, cropped_img, diacritized_text,
                                                        st.session_state.full_translation, st.session_state.verbs,
                                                        st.session_state.nouns, st.session_state.particles,
                                                        st.session_state.verb_results, st.session_state.last_analyzed_verb,
                                                        st.session_state.noun_results, st.session_state.last_analyzed_noun
                                                    )
                                                    st.toast("Auto-saved Noun analysis to database! 💾")
                                                except Exception as e:
                                                    st.error(f"Gemini Noun Analysis failed: {e}")
                                                    st.session_state.noun_results = None
                                    with col_nbtn2:
                                        sarf_verb_target = n_base_verb if n_base_verb else noun_to_analyze
                                        if st.button(f"⚡ Generate Sarf for '{sarf_verb_target}'", key="btn_analyze_noun_sarf", disabled=(not api_key), use_container_width=True):
                                            with st.spinner(f"Generating Sarf paradigm for base verb '{sarf_verb_target}'..."):
                                                try:
                                                    verb_data = get_verb_analysis(sarf_verb_target, n_root)
                                                    st.session_state.verb_results = verb_data
                                                    st.session_state.last_analyzed_verb = sarf_verb_target
                                                    auto_save_or_update_current_entry(
                                                        selected_file_name, cropped_img, diacritized_text,
                                                        st.session_state.full_translation, st.session_state.verbs,
                                                        st.session_state.nouns, st.session_state.particles,
                                                        st.session_state.verb_results, st.session_state.last_analyzed_verb,
                                                        st.session_state.noun_results, st.session_state.last_analyzed_noun
                                                    )
                                                    st.success(f"Generated Sarf paradigm for '{sarf_verb_target}'! Auto-saved to study database 💾")
                                                except Exception as e:
                                                    st.error(f"Sarf generation failed: {e}")
                                else:
                                    if st.button("Analyze Noun via Gemini", key="btn_analyze_noun", disabled=(not api_key)):
                                        with st.spinner("Analyzing noun forms..."):
                                            try:
                                                noun_data = get_noun_analysis(noun_to_analyze, clean_noun_root)
                                                st.session_state.noun_results = noun_data
                                                st.session_state.last_analyzed_noun = noun_to_analyze
                                                auto_save_or_update_current_entry(
                                                    selected_file_name, cropped_img, diacritized_text,
                                                    st.session_state.full_translation, st.session_state.verbs,
                                                    st.session_state.nouns, st.session_state.particles,
                                                    st.session_state.verb_results, st.session_state.last_analyzed_verb,
                                                    st.session_state.noun_results, st.session_state.last_analyzed_noun
                                                )
                                                st.toast("Auto-saved Noun analysis to database! 💾")
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
                                    render_custom_table(noun_df)

                                # Render Verb Sarf paradigm independently if generated for this derived noun
                                if st.session_state.get('verb_results'):
                                    v_data = st.session_state.verb_results
                                    sarf_title_verb = st.session_state.get('last_analyzed_verb', 'Base Verb')
                                    st.markdown("---")
                                    st.markdown(f"#### ⚡ Verbal Sarf Paradigm for Base Verb `{sarf_title_verb}`")
                                    col_v1, col_v2, col_v3 = st.columns(3)
                                    with col_v1:
                                        st.metric(label="Pattern / Form (الوزن)", value=v_data.get('wazn', 'N/A'))
                                    with col_v2:
                                        st.metric(label="Past Tense (الماضي)", value=v_data.get('madi', 'N/A'))
                                    with col_v3:
                                        st.metric(label="Present Tense (المضارع)", value=v_data.get('mudari', 'N/A'))

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
                                    render_custom_table(verb_df)
                                
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
                                render_custom_table(p_df)
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
                                st.success("Saved entry to study database! 🎉 Switch to the 📚 Saved History page in the sidebar to inspect it anytime.")
                            else:
                                st.warning("Please process a crop first before saving to database.")

    else:
        st.info("Please upload one or more manga pages to begin.")

elif nav_page == "📚 Saved History & Anki Export":
    st.title("📚 Saved Study History & Anki Flashcard Exporter")
    
    entries = get_all_study_entries()
    if entries:
        # 1. Left Sidebar Entry Selector List
        st.sidebar.markdown("### 📚 Saved Entries")
        entry_map = {}
        dropdown_options = []
        
        for row in entries:
            entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = row
            snippet = tashkeel[:25] + "..." if len(tashkeel) > 25 else tashkeel
            opt_str = f"[#{entry_id}] {timestamp} | {snippet}"
            dropdown_options.append(opt_str)
            entry_map[opt_str] = row
            
        selected_option = st.sidebar.radio(
            "Select entry to inspect:",
            options=dropdown_options,
            key="sidebar_entry_radio"
        )


        # 2. Main Area TOP: Summary Table & Anki Deck Export
        st.subheader(f"Summary Table & Anki Deck Export ({len(entries)} Entries)")
        
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

        col_df, col_anki = st.columns([3, 1])
        with col_df:
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True)
        with col_anki:
            df_anki = pd.DataFrame(anki_rows)
            csv_buffer = df_anki.to_csv(index=False).encode('utf-8')
            st.write("")
            st.download_button(
                label="📥 Download Anki CSV",
                data=csv_buffer,
                file_name="arabic_manga_anki_deck.csv",
                mime="text/csv",
                key="download_anki_csv",
                use_container_width=True
            )

        st.markdown("---")

        # 3. Full Entry Inspector View Below
        selected_row = entry_map[selected_option]
        entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = selected_row

        col_hdr, col_del, col_htts = st.columns([4, 1, 1])
        with col_hdr:
            st.markdown(f"### Inspector View: Entry #{entry_id} (`{fname}` — *{timestamp}*)")
        with col_del:
            if st.button("🗑️ Delete Entry", key=f"btn_del_{entry_id}", use_container_width=True):
                delete_study_entry(entry_id)
                st.success(f"Deleted entry #{entry_id}!")
                st.rerun()
        with col_htts:
                render_arabic_tts(tashkeel, tts_engine_choice, key_suffix=f"hist_{entry_id}")

        st.markdown('<div class="saved-inspector">', unsafe_allow_html=True)

        st.markdown("""
        <style>
        @media (max-width: 1024px) {
            .saved-inspector [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }
            .saved-inspector [data-testid="stHorizontalBlock"] > [data-testid="column"],
            .saved-inspector [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                width: 100% !important;
                flex: 0 0 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
        }
        .tashkeeled {
            background-color: rgba(240,94,86, 0.15) !important;
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 12px;
        }
        </style>
        """, unsafe_allow_html=True)

        col_img, col_details = st.columns([2, 2])

        with col_img:
            st.markdown("#### ✨ Captured Image/pdf")
            if img_b64:
                try:
                    img_bytes = base64.b64decode(img_b64)
                    saved_pil_img = Image.open(io.BytesIO(img_bytes))
                    st.image(saved_pil_img, caption=f"Saved Crop ({fname})", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {e}")
            else:
                st.caption("No image data stored.")
                
        # Parse saved JSON fields early
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

        # Build word map & tokens for hover tooltips and word chips
        saved_word_map = build_word_meaning_map(saved_verbs, saved_nouns, saved_particles)
        saved_tokens = [w.strip() for w in re.split(r'[\s،؛؟\.\!\:\-"\']+', tashkeel) if w.strip()]

        with col_details:
            col_hlbl, col_htog = st.columns([4, 1])
            with col_hlbl:
                st.markdown("#### ✨ Tashkeel Text")
            with col_htog:
                show_saved_vowels = st.toggle("Show Vowels", value=True, key=f"hist_vowel_toggle_{entry_id}")

            v_html, nv_html, _ = build_interactive_tashkeel_html(
                tashkeel,
                verbs_str or "",
                nouns_str or "",
                particles_str or "",
            )
            interactive_saved_tashkeel = v_html if show_saved_vowels else nv_html

            st.markdown(f"""
            <div dir="rtl" class="arabic-text tashkeeled">
            {interactive_saved_tashkeel}
            </div>
            """, unsafe_allow_html=True)
            
            if translation:
                st.info(f"💡 **English Translation:** {translation}")
            else:
                st.caption("No translation recorded.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Interactive Word Lookup Reader for Saved Entry
        st.markdown("#### 👆 Interactive Word Lookup (Click Word to Inspect)")
        selected_hist_word = st.pills(
            "Select a word from this saved entry:",
            options=saved_tokens,
            selection_mode="single",
            key=f"hist_word_pills_{entry_id}"
        )

        if selected_hist_word:
            raw_sel = strip_tashkeel(selected_hist_word)
            h_info = saved_word_map.get(selected_hist_word) or saved_word_map.get(raw_sel)
            
            try:
                stemmed = farasa_stemmer.stem(selected_hist_word)
                clean_root = re.sub(r'[^\u0621-\u064A]', '', stemmed)
                root_display = " - ".join(list(clean_root)) if clean_root else "N/A"
            except Exception:
                root_display = "N/A"

            with st.container(border=True):
                col_hw1, col_hw2 = st.columns([1, 2])
                with col_hw1:
                    st.markdown(f"""
                    <div dir="rtl" class="arabic-text arabic-large" style="color: #1E88E5; font-weight: bold; padding: 12px; border-radius: 6px; background-color: rgba(30, 136, 229, 0.08); text-align: center; margin-bottom: 8px;">
                        {selected_hist_word}
                    </div>
                    """, unsafe_allow_html=True)

                    # Word Audio Player Controls
                    if "Microsoft Edge" in tts_engine_choice:
                        if st.button("🔊 Pronounce Word", key=f"btn_hist_edge_{entry_id}_{selected_hist_word}"):
                            try:
                                w_audio = generate_edge_audio(selected_hist_word)
                                st.audio(w_audio, format="audio/mp3", autoplay=True)
                            except Exception as e:
                                st.error(f"TTS Error: {e}")
                    elif "Google Voice" in tts_engine_choice:
                        if st.button("🔊 Pronounce Word", key=f"btn_hist_gtts_{entry_id}_{selected_hist_word}"):
                            try:
                                w_audio = generate_gtts_audio(selected_hist_word)
                                st.audio(w_audio, format="audio/mp3", autoplay=True)
                            except Exception as e:
                                st.error(f"TTS Error: {e}")
                    else:
                        esc_w = selected_hist_word.replace("'", "\\'").replace('"', '\\"')
                        html_word = f"""
                        <button onclick="speakWord('{esc_w}')" style="background:#1E88E5; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; width:100%;">
                            ⚡ 🔊 Pronounce Word
                        </button>
                        <script>
                        function speakWord(w) {{
                            if ('speechSynthesis' in window) {{
                                window.speechSynthesis.cancel();
                                const msg = new SpeechSynthesisUtterance(w);
                                msg.lang = 'ar-SA';
                                msg.rate = 0.8;
                                const voices = window.speechSynthesis.getVoices();
                                const v = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('ar'));
                                if (v) msg.voice = v;
                                window.speechSynthesis.speak(msg);
                            }}
                        }}
                        </script>
                        """
                        components.html(html_word, height=40)
                with col_hw2:
                    if h_info:
                        h_sub_type = h_info.get('sub_type', 'Noun')
                        h_derived = h_info.get('derived', False)
                        h_base_verb = h_info.get('base_verb')
                        h_root = h_info.get('root') or clean_root

                        st.markdown(f"**💡 Contextual Meaning:** {h_info.get('meaning', 'N/A')}")
                        st.markdown(f"**🏷️ Category / Role:** {h_info.get('role', 'Unknown')}")
                        st.markdown(f"**🏷️ Morphological Type:** {render_sub_type_badge(h_sub_type, h_derived, h_base_verb)}", unsafe_allow_html=True)
                        if h_info.get('effect'):
                            st.markdown(f"**⚡ Grammatical Effect:** {h_info.get('effect')}")

                        # "Generate Sarf" button condition for saved word
                        can_hist_sarf = h_derived or h_sub_type == 'Verb' or (h_base_verb is not None) or (h_root is not None)
                        if can_hist_sarf:
                            h_sarf_target = h_base_verb if h_base_verb else selected_hist_word
                            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                            if st.button(f"⚡ Generate Sarf for '{h_sarf_target}'", key=f"btn_hist_sarf_{entry_id}_{selected_hist_word}"):
                                with st.spinner(f"Generating Sarf paradigm for base verb '{h_sarf_target}'..."):
                                    try:
                                        h_target_r = h_root if h_root else clean_root
                                        s_data = get_verb_analysis(h_sarf_target, h_target_r)
                                        st.session_state.verb_results = s_data
                                        st.session_state.last_analyzed_verb = h_sarf_target
                                        updated_deep_sarf = {
                                            "last_verb": h_sarf_target,
                                            "verb_sarf": s_data,
                                            "last_noun": saved_deep_sarf.get("last_noun"),
                                            "noun_sarf": saved_deep_sarf.get("noun_sarf")
                                        }
                                        update_study_entry_sarf(entry_id, updated_deep_sarf)
                                        st.success(f"Generated Sarf paradigm for '{h_sarf_target}'! Auto-updated Saved Entry #{entry_id} in study database 💾")
                                    except Exception as e:
                                        st.error(f"Sarf generation failed: {e}")
                    else:
                        try:
                            quick_trans = GoogleTranslator(source='ar', target='en').translate(selected_hist_word)
                        except Exception:
                            quick_trans = "N/A"
                        st.markdown(f"**💡 Contextual Meaning:** {quick_trans}")
                        st.markdown("**🏷️ Category / Role:** Unclassified Word")

                    if root_display and root_display != "N/A":
                        st.markdown(f"**🌱 Extracted Root:** `{root_display}`")

        # Categorized Word Tabs for Saved Entry
        st.markdown("#### Categorized Vocabulary Breakdown")
        tab_v, tab_n, tab_p = st.tabs(["⚙️ Saved Verbs", "🏷️ Saved Nouns", "📌 Saved Particles"])
        
        with tab_v:
            if saved_verbs:
                v_rows = []
                v_options = []
                v_map = {}
                for v in saved_verbs:
                    if isinstance(v, dict):
                        w = v.get("word", "N/A")
                        m = v.get("meaning", "N/A")
                        b = v.get("base_verb") or w
                        v_rows.append({"Word (الْكَلِمَة)": w, "Form/Base (الأَصْل)": b, "Meaning (الْمَعْنَى)": m})
                        disp = f"{w} ({m})" if m != "N/A" else w
                        v_options.append(disp)
                        v_map[disp] = w
                    else:
                        w = str(v)
                        v_rows.append({"Word (الْكَلِمَة)": w, "Form/Base (الأَصْل)": "N/A", "Meaning (الْمَعْنَى)": "N/A"})
                        v_options.append(w)
                        v_map[w] = w
                
                render_custom_table(pd.DataFrame(v_rows))
                
                st.markdown("---")
                st.markdown("##### ⚡ Analyze & Conjugate Saved Verb")
                sel_hv_disp = st.selectbox("Select a verb to analyze / generate Sarf:", options=v_options, key=f"hist_v_sel_{entry_id}")
                hv_to_analyze = v_map.get(sel_hv_disp, sel_hv_disp)
                
                if hv_to_analyze:
                    try:
                        h_stemmed_v = farasa_stemmer.stem(hv_to_analyze)
                        h_clean_v_root = re.sub(r'[^\u0621-\u064A]', '', h_stemmed_v)
                        h_v_root_disp = " - ".join(list(h_clean_v_root))
                    except Exception:
                        h_clean_v_root = hv_to_analyze
                        h_v_root_disp = "Unknown"
                    
                    st.info(f"🌱 Extracted Local Root: `{h_v_root_disp}`")
                    
                    api_key = get_gemini_api_key()
                    if st.button("Analyze Verb & Generate Sarf via Gemini", key=f"btn_hist_analyze_v_{entry_id}", disabled=(not api_key)):
                        with st.spinner("Analyzing verb paradigm via Gemini..."):
                            try:
                                v_data = get_verb_analysis(hv_to_analyze, h_clean_v_root)
                                updated_deep = {
                                    "last_verb": hv_to_analyze,
                                    "verb_sarf": v_data,
                                    "last_noun": saved_deep_sarf.get("last_noun"),
                                    "noun_sarf": saved_deep_sarf.get("noun_sarf")
                                }
                                update_study_entry_sarf(entry_id, updated_deep)
                                st.session_state[f"hist_v_res_{entry_id}"] = v_data
                                st.success(f"Generated and auto-saved Sarf paradigm for '{hv_to_analyze}' to Entry #{entry_id}! 💾")
                            except Exception as e:
                                st.error(f"Verb analysis failed: {e}")
                    
                    hist_v_res = st.session_state.get(f"hist_v_res_{entry_id}") or (saved_deep_sarf.get("verb_sarf") if saved_deep_sarf.get("last_verb") == hv_to_analyze else None)
                    if hist_v_res:
                        col_hv1, col_hv2, col_hv3 = st.columns(3)
                        with col_hv1:
                            st.metric(label="Pattern / Form (الوزن)", value=hist_v_res.get('wazn', 'N/A'))
                        with col_hv2:
                            st.metric(label="Past Tense (الماضي)", value=hist_v_res.get('madi', 'N/A'))
                        with col_hv3:
                            st.metric(label="Present Tense (المضارع)", value=hist_v_res.get('mudari', 'N/A'))
                        
                        st.markdown("##### Morphological paradigm table")
                        w_to_trans = [
                            hist_v_res.get('madi', ''),
                            hist_v_res.get('mudari', ''),
                            hist_v_res.get('amr', ''),
                            hist_v_res.get('masdar', ''),
                            hist_v_res.get('ism_faail', ''),
                            hist_v_res.get('ism_mafool', '')
                        ]
                        trans_v = translate_words(w_to_trans)
                        h_v_df = pd.DataFrame({
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
                                hist_v_res.get('wazn', 'N/A'),
                                hist_v_res.get('madi', 'N/A'),
                                hist_v_res.get('mudari', 'N/A'),
                                hist_v_res.get('amr', 'N/A'),
                                hist_v_res.get('masdar', 'N/A'),
                                hist_v_res.get('ism_faail', 'N/A'),
                                hist_v_res.get('ism_mafool', 'N/A')
                            ],
                            "English Translation": [
                                "N/A",
                                trans_v[0],
                                trans_v[1],
                                trans_v[2],
                                trans_v[3],
                                trans_v[4],
                                trans_v[5]
                            ]
                        })
                        render_custom_table(h_v_df)
            else:
                st.caption("No verbs recorded for this entry.")
                
        with tab_n:
            if saved_nouns:
                n_rows = []
                n_options = []
                n_map = {}
                for n in saved_nouns:
                    if isinstance(n, dict):
                        w = n.get("word", "N/A")
                        sub_t = n.get("sub_type", "Solid Noun")
                        is_d = n.get("derived", False)
                        b_v = n.get("base_verb")
                        m = n.get("meaning", "N/A")
                        
                        n_rows.append({
                            "Word (الْكَلِمَة)": w,
                            "Type (النَّوْع)": sub_t,
                            "Derived (مُشْتَقّ)": "Yes ⚡" if is_d else "No",
                            "Base Verb (الأَصْل)": b_v if b_v else "N/A",
                            "Meaning (الْمَعْنَى)": m
                        })
                        disp = f"{w} ({sub_t} - {m})" if m != "N/A" else w
                        n_options.append(disp)
                        n_map[disp] = n
                    else:
                        w = str(n)
                        n_rows.append({
                            "Word (الْكَلِمَة)": w,
                            "Type (النَّوْع)": "Solid Noun",
                            "Derived (مُشْتَقّ)": "No",
                            "Base Verb (الأَصْل)": "N/A",
                            "Meaning (الْمَعْنَى)": "N/A"
                        })
                        n_options.append(w)
                        n_map[w] = {"word": w}
                
                render_custom_table(pd.DataFrame(n_rows))
                
                st.markdown("---")
                st.markdown("##### ⚡ Analyze Saved Noun / Generate Sarf")
                sel_hn_disp = st.selectbox("Select a noun to analyze:", options=n_options, key=f"hist_n_sel_{entry_id}")
                hn_obj = n_map.get(sel_hn_disp, {})
                hn_to_analyze = hn_obj.get("word", sel_hn_disp)
                hn_sub_t = hn_obj.get("sub_type", "Solid Noun")
                hn_is_d = hn_obj.get("derived", False)
                hn_b_v = hn_obj.get("base_verb")
                hn_root_val = hn_obj.get("root")
                
                if hn_to_analyze:
                    try:
                        h_stemmed_n = farasa_stemmer.stem(hn_to_analyze)
                        h_clean_n_root = re.sub(r'[^\u0621-\u064A]', '', h_stemmed_n)
                        h_n_root_disp = " - ".join(list(h_clean_n_root))
                    except Exception:
                        h_clean_n_root = hn_to_analyze
                        h_n_root_disp = "Unknown"
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(33, 150, 243, 0.08); border-left: 4px solid #2196F3; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px; font-size: 15px;">
                        🌱 <strong>Extracted Local Root:</strong> <code>{h_n_root_disp}</code> {render_sub_type_badge(hn_sub_t, hn_is_d, hn_b_v)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    api_key = get_gemini_api_key()
                    is_hn_derived = hn_is_d or (hn_sub_t in ["Ism Fa'il", "Ism Maf'ul", "Masdar", "Sifah Mushabbahah", "Verb"]) or (hn_b_v is not None)
                    
                    if is_hn_derived:
                        col_hnb1, col_hnb2 = st.columns(2)
                        with col_hnb1:
                            if st.button("Analyze Noun via Gemini", key=f"btn_hist_an_n_{entry_id}", disabled=(not api_key), use_container_width=True):
                                with st.spinner("Analyzing noun forms via Gemini..."):
                                    try:
                                        n_data = get_noun_analysis(hn_to_analyze, h_clean_n_root)
                                        updated_deep = {
                                            "last_verb": saved_deep_sarf.get("last_verb"),
                                            "verb_sarf": saved_deep_sarf.get("verb_sarf"),
                                            "last_noun": hn_to_analyze,
                                            "noun_sarf": n_data
                                        }
                                        update_study_entry_sarf(entry_id, updated_deep)
                                        st.session_state[f"hist_n_res_{entry_id}"] = n_data
                                        st.success(f"Analyzed and auto-saved Noun declension for '{hn_to_analyze}' to Entry #{entry_id}! 💾")
                                    except Exception as e:
                                        st.error(f"Noun analysis failed: {e}")
                        with col_hnb2:
                            sarf_hverb_target = hn_b_v if hn_b_v else hn_to_analyze
                            if st.button(f"⚡ Generate Sarf for '{sarf_hverb_target}'", key=f"btn_hist_an_n_sarf_{entry_id}", disabled=(not api_key), use_container_width=True):
                                with st.spinner(f"Generating Sarf paradigm for '{sarf_hverb_target}' via Gemini..."):
                                    try:
                                        v_data = get_verb_analysis(sarf_hverb_target, hn_root_val or h_clean_n_root)
                                        updated_deep = {
                                            "last_verb": sarf_hverb_target,
                                            "verb_sarf": v_data,
                                            "last_noun": saved_deep_sarf.get("last_noun"),
                                            "noun_sarf": saved_deep_sarf.get("noun_sarf")
                                        }
                                        update_study_entry_sarf(entry_id, updated_deep)
                                        st.session_state[f"hist_v_res_{entry_id}"] = v_data
                                        st.success(f"Generated and auto-saved Sarf paradigm for '{sarf_hverb_target}' to Entry #{entry_id}! View in Saved Verbs tab. 💾")
                                    except Exception as e:
                                        st.error(f"Sarf generation failed: {e}")
                    else:
                        if st.button("Analyze Noun via Gemini", key=f"btn_hist_an_n_{entry_id}", disabled=(not api_key)):
                            with st.spinner("Analyzing noun forms via Gemini..."):
                                try:
                                    n_data = get_noun_analysis(hn_to_analyze, h_clean_n_root)
                                    updated_deep = {
                                        "last_verb": saved_deep_sarf.get("last_verb"),
                                        "verb_sarf": saved_deep_sarf.get("verb_sarf"),
                                        "last_noun": hn_to_analyze,
                                        "noun_sarf": n_data
                                    }
                                    update_study_entry_sarf(entry_id, updated_deep)
                                    st.session_state[f"hist_n_res_{entry_id}"] = n_data
                                    st.success(f"Analyzed and auto-saved Noun declension for '{hn_to_analyze}' to Entry #{entry_id}! 💾")
                                except Exception as e:
                                    st.error(f"Noun analysis failed: {e}")

                    hist_n_res = st.session_state.get(f"hist_n_res_{entry_id}") or (saved_deep_sarf.get("noun_sarf") if saved_deep_sarf.get("last_noun") == hn_to_analyze else None)
                    if hist_n_res:
                        col_hn1, col_hn2, col_hn3 = st.columns(3)
                        with col_hn1:
                            st.metric(label="Classification (النوع)", value=hist_n_res.get('noun_type', 'N/A'))
                        with col_hn2:
                            st.metric(label="Category (الفئة)", value=hist_n_res.get('category', 'N/A'))
                        with col_hn3:
                            st.metric(label="Pattern / Weight (الوزن)", value=hist_n_res.get('wazn', 'N/A'))
                        
                        st.markdown("##### Noun paradigm table")
                        w_n_trans = [
                            hist_n_res.get('singular', ''),
                            hist_n_res.get('dual', ''),
                            hist_n_res.get('plural', ''),
                            hist_n_res.get('root_verb', '')
                        ]
                        trans_n = translate_words(w_n_trans)
                        h_n_df = pd.DataFrame({
                            "Grammatical Form": [
                                "Pattern / Weight (الوزن)",
                                "Singular Form (الْمُفْرَد)",
                                "Dual Form (الْمُثَنَّى)",
                                "Plural Form (الْجَمْع)",
                                "Root Past Verb (الْفِعْل الأَصْلِي)"
                            ],
                            "Arabic Word (with Tashkeel)": [
                                hist_n_res.get('wazn', 'N/A'),
                                hist_n_res.get('singular', 'N/A'),
                                hist_n_res.get('dual', 'N/A'),
                                hist_n_res.get('plural', 'N/A'),
                                hist_n_res.get('root_verb', 'N/A')
                            ],
                            "English Translation": [
                                "N/A",
                                trans_n[0],
                                trans_n[1],
                                trans_n[2],
                                trans_n[3]
                            ]
                        })
                        render_custom_table(h_n_df)

                # Render Verb Sarf paradigm independently if generated for this derived noun inside History tab_n
                hist_v_res_noun_context = st.session_state.get(f"hist_v_res_{entry_id}") or saved_deep_sarf.get("verb_sarf")
                if hist_v_res_noun_context:
                    h_v_title = st.session_state.get('last_analyzed_verb') or saved_deep_sarf.get("last_verb") or "Base Verb"
                    st.markdown("---")
                    st.markdown(f"##### ⚡ Verbal Sarf Paradigm for Base Verb `{h_v_title}`")
                    col_hsv1, col_hsv2, col_hsv3 = st.columns(3)
                    with col_hsv1:
                        st.metric(label="Pattern / Form (الوزن)", value=hist_v_res_noun_context.get('wazn', 'N/A'))
                    with col_hsv2:
                        st.metric(label="Past Tense (الماضي)", value=hist_v_res_noun_context.get('madi', 'N/A'))
                    with col_hsv3:
                        st.metric(label="Present Tense (المضارع)", value=hist_v_res_noun_context.get('mudari', 'N/A'))

                    w_hv_trans = [
                        hist_v_res_noun_context.get('madi', ''),
                        hist_v_res_noun_context.get('mudari', ''),
                        hist_v_res_noun_context.get('amr', ''),
                        hist_v_res_noun_context.get('masdar', ''),
                        hist_v_res_noun_context.get('ism_faail', ''),
                        hist_v_res_noun_context.get('ism_mafool', '')
                    ]
                    trans_hv = translate_words(w_hv_trans)
                    h_v_noun_df = pd.DataFrame({
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
                            hist_v_res_noun_context.get('wazn', 'N/A'),
                            hist_v_res_noun_context.get('madi', 'N/A'),
                            hist_v_res_noun_context.get('mudari', 'N/A'),
                            hist_v_res_noun_context.get('amr', 'N/A'),
                            hist_v_res_noun_context.get('masdar', 'N/A'),
                            hist_v_res_noun_context.get('ism_faail', 'N/A'),
                            hist_v_res_noun_context.get('ism_mafool', 'N/A')
                        ],
                        "English Translation": [
                            "N/A",
                            trans_hv[0],
                            trans_hv[1],
                            trans_hv[2],
                            trans_hv[3],
                            trans_hv[4],
                            trans_hv[5]
                        ]
                    })
                    render_custom_table(h_v_noun_df)
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
                render_custom_table(pd.DataFrame(p_rows))
            else:
                st.caption("No particles recorded for this entry.")

        # Deep Sarf Cache Viewer
        if saved_deep_sarf and (saved_deep_sarf.get("verb_sarf") or saved_deep_sarf.get("noun_sarf")):
            with st.expander("📖 Saved Deep Sarf Analysis", expanded=False):
                if saved_deep_sarf.get("verb_sarf"):
                    st.markdown(f"**Verb Sarf for '{saved_deep_sarf.get('last_verb', '')}':**")
                    st.json(saved_deep_sarf["verb_sarf"])
                if saved_deep_sarf.get("noun_sarf"):
                    st.markdown(f"**Noun Sarf for '{saved_deep_sarf.get('last_noun', '')}':**")
                    st.json(saved_deep_sarf["noun_sarf"])
    else:
        st.info("No entries saved in the database yet. Process a crop on the Diacritizer page and click '💾 Save Entry to Database'!")
elif nav_page == "🔍 Combined Vocabulary":
    st.title("🔍 Combined Vocabulary Across All Entries")

    entries = get_all_study_entries()
    if not entries:
        st.info("No entries saved in the database yet. Process a crop on the 📖 Diacritizer & Analyzer page and save an entry.")
    else:
        all_verbs, all_nouns, all_particles = aggregate_vocabulary_across_entries(entries)
        total_verb_occ = sum(r["count"] for r in all_verbs)
        total_noun_occ = sum(r["count"] for r in all_nouns)
        total_particle_occ = sum(r["count"] for r in all_particles)

        st.markdown(
            f"This page aggregates and deduplicates every verb, noun, and particle extracted from **{len(entries)}** "
            f"saved entries, giving you a master vocabulary list with source references."
        )
        st.space("medium")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Saved entries", value=len(entries))
        with col2:
            st.metric(label="Unique verbs (فِعْل)", value=len(all_verbs))
        with col3:
            st.metric(label="Unique nouns (اسْم)", value=len(all_nouns))
        with col4:
            st.metric(label="Total occurrences", value=total_verb_occ + total_noun_occ + total_particle_occ)

        st.space("medium")
        tab_v, tab_n, tab_p = st.tabs([
            f"⚙️ All Verbs ({len(all_verbs)})",
            f"🏷️ All Nouns ({len(all_nouns)})",
            f"📌 All Particles ({len(all_particles)})",
        ])

        with tab_v:
            if all_verbs:
                v_rows = []
                for v in all_verbs:
                    sources_str = ", ".join(f"#{s['id']}" for s in v["sources"])
                    v_rows.append({
                        "Word (الْكَلِمَة)": v["word"],
                        "Meaning (الْمَعْنَى)": v["meaning"] or "N/A",
                        "Type (النَّوْع)": v["sub_type"],
                        "Derived (مُشْتَقّ)": "Yes ⚡" if v["derived"] else "No",
                        "Base Verb (الأَصْل)": v["base_verb"] if v["base_verb"] else "N/A",
                        "Root (الْجَذْر)": v["root"] if v["root"] else "N/A",
                        "Occurrences": v["count"],
                        "Source Entries (#)": sources_str,
                    })
                v_df = pd.DataFrame(v_rows)
                st.caption(
                    f"Showing {len(v_df)} unique verbs, deduplicated across all saved entries. "
                    ":orange[Occurrences] counts how many entries each verb appeared in."
                )
                render_custom_table(v_df)
                v_csv = v_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download all verbs as CSV",
                    icon=":material/download:",
                    data=v_csv,
                    file_name="combined_verbs.csv",
                    mime="text/csv",
                    key="download_combined_verbs",
                    width="stretch",
                )
            else:
                st.caption("No verbs found across saved entries.")

        with tab_n:
            if all_nouns:
                n_rows = []
                for n in all_nouns:
                    sources_str = ", ".join(f"#{s['id']}" for s in n["sources"])
                    n_rows.append({
                        "Word (الْكَلِمَة)": n["word"],
                        "Meaning (الْمَعْنَى)": n["meaning"] or "N/A",
                        "Type (النَّوْع)": n["sub_type"],
                        "Derived (مُشْتَقّ)": "Yes ⚡" if n["derived"] else "No",
                        "Base Verb (الأَصْل)": n["base_verb"] if n["base_verb"] else "N/A",
                        "Root (الْجَذْر)": n["root"] if n["root"] else "N/A",
                        "Occurrences": n["count"],
                        "Source Entries (#)": sources_str,
                    })
                n_df = pd.DataFrame(n_rows)
                st.caption(
                    f"Showing {len(n_df)} unique nouns, deduplicated across all saved entries. "
                    ":orange[Occurrences] counts how many entries each noun appeared in."
                )
                render_custom_table(n_df)
                n_csv = n_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download all nouns as CSV",
                    icon=":material/download:",
                    data=n_csv,
                    file_name="combined_nouns.csv",
                    mime="text/csv",
                    key="download_combined_nouns",
                    width="stretch",
                )
            else:
                st.caption("No nouns found across saved entries.")

        with tab_p:
            if all_particles:
                p_rows = []
                for p in all_particles:
                    sources_str = ", ".join(f"#{s['id']}" for s in p["sources"])
                    p_rows.append({
                        "Word (الْحَرْف)": p["word"],
                        "Category (النَّوْع)": p["category"] or "N/A",
                        "Meaning (الْمَعْنَى)": p["meaning"] or "N/A",
                        "Grammatical Effect (الأَثَر الإِعْرَابِي)": p["effect"] or "N/A",
                        "Occurrences": p["count"],
                        "Source Entries (#)": sources_str,
                    })
                p_df = pd.DataFrame(p_rows)
                st.caption(
                    f"Showing {len(p_df)} unique particles, deduplicated across all saved entries. "
                    ":orange[Occurrences] counts how many entries each particle appeared in."
                )
                render_custom_table(p_df)
                p_csv = p_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download all particles as CSV",
                    icon=":material/download:",
                    data=p_csv,
                    file_name="combined_particles.csv",
                    mime="text/csv",
                    key="download_combined_particles",
                    width="stretch",
                )
            else:
                st.caption("No particles found across saved entries.")
elif nav_page == "🖼️ Saved Entry Gallery":
    st.title("🖼️ Saved Entry Gallery")

    entries = get_all_study_entries()
    if not entries:
        st.info("No entries saved in the database yet. Process a crop on the 📖 Diacritizer & Analyzer page and save an entry.")
    else:
        entry_options = []
        entry_map = {}
        for row in entries:
            entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = row
            snippet = tashkeel[:35] + "..." if len(tashkeel) > 35 else tashkeel
            label = f"[#{entry_id}] {timestamp} | {fname} — {snippet}"
            entry_options.append(label)
            entry_map[label] = row

        if "gallery_multiselect" not in st.session_state:
            st.session_state.gallery_multiselect = entry_options

        selected_labels = st.sidebar.multiselect(
            "Select entries to display:",
            options=entry_options,
            key="gallery_multiselect",
        )

        if not selected_labels:
            st.info("Select one or more entries from the sidebar to view them here.")
        else:
            selected_rows = [entry_map[l] for l in selected_labels if l in entry_map]
            selected_rows.sort(key=lambda r: r[0])

            st.markdown(
                f"Displaying **{len(selected_rows)}** selected entries "
                f"(oldest → newest, by entry #):"
            )
            st.space("medium")

            for row in selected_rows:
                entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = row

                st.markdown(f"### Entry #{entry_id} — `{fname}` — *{timestamp}*")

                if img_b64:
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        saved_pil_img = Image.open(io.BytesIO(img_bytes))
                        st.image(saved_pil_img, caption=f"Crop from {fname}", width="stretch")
                    except Exception as e:
                        st.warning(f"Could not load image: {e}")
                else:
                    st.caption("No image data stored for this entry.")

                if tashkeel:
                    st.markdown("**Tashkeel text:**")
                    st.markdown(
                        f"""<div dir="rtl" class="arabic-text arabic-large" style="padding: 12px; border: 2px solid #4CAF50; border-radius: 8px; color: #2E7D32; background-color: rgba(76, 175, 80, 0.05); margin-bottom: 12px; text-align: right;">
                         {tashkeel}
                    </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No tashkeel text recorded.")

                if translation:
                    st.info(f"**English Translation:** {translation}")
                else:
                    st.caption("No translation recorded.")

                st.space("medium")
                st.divider()
else:
    st.info("Please upload one or more manga pages to begin.")
