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

# Database Connection Wrapper & Helper Functions
class LibsqlCursorWrapper:
    def __init__(self, client):
        self.client = client
        self.lastrowid = None
        self._rows = []

    def execute(self, sql, params=()):
        formatted_params = []
        for p in params:
            if isinstance(p, (dict, list)):
                formatted_params.append(json.dumps(p, ensure_ascii=False))
            else:
                formatted_params.append(p)

        res = self.client.execute(sql, formatted_params)
        self.lastrowid = getattr(res, 'last_insert_rowid', None)
        self._rows = [tuple(r) for r in res.rows] if res.rows else []
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class LibsqlConnWrapper:
    def __init__(self, client):
        self.client = client

    def cursor(self):
        return LibsqlCursorWrapper(self.client)

    def commit(self):
        pass

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


def get_db_connection():
    """Returns a connection to Turso cloud DB if secrets exist, else falls back to local SQLite."""
    turso_url = os.getenv("TURSO_DATABASE_URL")
    turso_token = os.getenv("TURSO_AUTH_TOKEN")

    if not turso_url or not turso_token:
        try:
            turso_url = turso_url or st.secrets.get("TURSO_DATABASE_URL")
            turso_token = turso_token or st.secrets.get("TURSO_AUTH_TOKEN")
        except Exception:
            pass

    if turso_url and turso_token:
        # 1. Try libsql_experimental (native driver)
        try:
            import libsql_experimental as libsql
            return libsql.connect(database=turso_url, auth_token=turso_token)
        except Exception:
            pass

        # 2. Try libsql_client (pure-Python HTTP driver - works everywhere without compilation)
        try:
            import libsql_client
            http_url = turso_url.replace("libsql://", "https://")
            client = libsql_client.create_client_sync(http_url, auth_token=turso_token)
            return LibsqlConnWrapper(client)
        except Exception as err:
            st.warning(f"Turso connection attempt failed ({err}). Falling back to local SQLite.")

    return sqlite3.connect("arabic_study_history.db", check_same_thread=False)

def init_db():
    """Initialize database for storing Arabic study logs."""
    conn = get_db_connection()
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
    """Insert a study entry into database and return new ID."""
    conn = get_db_connection()
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
    """Update deep_sarf_json for an existing study log entry in database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE study_logs
        SET deep_sarf_json = ?
        WHERE id = ?
    """, (json.dumps(deep_sarf, ensure_ascii=False), entry_id))
    conn.commit()
    conn.close()

def auto_save_or_update_current_entry(source_filename, cropped_img, diacritized_text, full_translation, verbs, nouns, particles, verb_results, last_verb, noun_results, last_noun):
    """Auto-save or update the current study entry in database whenever Sarf or Noun analysis is generated."""
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
        conn = get_db_connection()
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
    """Retrieve all study entries from database."""
    conn = get_db_connection()
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
    """Delete a study log entry by ID from database."""
    conn = get_db_connection()
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

def _safe_word(text):
    if text is None:
        return ""
    return str(text)

def build_word_meaning_map(verbs: list, nouns: list, particles: list) -> dict:
    """Build a lookup dictionary mapping diacritized & raw words to their grammar category, sub_type, derived status, base_verb, root, and meaning."""
    lookup = {}

    for v in verbs:
        if isinstance(v, dict):
            w = _safe_word(v.get("word", ""))
            m = v.get("meaning", "") or ""
            derived = v.get("derived", True)
            sub_type = v.get("sub_type", "Verb")
            base_verb = v.get("base_verb")
            root = v.get("root")
        elif hasattr(v, 'word'):
            w = _safe_word(getattr(v, 'word', ""))
            m = getattr(v, 'meaning', "") or ""
            derived = getattr(v, 'derived', True)
            sub_type = getattr(v, 'sub_type', "Verb")
            base_verb = getattr(v, 'base_verb', None)
            root = getattr(v, 'root', None)
        else:
            w, m = _safe_word(v), ""
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
            raw = strip_tashkeel(w)
            if raw:
                lookup[raw] = entry

    for n in nouns:
        if isinstance(n, dict):
            w = _safe_word(n.get("word", ""))
            m = n.get("meaning", "") or ""
            derived = n.get("derived", False)
            sub_type = n.get("sub_type", "Solid Noun")
            base_verb = n.get("base_verb")
            root = n.get("root")
        elif hasattr(n, 'word'):
            w = _safe_word(getattr(n, 'word', ""))
            m = getattr(n, 'meaning', "") or ""
            derived = getattr(n, 'derived', False)
            sub_type = getattr(n, 'sub_type', "Solid Noun")
            base_verb = getattr(n, 'base_verb', None)
            root = getattr(n, 'root', None)
        else:
            w, m = _safe_word(n), ""
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
            raw = strip_tashkeel(w)
            if raw:
                lookup[raw] = entry

    for p in particles:
        if isinstance(p, dict):
            w = _safe_word(p.get("word") or p.get("particle", ""))
            m = p.get("meaning", "") or ""
            p_type = p.get("type", "Particle (حَرْف)")
            effect = p.get("effect", "") or ""
        elif hasattr(p, 'word'):
            w = _safe_word(getattr(p, 'word', ""))
            m = getattr(p, 'meaning', "") or ""
            p_type = getattr(p, 'type', "Particle (حَرْف)")
            effect = getattr(p, 'effect', "") or ""
        else:
            w = _safe_word(p)
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
            raw = strip_tashkeel(w)
            if raw:
                lookup[raw] = entry

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
        safe_token = _escape_html_attr(token) + "&#8203;"
        safe_raw = _escape_html_attr(raw_token) + "&#8203;"
        vowel_spans.append(f'<span class="tashkeel-word" data-tooltip="{safe_tooltip}" data-vowel="{safe_token}" data-novowel="{safe_raw}" role="button" tabindex="0">{token}</span>')
        novowel_spans.append(f'<span class="tashkeel-word" data-tooltip="{safe_tooltip}" data-vowel="{safe_token}" data-novowel="{safe_raw}" role="button" tabindex="0">{raw_token}</span>')

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
    """Render a centered audio player right below the Tashkeel text box."""
    if not text or not text.strip():
        return

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    if "Microsoft Edge" in engine_choice:
        col_d1, col_play, col_d2 = st.columns([1, 2, 1])
        with col_play:
            if st.button("🔊 Play Audio (Edge Neural)", key=f"btn_tts_{key_suffix}", use_container_width=True):
                with st.spinner("Generating lifelike Edge Neural audio..."):
                    try:
                        audio_bytes = generate_edge_audio(text)
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"Edge TTS error: {e}")
    elif "Google Voice" in engine_choice:
        col_d1, col_play, col_d2 = st.columns([1, 2, 1])
        with col_play:
            if st.button("🔊 Play Audio (gTTS)", key=f"btn_tts_{key_suffix}", use_container_width=True):
                with st.spinner("Generating Google Voice audio..."):
                    try:
                        audio_bytes = generate_gtts_audio(text)
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.error(f"gTTS error: {e}")
    else:
        escaped_txt = text.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
        components.html(f"""
        <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <button onclick="speakText()" title="Play Audio" style="
                background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); 
                color: #ffffff; 
                border: none; 
                padding: 10px 24px; 
                border-radius: 20px; 
                font-size: 15px; 
                font-weight: 600;
                font-family: system-ui, -apple-system, sans-serif;
                cursor: pointer; 
                display: inline-flex; 
                align-items: center; 
                justify-content: center; 
                gap: 8px;
                box-shadow: 0 4px 12px rgba(21, 101, 192, 0.3);
                transition: transform 0.15s ease, box-shadow 0.15s ease;"
                onmouseover="this.style.transform='scale(1.03)'"
                onmouseout="this.style.transform='scale(1.0)'">
                🔊 Listen to Pronunciation
            </button>
        </div>
        <script>
        function speakText() {{
            const pWin = window.parent;
            if ('speechSynthesis' in pWin) {{
                pWin.speechSynthesis.cancel();
                const msg = new pWin.SpeechSynthesisUtterance('{escaped_txt}');
                msg.lang = 'ar-SA';
                msg.rate = 0.85;
                let voices = pWin.speechSynthesis.getVoices();
                let arVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('ar'));
                if (arVoice) msg.voice = arVoice;
                pWin.speechSynthesis.speak(msg);
            }}
        }}
        </script>
        """, height=50)

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

# 2. Inject global container max-width using st.markdown
st.markdown("""
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

        /*.tashkeeled {
            background-color: rgba(240,94,86, 0.15) !important;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 12px;
            font-weight: bold;
        }*/

        /* Dynamic Theme-Aware Tashkeeled Container */
        .tashkeeled {
            background-color: rgba(33, 150, 243, 0.12) !important;
            color: var(--text-color) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 24px 28px !important;
            margin-top: 18px !important;
            margin-bottom: 16px !important;
            font-size: 28px !important;
            line-height: 2.1 !important;
            font-weight: bold !important;
            box-shadow: none !important;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        @media (prefers-color-scheme: dark) {
            .tashkeeled {
                background-color: rgba(61, 157, 243, 0.15) !important;
            }
        }

        [data-theme="dark"] .tashkeeled,
        [theme="dark"] .tashkeeled,
        .stApp[data-theme="dark"] .tashkeeled {
            background-color: rgba(61, 157, 243, 0.15) !important;
        }


        /* Hover & Tap highlight for interactive tashkeel words */
        .tashkeel-word {
            position: relative !important;
            cursor: pointer;
            padding: 0 4px;
            display: inline-block;
            -webkit-tap-highlight-color: rgba(33, 150, 243, 0.3);
            user-select: none;
            -webkit-user-select: none;
        }
        .tashkeel-word:hover, .tashkeel-word:active, .tashkeel-word:focus {
            background-color: rgba(33, 150, 243, 0.2) !important;
            border-radius: 4px;
            outline: none;
        }

        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {
            max-width: 900px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* Mobile-only layout override: prevent Tashkeel title & toggle from stacking vertically */
        @media (max-width: 640px) {
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stToggle"]),
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stCheckbox"]) {
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                align-items: center !important;
            }
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stToggle"]) > div[data-testid="stColumn"]:first-child,
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stCheckbox"]) > div[data-testid="stColumn"]:first-child {
                flex: 3 1 0% !important;
                width: 65% !important;
                min-width: 0 !important;
            }
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stToggle"]) > div[data-testid="stColumn"]:last-child,
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stCheckbox"]) > div[data-testid="stColumn"]:last-child {
                flex: 1 1 0% !important;
                width: 35% !important;
                min-width: 0 !important;
            }
        }

        /* Flush right alignment for toggle switch */
        div[data-testid="stToggle"] {
            margin-left: auto !important;
            display: flex !important;
            justify-content: flex-end !important;
        }

        div[data-testid="stToggle"] > label {
            margin-left: auto !important;
            margin-right: 0 !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }



        /* 2. Expand the toggle container and force its contents flush right */
        div[data-testid="stToggle"],
        div[data-testid="stCheckbox"],
        div[data-testid="stElementContainer"]:has(div[data-testid="stToggle"]),
        div[data-testid="stElementContainer"]:has(div[data-testid="stCheckbox"]) {
            width: 100% !important;
            display: flex !important;
            justify-content: flex-end !important;
            margin-left: auto !important;
        }

        /* 3. Push the label wrapper flush against the right margin */
        div[data-testid="stToggle"] > label,
        div[data-testid="stCheckbox"] > label,
        label[data-baseweb="checkbox"] {
            margin-left: auto !important;
            margin-right: 0 !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }

    </style>
    """, unsafe_allow_html=True)

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

def extract_root_clean(word: str, info: dict | None = None) -> tuple[str, str]:
    """
    Extract clean root string (e.g. 'بسم') and display string (e.g. 'ب - س - م').
    Prefers Gemini's returned root field when available.
    """
    if info and isinstance(info, dict):
        g_root = info.get("root")
        if g_root and g_root != "N/A":
            clean = g_root.replace("-", "").strip()
            display = " - ".join(g_root.replace("-", " ").split())
            return clean, display

    raw = strip_tashkeel(word) if word else ""
    clean = re.sub(r'[^\u0621-\u064A]', '', raw)
    display = " - ".join(list(clean)) if clean else "N/A"
    return clean, display

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
if 'cropper_collapsed' not in st.session_state:
    st.session_state.cropper_collapsed = False
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None
if 'verbs' not in st.session_state:
    st.session_state.verbs = []
if 'nouns' not in st.session_state:
    st.session_state.nouns = []
if 'particles' not in st.session_state:
    st.session_state.particles = []
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "📁 Upload Image"
if 'prev_input_mode' not in st.session_state:
    st.session_state.prev_input_mode = "📁 Upload Image"

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
    <div dir="rtl" class="arabic-text" style="font-size: 24px !important; font-weight: normal !important; padding: 24px 28px !important; border: 2px solid #2196F3 !important; background-color: rgba(33, 150, 243, 0.18) !important; border-radius: 10px !important; text-align: right !important; line-height: 2.0 !important;">
        {diacritized_text}
    </div>
    """, unsafe_allow_html=True)

# Sidebar Navigation Menu
st.sidebar.title("📌 Navigation")
nav_page = st.sidebar.radio(
    "Go to page:",
    [
        "📖 Diacritizer & Analyzer",
        "📚 Saved Entry Inspector",
        "📊 History & Anki Export",
        "🔍 Combined Vocabulary",
        "🖼️ Saved Entry Gallery"
    ],
    key="nav_page_selection"
)

st.sidebar.markdown("---")

input_mode = st.sidebar.radio(
    "📥 Input Method",
    ["📁 Upload Image", "✏️ Input Text"],
    index=0 if st.session_state.input_mode == "📁 Upload Image" else 1,
    key="input_mode_selector"
)
st.session_state.input_mode = input_mode

if input_mode != st.session_state.prev_input_mode:
    st.session_state.prev_input_mode = input_mode
    st.session_state.current_file = ""
    st.session_state.diacritized_text = ""
    st.session_state.extracted_text = ""
    st.session_state.processed_by = ""
    st.session_state.verbs = []
    st.session_state.nouns = []
    st.session_state.particles = []
    st.session_state.cropped_img = None
    st.session_state.cropper_collapsed = False
    st.session_state.sarf_results = None
    st.session_state.sarf_word = ""
    st.session_state.verb_results = None
    st.session_state.last_analyzed_verb = ""
    st.session_state.noun_results = None
    st.session_state.last_analyzed_noun = ""
    st.rerun()

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

    selected_file_name = "Direct Text Input"
    cropped_img = None
    process_clicked = False
    text_process_clicked = False

    if input_mode == "📁 Upload Image":
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

            # ── Step 1: Crop (collapsible after processing) ──────────────────────
            # Toggle collapse on header click
            step1_label = "✅ Step 1: Crop Selection (click to expand)" if st.session_state.cropper_collapsed else "📌 Step 1: Select the Area to Crop"
            if st.button(step1_label, key="toggle_step1", use_container_width=True):
                st.session_state.cropper_collapsed = not st.session_state.cropper_collapsed
                st.rerun()

            cropped_img = None  # default

            if not st.session_state.cropper_collapsed:
                st.caption("Draw a rectangle over the speech bubble you want to extract text from. The image will scale to fit your screen — your crop will be mapped to the full-resolution image for best quality.")

                # Cropper key (remount when file/page changes)
                cropper_key = f"cropper_{selected_file_name}"
                if is_pdf:
                    cropper_key += f"_p{st.session_state.pdf_page}"

                # ── Pre-scale image to Streamlit column width ──────────────────────
                # The Fabric.js canvas inside the iframe is sized from the image we pass in,
                # NOT from CSS. So we control the canvas width by scaling the image ourselves.
                DISPLAY_WIDTH = 700  # matches Streamlit's default column width
                display_img = fit_image_to_max_width(img, max_width=DISPLAY_WIDTH)
                # Also scale UP small images so they fill the full column
                if display_img.width < DISPLAY_WIDTH:
                    scale_up = DISPLAY_WIDTH / display_img.width
                    display_img = display_img.resize(
                        (DISPLAY_WIDTH, int(display_img.height * scale_up)),
                        Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                    )



                # Cropper operates on the display image; canvas = display_img size
                box = st_cropper(
                    display_img,
                    realtime_update=True,
                    box_color='#FF4444',
                    aspect_ratio=None,
                    return_type='box',
                    should_resize_image=False,
                    key=cropper_key
                )

                # Inject JS (via a height=0 helper iframe) that reaches INTO the cropper iframe
                # and centers the .canvas-container — CSS from the parent page can't do this.
                st.components.v1.html("""
                <script>
                (function() {
                    function centerCropperCanvas() {
                        var parentDoc = window.parent.document;
                        var iframes = parentDoc.querySelectorAll('iframe');
                        var found = false;
                        iframes.forEach(function(iframe) {
                            try {
                                var iDoc = iframe.contentDocument || iframe.contentWindow.document;
                                if (!iDoc) return;
                                var container = iDoc.querySelector('.canvas-container');
                                if (!container) return;
                                found = true;
                                // Center the canvas wrapper inside the iframe
                                container.style.marginLeft  = 'auto';
                                container.style.marginRight = 'auto';
                                container.style.display     = 'block';
                                // Flex-center the iframe body so margin: auto works
                                var body = iDoc.body;
                                if (body) {
                                    body.style.margin          = '0';
                                    body.style.padding         = '0';
                                    body.style.display         = 'flex';
                                    body.style.flexDirection   = 'column';
                                    body.style.alignItems      = 'center';
                                    body.style.justifyContent  = 'flex-start';
                                    body.style.overflow        = 'hidden';
                                    body.style.width           = '100%';
                                }
                            } catch(e) { /* cross-origin, skip */ }
                        });
                        return found;
                    }
                    // Retry because the cropper iframe loads asynchronously
                    [100, 400, 900, 2000].forEach(function(delay) {
                        setTimeout(centerCropperCanvas, delay);
                    });
                })();
                </script>
                """, height=0)

                # Map box coordinates back to the original full-resolution image
                scale_ratio = img.width / display_img.width  # e.g. 2× if original was 1400px
                if box is not None:
                    left   = max(0, int(box.get('left',   0) * scale_ratio))
                    top    = max(0, int(box.get('top',    0) * scale_ratio))
                    width  = max(1, int(box.get('width',  display_img.width)  * scale_ratio))
                    height = max(1, int(box.get('height', display_img.height) * scale_ratio))
                    right  = min(img.width,  left + width)
                    bottom = min(img.height, top  + height)
                    cropped_img = img.crop((left, top, right, bottom))
                    st.session_state.cropped_img = cropped_img  # persist across reruns

                # Centered "Process & Add Vowels" button below the cropper
                st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
                _, btn_col, _ = st.columns([1, 2, 1])

                with btn_col:
                    process_clicked = st.button("⚡ Process & Add Vowels", type="primary", use_container_width=True)
                st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

            else:
                # Collapsed — still need a placeholder so the button logic below works
                process_clicked = False

    # ── Step 2: Results ──────────────────────────────────────────────────
    st.markdown("---")
    if cropped_img or st.session_state.diacritized_text or text_process_clicked:
        # User feedback badge
        if st.session_state.processed_by:
            st.info(f"⚡ **Last processed by:** {st.session_state.processed_by}")
        elif cropped_img and not st.session_state.cropper_collapsed:
            st.caption(f"Ready to process using: **{ocr_mode}**")
        if process_clicked and cropped_img:
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
                                # Persist cropped image & collapse step 1
                                st.session_state.cropped_img = cropped_img
                                st.session_state.cropper_collapsed = True
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
                                # Persist cropped image & collapse step 1
                                st.session_state.cropped_img = cropped_img
                                st.session_state.cropper_collapsed = True
                                st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
        # For results, use the live crop if available; else fall back to the last saved one
        display_cropped_img = cropped_img or st.session_state.cropped_img
        # Render results if we have diacritized text in session state
        if st.session_state.diacritized_text:
            extracted_text = st.session_state.extracted_text
            diacritized_text = st.session_state.diacritized_text
            # Top Toolbar: Centered Action Buttons
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if st.button("📄 Raw Text", key="btn_open_raw_modal", icon=":material/description:", use_container_width=True):
                    show_raw_text_modal()
            with col_m2:
                if st.button("✨ Diacritized Text", key="btn_open_diacritized_modal", icon=":material/edit_note:", use_container_width=True):
                    show_diacritized_text_modal()
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
            # ── Captured image above results ──
            if display_cropped_img:
                import base64 as _b64
                from io import BytesIO as _BytesIO
                _buf = _BytesIO()
                display_cropped_img.save(_buf, format="PNG")
                _img_str = _b64.b64encode(_buf.getvalue()).decode()
                st.markdown(f"""
                <div style="padding:25px;border:1px solid rgba(128,128,128,0.25);border-radius:10px;margin-bottom:1rem;">
                    <img src="data:image/png;base64,{_img_str}" style="width:100%;height:auto;display:block;border-radius:6px;">
                    <div style="text-align:center;font-size:0.82rem;margin-top:10px;opacity:0.6;">✨ Captured Image</div>
                </div>
                """, unsafe_allow_html=True)
            with st.container(border=True):
                col_thdr, col_ttog = st.columns([2, 1], vertical_alignment="center")
                with col_thdr:
                    st.markdown("#### ✨ Tashkeel Text")
                with col_ttog:
                    show_vowels = st.toggle("Vowels", value=True, key="main_vowel_toggle")
                v_html, nv_html, sentence_tokens = build_interactive_tashkeel_html(
                    diacritized_text,
                    json.dumps(st.session_state.verbs),
                    json.dumps(st.session_state.nouns),
                    json.dumps(st.session_state.particles),
                )
                st.markdown(f'<div dir="rtl" class="arabic-text tashkeeled" id="tashkeel-container-main">{v_html}</div>', unsafe_allow_html=True)
                st.components.v1.html(f"""
                <script>
                (function() {{
                    var SHOW_VOWELS = {str(show_vowels).lower()};
                    var container = window.parent.document.getElementById('tashkeel-container-main');
                    if (!container) return;
                    var spans = container.querySelectorAll('.tashkeel-word');
                    for (var i = 0; i < spans.length; i++) {{
                        var span = spans[i];
                        span.textContent = span.getAttribute('data-vowel') || span.textContent;
                        span.style.minWidth = span.getBoundingClientRect().width + 'px';
                        if (!SHOW_VOWELS) {{
                            span.textContent = span.getAttribute('data-novowel') || span.textContent;
                        }}
                    }}
                }})();
                </script>
                """, height=0)
                render_arabic_tts(diacritized_text, tts_engine_choice, key_suffix="full_sentence")
                st.components.v1.html("""
                <script>
                (function() {
                    try {
                        var pDoc = window.parent.document;
                        var pWin = window.parent;
                        var existingTip = pDoc.getElementById('global-tashkeel-tooltip');
                        if (existingTip) existingTip.remove();
                        var tip = pDoc.createElement('div');
                        tip.id = 'global-tashkeel-tooltip';
                        tip.style.cssText = 'position: absolute; background-color: #0f172a; color: #ffffff; padding: 8px 14px; border-radius: 8px; font-size: 13px; font-family: system-ui, -apple-system, sans-serif; font-weight: 500; z-index: 999999; pointer-events: none; white-space: normal; max-width: 280px; word-wrap: break-word; display: none; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.2); text-align: center; line-height: 1.4; transition: opacity 0.15s ease; opacity: 0;';
                        pDoc.body.appendChild(tip);
                        function showTip(span) {
                            var tipText = span.getAttribute('data-tooltip') || span.getAttribute('title');
                            if (!tipText) return;
                            tip.textContent = tipText;
                            tip.style.display = 'block';
                            tip.style.opacity = '1';
                            var rect = span.getBoundingClientRect();
                            var scrollTop = pWin.pageYOffset || pDoc.documentElement.scrollTop;
                            var scrollLeft = pWin.pageXOffset || pDoc.documentElement.scrollLeft;
                            var tipW = tip.offsetWidth;
                            var tipH = tip.offsetHeight;
                            var screenW = pDoc.documentElement.clientWidth;
                            var top = rect.top + scrollTop - tipH - 8;
                            if (top < scrollTop) {
                                top = rect.bottom + scrollTop + 8;
                            }
                            var left = rect.left + scrollLeft + (rect.width / 2) - (tipW / 2);
                            left = Math.max(12, Math.min(left, screenW - tipW - 12));
                            tip.style.top = top + 'px';
                            tip.style.left = left + 'px';
                            pWin.clearTimeout(pWin._tashkeelTipTimer);
                            pWin._tashkeelTipTimer = pWin.setTimeout(function() {
                                tip.style.opacity = '0';
                                pWin.setTimeout(function() { if (tip.style.opacity === '0') tip.style.display = 'none'; }, 150);
                            }, 3500);
                        }
                        function hideTip() {
                            if (tip) {
                                tip.style.opacity = '0';
                                pWin.setTimeout(function() { if (tip.style.opacity === '0') tip.style.display = 'none'; }, 150);
                            }
                        }
                        if (pWin._tashkeelHandler) {
                            pDoc.removeEventListener('click', pWin._tashkeelHandler, true);
                            pDoc.removeEventListener('touchend', pWin._tashkeelHandler, true);
                        }
                        pWin._tashkeelHandler = function(e) {
                            var span = e.target.closest('.tashkeel-word');
                            if (span) {
                                showTip(span);
                            } else {
                                hideTip();
                            }
                        };
                        pDoc.addEventListener('click', pWin._tashkeelHandler, true);
                        pDoc.addEventListener('touchend', pWin._tashkeelHandler, true);
                    } catch(err) {
                        console.error(err);
                    }
                })();
                </script>
                """, height=0)
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
                clean_root, root_display = extract_root_clean(selected_word, info)
                # Render the card container
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
                                                selected_file_name, display_cropped_img, diacritized_text,
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
                        sel_v_info = (word_map.get(verb_to_analyze) or word_map.get(strip_tashkeel(verb_to_analyze))) if 'word_map' in locals() else None
                        clean_verb_root, verb_root_display = extract_root_clean(verb_to_analyze, sel_v_info)
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
                                        selected_file_name, display_cropped_img, diacritized_text,
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
                        sel_noun_obj = word_map.get(noun_to_analyze) or word_map.get(strip_tashkeel(noun_to_analyze)) or {}
                        clean_noun_root, noun_root_display = extract_root_clean(noun_to_analyze, sel_noun_obj)
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
                                                selected_file_name, display_cropped_img, diacritized_text,
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
                                                selected_file_name, display_cropped_img, diacritized_text,
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
                                            selected_file_name, display_cropped_img, diacritized_text,
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
                    save_img = display_cropped_img
                    if diacritized_text:
                        img_b64 = image_to_base64(save_img) if save_img else ""
                        deep_sarf = {
                            "last_verb": st.session_state.last_analyzed_verb,
                            "verb_sarf": st.session_state.verb_results,
                            "last_noun": st.session_state.last_analyzed_noun,
                            "noun_sarf": st.session_state.noun_results
                        }
                        source_name = selected_file_name if save_img else "Direct Text Input"
                        save_study_entry(
                            source_filename=source_name,
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
                        st.warning("Please process text first before saving to database.")
        else:
            st.info("Please upload one or more manga pages to begin.")

    elif input_mode == "✏️ Input Text":
        st.markdown("### ✏️ Direct Text Input")
        st.caption("Paste or type Arabic text directly. It can already have vowels (Tashkeel) or be plain text — we'll process it the same way.")
        text_input_val = st.text_area("Arabic Text", height=150, placeholder="اكتب أو الصق النص العربي هنا...")
        text_process_clicked = st.button("⚡ Process & Add Vowels", type="primary", use_container_width=True)
        
        if text_process_clicked:
            if not text_input_val.strip():
                st.warning("Please enter some Arabic text first.")
            else:
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text_input_val.strip()) if p.strip()]
                if len(paragraphs) <= 1:
                    paragraphs = [p.strip() for p in text_input_val.strip().splitlines() if p.strip()]
                para_count = len(paragraphs)
                with st.spinner(f"Processing {para_count} paragraph{'s' if para_count != 1 else ''} and applying Tashkeel..."):
                    try:
                        gemini_res = gemini_tashkeel(text_input_val.strip())
                        diacritized_text = gemini_res.get("tashkeel_text", text_input_val.strip())
                        full_trans = gemini_res.get("full_translation", "")
                        verbs_list = gemini_res.get("verbs", [])
                        nouns_list = gemini_res.get("nouns", [])
                        particles_list = gemini_res.get("particles", [])
                        
                        st.session_state.extracted_text = text_input_val.strip()
                        st.session_state.diacritized_text = diacritized_text
                        st.session_state.full_translation = full_trans
                        st.session_state.processed_by = "Direct Text Input"
                        st.session_state.verbs = verbs_list
                        st.session_state.nouns = nouns_list
                        st.session_state.particles = particles_list
                        st.session_state.sarf_results = None
                        st.session_state.sarf_word = ""
                        st.session_state.verb_results = None
                        st.session_state.last_analyzed_verb = ""
                        st.session_state.noun_results = None
                        st.session_state.last_analyzed_noun = ""
                        st.session_state.cropped_img = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
elif nav_page == "📚 Saved Entry Inspector":
    st.title("📚 Saved Entry Inspector")
    
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

        # Full Entry Inspector View
        selected_row = entry_map[selected_option]
        entry_id, timestamp, fname, img_b64, tashkeel, translation, verbs_str, nouns_str, particles_str, deep_sarf_str = selected_row

        col_hdr, col_del = st.columns([4, 1])
        with col_hdr:
            st.markdown(f"### Inspector View: Entry #{entry_id}")
        with col_del:
            if st.button("🗑️ Delete Entry", key=f"btn_del_{entry_id}", use_container_width=True):
                delete_study_entry(entry_id)
                st.success(f"Deleted entry #{entry_id}!")
                st.rerun()

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

        col_title, col_toggle, col_copy = st.columns([2, 1, 2], vertical_alignment="center")

        with col_title:
            st.markdown("#### ✨ Tashkeel Text")

        with col_toggle:
            show_saved_vowels = st.toggle("Vowels", value=True, key=f"hist_vowel_toggle_{entry_id}")

        with col_copy:
            copy_with_vowels = tashkeel.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").replace("\n", " ").replace("\r", "")
            copy_without_vowels = strip_tashkeel(tashkeel).replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").replace("\n", " ").replace("\r", "")
            components.html(f"""
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <button id="btn-copy-with-{entry_id}" title="Copy Arabic text with vowels" style="
                    background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
                    color: #ffffff;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: 600;
                    font-family: system-ui, -apple-system, sans-serif;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 4px;
                    box-shadow: 0 2px 6px rgba(46,125,50,0.25);
                    transition: transform 0.15s ease;
                " onmouseover="this.style.transform='scale(1.03)'" onmouseout="this.style.transform='scale(1.0)'">
                    📋 Copy with Vowels
                </button>
                <button id="btn-copy-without-{entry_id}" title="Copy Arabic text without vowels" style="
                    background: linear-gradient(135deg, #FB8C00 0%, #E65100 100%);
                    color: #ffffff;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: 600;
                    font-family: system-ui, -apple-system, sans-serif;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 4px;
                    box-shadow: 0 2px 6px rgba(230,81,0,0.25);
                    transition: transform 0.15s ease;
                " onmouseover="this.style.transform='scale(1.03)'" onmouseout="this.style.transform='scale(1.0)'">
                    📋 Copy without Vowels
                </button>
            </div>
            <script>
            const COPY_TEXT_WITH_{entry_id} = "{copy_with_vowels}";
            const COPY_TEXT_WITHOUT_{entry_id} = "{copy_without_vowels}";
            (function() {{
                var btnWith = document.getElementById('btn-copy-with-{entry_id}');
                var btnWithout = document.getElementById('btn-copy-without-{entry_id}');
                if (btnWith) {{
                    btnWith.addEventListener('click', function() {{ copyText(COPY_TEXT_WITH_{entry_id}); }});
                }}
                if (btnWithout) {{
                    btnWithout.addEventListener('click', function() {{ copyText(COPY_TEXT_WITHOUT_{entry_id}); }});
                }}
            }})();
            function copyText(text) {{
                var pWin = window.parent;
                var pDoc = window.parent.document;
                if (pWin.navigator && pWin.navigator.clipboard) {{
                    pWin.navigator.clipboard.writeText(text).then(function() {{
                        showToast('Copied!');
                    }}).catch(function() {{
                        fallbackCopy(text);
                    }});
                }} else {{
                    fallbackCopy(text);
                }}
            }}
            function fallbackCopy(text) {{
                var textarea = pDoc.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                pDoc.body.appendChild(textarea);
                textarea.select();
                try {{ pDoc.execCommand('copy'); }} catch(e) {{}}
                pDoc.body.removeChild(textarea);
                showToast('Copied!');
            }}
            function showToast(msg) {{
                var pWin = window.parent;
                var pDoc = window.parent.document;
                var toast = pDoc.getElementById('copy-toast');
                if (!toast) {{
                    toast = pDoc.createElement('div');
                    toast.id = 'copy-toast';
                    toast.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1E88E5;color:#fff;padding:8px 20px;border-radius:20px;font-size:14px;font-weight:600;font-family:system-ui,-apple-system,sans-serif;z-index:999999;box-shadow:0 4px 12px rgba(0,0,0,0.3);transition:opacity 0.3s ease;';
                    pDoc.body.appendChild(toast);
                }}
                toast.textContent = msg;
                toast.style.opacity = '1';
                pWin.clearTimeout(pWin._copyToastTimer);
                pWin._copyToastTimer = pWin.setTimeout(function() {{
                    toast.style.opacity = '0';
                }}, 1500);
            }}
            </script>
            """, height=40)

        v_html, nv_html, _ = build_interactive_tashkeel_html(
            tashkeel,
            verbs_str or "",
            nouns_str or "",
            particles_str or "",
        )

        st.markdown(f'<div dir="rtl" class="arabic-text tashkeeled" id="tashkeel-container-{entry_id}">{v_html}</div>', unsafe_allow_html=True)

        st.components.v1.html(f"""
        <script>
        (function() {{
            var SHOW_VOWELS = {str(show_saved_vowels).lower()};
            var container = window.parent.document.getElementById('tashkeel-container-{entry_id}');
            if (!container) return;
            var spans = container.querySelectorAll('.tashkeel-word');
            for (var i = 0; i < spans.length; i++) {{
                var span = spans[i];
                span.textContent = span.getAttribute('data-vowel') || span.textContent;
                span.style.minWidth = span.getBoundingClientRect().width + 'px';
                if (!SHOW_VOWELS) {{
                    span.textContent = span.getAttribute('data-novowel') || span.textContent;
                }}
            }}
        }})();
        </script>
        """, height=0)
        render_arabic_tts(tashkeel, tts_engine_choice, key_suffix=f"hist_{entry_id}")

        st.components.v1.html("""
        <script>
        (function() {
            try {
                var pDoc = window.parent.document;
                var pWin = window.parent;

                var existingTip = pDoc.getElementById('global-tashkeel-tooltip');
                if (existingTip) existingTip.remove();

                var tip = pDoc.createElement('div');
                tip.id = 'global-tashkeel-tooltip';
                tip.style.cssText = 'position: absolute; background-color: #0f172a; color: #ffffff; padding: 8px 14px; border-radius: 8px; font-size: 13px; font-family: system-ui, -apple-system, sans-serif; font-weight: 500; z-index: 999999; pointer-events: none; white-space: normal; max-width: 280px; word-wrap: break-word; display: none; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.2); text-align: center; line-height: 1.4; transition: opacity 0.15s ease; opacity: 0;';
                pDoc.body.appendChild(tip);

                function showTip(span) {
                    var tipText = span.getAttribute('data-tooltip') || span.getAttribute('title');
                    if (!tipText) return;

                    tip.textContent = tipText;
                    tip.style.display = 'block';
                    tip.style.opacity = '1';

                    var rect = span.getBoundingClientRect();
                    var scrollTop = pWin.pageYOffset || pDoc.documentElement.scrollTop;
                    var scrollLeft = pWin.pageXOffset || pDoc.documentElement.scrollLeft;

                    var tipW = tip.offsetWidth;
                    var tipH = tip.offsetHeight;
                    var screenW = pDoc.documentElement.clientWidth;

                    var top = rect.top + scrollTop - tipH - 8;
                    if (top < scrollTop) {
                        top = rect.bottom + scrollTop + 8;
                    }

                    var left = rect.left + scrollLeft + (rect.width / 2) - (tipW / 2);
                    left = Math.max(12, Math.min(left, screenW - tipW - 12));

                    tip.style.top = top + 'px';
                    tip.style.left = left + 'px';

                    pWin.clearTimeout(pWin._tashkeelTipTimer);
                    pWin._tashkeelTipTimer = pWin.setTimeout(function() {
                        tip.style.opacity = '0';
                        pWin.setTimeout(function() { if (tip.style.opacity === '0') tip.style.display = 'none'; }, 150);
                    }, 3500);
                }

                function hideTip() {
                    if (tip) {
                        tip.style.opacity = '0';
                        pWin.setTimeout(function() { if (tip.style.opacity === '0') tip.style.display = 'none'; }, 150);
                    }
                }

                if (pWin._tashkeelHandler) {
                    pDoc.removeEventListener('click', pWin._tashkeelHandler, true);
                    pDoc.removeEventListener('touchend', pWin._tashkeelHandler, true);
                }

                pWin._tashkeelHandler = function(e) {
                    var span = e.target.closest('.tashkeel-word');
                    if (span) {
                        showTip(span);
                    } else {
                        hideTip();
                    }
                };

                pDoc.addEventListener('click', pWin._tashkeelHandler, true);
                pDoc.addEventListener('touchend', pWin._tashkeelHandler, true);
            } catch(err) {
                console.error(err);
            }
        })();
        </script>
        """, height=0)


        
        click_word = components.html(f"""
        <script>
        (function() {{
            const parentDoc = window.parent.document;
            const container = parentDoc.getElementById('tashkeel-container-{entry_id}');
            if (!container) return;

            container.addEventListener('click', function(e) {{
                const span = e.target.closest('.tashkeel-word');
                if (!span) return;
                e.preventDefault();
                e.stopPropagation();
                const text = span.textContent || span.innerText;
                const tipText = span.getAttribute('title') || '';
                
                let tip = parentDoc.getElementById('custom-tashkeel-tooltip-{entry_id}');
                if (!tip) {{
                    tip = parentDoc.createElement('div');
                    tip.id = 'custom-tashkeel-tooltip-{entry_id}';
                    tip.style.cssText = 'position:absolute;background:#333;color:white;padding:6px 10px;border-radius:4px;font-size:13px;z-index:9999;pointer-events:none;white-space:normal;max-width:250px;word-wrap:break-word;display:none;';
                    parentDoc.body.appendChild(tip);
                }}
                tip.textContent = tipText;
                tip.style.display = 'block';
                const rect = span.getBoundingClientRect();
                tip.style.left = (rect.left + window.parent.scrollX) + 'px';
                tip.style.top = (rect.top + window.parent.scrollY - tip.offsetHeight - 8) + 'px';
                setTimeout(() => {{ tip.style.display = 'none'; }}, 3000);
                
                if (window.parent.Streamlit) {{
                    window.parent.Streamlit.setComponentValue(text);
                }}
            }});
        }})();
        </script>
        """, height=0)
        
        if translation:
            st.info(f"💡 **English Translation:** {translation}")
        else:
            st.caption("No translation recorded.")

        # Interactive Word Lookup Reader for Saved Entry
        st.markdown("#### 👆 Interactive Word Lookup (Click Word to Inspect)")
        
        pills_key = f"hist_word_pills_{entry_id}"
        counter_key = f"hist_word_click_counter_{entry_id}"
        
        if click_word and isinstance(click_word, str) and click_word.strip():
            new_word = click_word.strip()
            new_counter = st.session_state.get(counter_key, 0) + 1
            st.session_state[counter_key] = new_counter
            st.session_state[pills_key] = new_word
            st.rerun()

        selected_hist_word = st.pills(
            "Select a word from this saved entry:",
            options=saved_tokens,
            selection_mode="single",
            key=pills_key
        )

        if selected_hist_word:
            raw_sel = strip_tashkeel(selected_hist_word)
            h_info = saved_word_map.get(selected_hist_word) or saved_word_map.get(raw_sel)
            clean_root, root_display = extract_root_clean(selected_hist_word, h_info)

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
                    h_v_info = saved_word_map.get(hv_to_analyze) if 'saved_word_map' in locals() else None
                    h_clean_v_root, h_v_root_disp = extract_root_clean(hv_to_analyze, h_v_info)
                    
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
                    h_clean_n_root, h_n_root_disp = extract_root_clean(hn_to_analyze, hn_obj if isinstance(hn_obj, dict) else None)
                    
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
    else:
        st.info("No entries saved in the database yet. Process a crop on the Diacritizer page and click '💾 Save Entry to Database'!")

elif nav_page == "📊 History & Anki Export":
    st.title("📊 History Table & Anki Export")

    entries = get_all_study_entries()
    if not entries:
        st.info("No entries saved in the database yet. Process a crop on the 📖 Diacritizer & Analyzer page and save an entry.")
    else:
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

        col_title, col_anki = st.columns([3, 1], vertical_alignment="center")
        with col_title:
            st.caption(f"Master overview of all **{len(entries)}** saved study entries.")
        with col_anki:
            df_anki = pd.DataFrame(anki_rows)
            csv_buffer = df_anki.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Anki CSV",
                data=csv_buffer,
                file_name="arabic_manga_anki_deck.csv",
                mime="text/csv",
                key="download_anki_csv",
                use_container_width=True
            )

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
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
