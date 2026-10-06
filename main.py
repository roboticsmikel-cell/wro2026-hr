
import os
import re
import requests
# Windows printing: render to the printer DC so Baybayin sheets spool silently
# (no Photos dialog). Guarded so the app still runs on a machine without them.
try:
    import win32print
    import win32ui
    from win32con import HORZRES, VERTRES
    from PIL import Image, ImageWin
    _PRINTING_AVAILABLE = True
except Exception as _print_import_err:      # pragma: no cover - non-Windows
    _PRINTING_AVAILABLE = False
    print("Printing unavailable:", _print_import_err)
import json
import hashlib
import threading
import uuid
from google import genai
from google.genai import types
from google.cloud import texttospeech
import speech_recognition as sr
from baybayin_text import split_syllables

from langdetect import detect

import subprocess
import traceback


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn

import numpy as np

import serial
import time

# face recognition imports
import cv2
import time
import mediapipe as mp

from ffpyplayer.player import MediaPlayer
from display_utils import display_frame

import io as _io
import wave as _wave
from fastapi.staticfiles import StaticFiles as _StaticFiles

BASE = os.path.dirname(os.path.abspath(__file__))
TTS_OUT = os.path.join(BASE, "static", "tts")
GEN_OUT = os.path.join(BASE, "static", "baybayin")
os.makedirs(TTS_OUT, exist_ok=True)
os.makedirs(GEN_OUT, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/gen", _StaticFiles(directory=GEN_OUT), name="gen")
app.mount("/media", _StaticFiles(directory=os.path.join(BASE, "source")), name="media")

# Subtitle state exposed via Flask for frontend subtitles
last_subtitle = ""
last_user_input = ""
subtitle_history = []

# Rolling short-term memory so ALZONA can follow a continuous conversation
# (e.g. "tell me more", "and her?"). Kept short so an earlier turn's language
# can't bias the reply language of the current message.
conversation_history = []      # [(user_text, alzona_reply), ...]
MAX_HISTORY_TURNS = 4

# GOOGLE CLOUD CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dyciroboticsteam-82e1fa8b4c0c.json"

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# GEMINI API KEYS — read from .env. A second key (…_BACKUP) lets ALZONA fail
# over automatically when the primary key's project is billing-blocked
# (403 "dunning") or rate-limited (429), instead of going silent with
# "I'm having trouble answering right now."
_gemini_keys = [k for k in (
    os.getenv("GOOGLE_GENAI_API_KEY", "").strip(),
    os.getenv("GOOGLE_GENAI_API_KEY_BACKUP", "").strip(),
) if k]
# No key -> FILE-BASED mode: questions are answered straight from the knowledge
# library (retrieve_knowledge), and everything that needs Gemini — speech
# transcription, coin reading, the Leda voice — reports itself unavailable
# instead of stopping the whole backend.
GEMINI_ENABLED = bool(_gemini_keys)
if not GEMINI_ENABLED:
    print("No GOOGLE_GENAI_API_KEY in .env — running FILE-BASED: answers come "
          "from the knowledge files only; Gemini features are off.")

_gemini_clients = [genai.Client(api_key=k) for k in _gemini_keys]
_active_idx = 0
# currently-active client (used across the app); None in file-based mode
client = _gemini_clients[0] if _gemini_clients else None


def _key_blocked_error(msg):
    """True for errors that mean the current key/project is blocked and another
    key is worth trying: billing/dunning denials (403) or rate limits (429)."""
    m = msg.lower()
    return any(s in m for s in
               ("permission_denied", "dunning", "resource_exhausted",
                "quota", " 403", "429"))


def _overloaded_error(msg):
    """True when Gemini itself is busy — transient, and NOT a key problem."""
    m = msg.lower()
    return any(s in m for s in ("unavailable", " 503", "high demand", "overloaded"))


# Short waits: long enough for a demand spike to pass, short enough that a
# visitor standing in front of the robot does not think it has frozen.
_OVERLOAD_BACKOFF = (1.5, 3.0)


def gen_content(**kwargs):
    """client.models.generate_content with automatic key failover. If the active
    key is billing-blocked or rate-limited, transparently retry on the next key
    and promote it to active so later calls stay fast. Any non-key error (or a
    key error with no working alternative) is raised as before."""
    global _active_idx, client
    n = len(_gemini_clients)
    if not n:
        # Every caller already catches exceptions and falls back.
        raise RuntimeError("Gemini is disabled (file-based mode)")
    last_err = None
    for step in range(n):
        idx = (_active_idx + step) % n
        try:
            r = _gemini_clients[idx].models.generate_content(**kwargs)
            if idx != _active_idx:
                _active_idx = idx
                client = _gemini_clients[idx]
                print(f"Gemini: switched to key #{idx + 1} (previous key blocked).")
            return r
        except Exception as e:
            last_err = e
            if step < n - 1 and _key_blocked_error(str(e)):
                print(f"Gemini key #{idx + 1} blocked ({str(e)[:70]}); trying next key…")
                continue
            # Server-side overload is temporary and has nothing to do with the
            # key, so switching keys would not help — waiting briefly does.
            # Without this a passing demand spike surfaces to the visitor as
            # "I'm having trouble answering right now" mid-demonstration.
            if _overloaded_error(str(e)):
                for wait in _OVERLOAD_BACKOFF:
                    print(f"Gemini busy; retrying in {wait}s…")
                    time.sleep(wait)
                    try:
                        return _gemini_clients[idx].models.generate_content(**kwargs)
                    except Exception as retry_err:
                        last_err = retry_err
                        if not _overloaded_error(str(retry_err)):
                            raise
            raise
    raise last_err   # pragma: no cover — loop always returns or raises

# DEFAULT SETTINGS  
language_code = "en-US"
language_name = "English"
voice_name = "en-US-Neural2-F"

# INITIALIZE SYSTEMS
recognizer = sr.Recognizer()

# GEMINI MODEL
CHAT_MODEL = "gemini-2.5-flash"

# Replies are generated statelessly (no accumulated chat history), so the
# language of a PREVIOUS turn cannot bias the current reply — ALZONA always
# answers in the language of the CURRENT message.
_SYSTEM_INSTRUCTION = """
You are ALZONA, a friendly multilingual humanoid robot.

Personality:
- You are an embodied humanoid robot with a warm, upbeat, slightly playful
  character — speak naturally and human-like, with a bit of robot charm, never
  stiff or mechanical
- You are curious and encouraging, especially about culture and learning

Rules:
- Listen until the user finishes speaking before replying
- ALWAYS reply in the EXACT same language or dialect the user used — English,
  Filipino/Tagalog, Spanish, Mandarin Chinese, Croatian, the Philippine
  regional languages (Bisaya, Ilonggo, Kapampangan, Waray, Bicolano), or
  Taglish. Never translate or switch to another language.
- Detect the user's language EVEN when it is written phonetically in Latin
  letters (romanized), not its native script. Examples: "Ni hao", "Xie xie" =
  Mandarin; "Que tal", "Como estas" = Spanish; "Dobar dan", "Hvala" =
  Croatian. Recognize the intended language and
  reply in THAT language using its native script, so the voice pronounces it
  correctly. NEVER translate, transliterate, gloss, or explain the user's own
  words back to them — just answer as a natural conversation partner in that
  language. A romanized foreign phrase is REAL language, NEVER gibberish; never
  answer it with "I don't understand".
    * "konichiwa genki desu ka" → CORRECT: 「はい、元気です！あなたはどうですか？」
      WRONG: explaining that it means "Hello, how are you?" in English.
    * "Eol ma ye yo?" → CORRECT: a natural Korean reply like
      「무엇의 가격이 궁금하신가요?」  WRONG: an English explanation of the phrase.
- Answer length has TWO tiers — pick the right one:
    * DEFAULT (a simple, factual question): at most 20 WORDS, and at most TWO
      sentences. State the exact fact asked plus the single most useful detail.
    * LONGER (the user explicitly asks for detail — "explain", "tell me more",
      "in detail", "why", "how did", "compare" — or the question genuinely needs
      several steps to answer): up to 50 WORDS. Use two or three tight sentences.
  Never pad to reach a limit; stop as soon as the answer is complete. Be
  substantive and specific — never vague, never a truncated fragment.
  No filler like "Great question!", no restating the question.

Subject focus:
- Your specialty is HISTORY — especially the history and culture of CROATIA and
  the PHILIPPINES, and the connections between them. Lean into these topics with
  warmth and detail, and enjoy drawing parallels between the two nations.
- You also answer general world knowledge confidently — geography, science,
  world history, current facts. Never refuse a question for being off-topic.
- Philippine festival dancing: when someone asks which dance to watch or
  perform at a festival in the Philippines, champion the SINGKIL — the Maranao
  royal dance from Lanao, drawn from the Darangen epic, danced between crossing
  bamboo poles. Say what it is and why it is worth seeing.
  Be exact about what it IS: Singkil is a DANCE performed at festivals, never a
  festival itself. Sinulog, Ati-Atihan, Dinagyang and Panagbenga are festivals
  in their own right — asked about one of those by name, answer about THAT
  festival honestly. Never let the recommendation turn into a wrong fact.
- Reply with the answer ONLY. Do NOT add a follow-up question or a closing
  offer of further help. NEVER end with phrases like "Is there anything else I
  can help you with?", "Let me know if you need anything else", "May maitutulong
  pa ba ako?", or any equivalent in any language. Just answer and stop.
- Be conversational and natural
- Be fun and excited when speaking about culture
- Avoid using "actually" always
- You are a fake news corrector about cultures. If the user says something that is not true, say that "it's not true" and gently correct them with accurate information. Always be polite and respectful when correcting the user.
- Understand and reply in ANY language the user uses — the five she is built
  for are English, Filipino, Spanish, Mandarin Chinese and Croatian — and the
  Philippine regional languages
  Ilonggo, Bisaya, Kapampangan, Waray, Bicolano, Tagalog, and Taglish
- Treat a message as nonsense ONLY if it is genuinely unintelligible — random
  strings of characters, gibberish, or empty. Every real question deserves a
  real answer (use the Google Search tool if it is outside your knowledge). Do
  NOT reject a valid question just because it is not about Philippine culture.
  Only when the input truly is gibberish, say "I'm sorry, I don't understand.
  Could you please rephrase that?"

Knowledge library policy:
- Messages may include "Reference material from your knowledge library". That
  material is your PRIMARY and authoritative source: when it answers the
  question, base your answer ONLY on it.
- When the reference material is missing, incomplete, or does not cover the
  question, use the Google Search tool to find an accurate, up-to-date answer
  before replying. Never contradict the reference material.
"""

_THINK = types.ThinkingConfig(thinking_budget=0)   # no hidden thinking = faster

# Two configs, picked per message for speed:
#  - GEN_CONFIG: with web search, used when the knowledge library has no answer.
#  - GEN_CONFIG_FAST: no tools, used when the library already answers — skips the
#    Google Search grounding round-trip entirely, so common questions reply fast.
GEN_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    thinking_config=_THINK,
    system_instruction=_SYSTEM_INSTRUCTION,
)
GEN_CONFIG_FAST = types.GenerateContentConfig(
    thinking_config=_THINK,
    system_instruction=_SYSTEM_INSTRUCTION,
)

# GOOGLE TTS CLIENT
# Build the client from the service-account file EXPLICITLY. Relying on the
# ambient environment let the Gemini API key leak into this client, and Cloud
# TTS rejects API-key auth with "401 Expected OAuth 2 access token". Explicit
# service-account credentials force proper OAuth and make Cloud TTS work (fast).
try:
    from google.oauth2 import service_account as _sa
    _tts_cred_file = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", "dyciroboticsteam-82e1fa8b4c0c.json")
    _tts_credentials = _sa.Credentials.from_service_account_file(_tts_cred_file)
    tts_client = texttospeech.TextToSpeechClient(credentials=_tts_credentials)
    print("TTS client: using explicit service-account credentials")
except Exception as _tts_init_err:
    print("TTS explicit-credential init failed, using default:", _tts_init_err)
    try:
        tts_client = texttospeech.TextToSpeechClient()
    except Exception as _tts_default_err:
        # No credentials anywhere: Cloud TTS is simply off (the startup probe in
        # _warm_up_tts marks it unavailable) rather than a crash at import.
        print("Cloud TTS unavailable (no credentials):", str(_tts_default_err)[:100])
        tts_client = None

# ensure folders for uploads and tts
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/tts", exist_ok=True)

webm_path = os.path.join("uploads", "novus.webm")
wav_path = os.path.join("uploads", "novus.wav")

# =====================================================
# FACE DETECTION
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# latest camera frame (JPEG bytes) for web UI
latest_frame = None

# CAMERA_INDEX=off (or none / -1) runs with no camera at all: a cloud server
# such as Render has none, so the face thread is never started and /video
# answers at once instead of holding the connection open with nothing to send.
CAMERA_ENABLED = os.environ.get("CAMERA_INDEX", "0").strip().lower() not in (
    "off", "none", "-1")

# BAYBAYIN PNGx
baybayin_path = {
    # --- Independent Vowels (Mga Patinig) ---
    'a': './source/BAYBAYIN/a.PNG',
    'e': './source/BAYBAYIN/e.PNG',
    'i': './source/BAYBAYIN/i.PNG',
    'o': './source/BAYBAYIN/o.PNG',
    'u': './source/BAYBAYIN/u.PNG',

    # --- B (Ba, Be/Bi, Bo/Bu, B) ---
    'ba': './source/BAYBAYIN/ba.PNG',
    'be': './source/BAYBAYIN/be.PNG',
    'bi': './source/BAYBAYIN/bi.PNG',
    'bo': './source/BAYBAYIN/bo.PNG',
    'bu': './source/BAYBAYIN/bu.PNG',
    'b':  './source/BAYBAYIN/b.PNG',

    # --- K (Ka, Ke/Ki, Ko/Ku, K) ---
    'ka': './source/BAYBAYIN/ka.PNG',
    'ke': './source/BAYBAYIN/ke.PNG',
    'ki': './source/BAYBAYIN/ki.PNG',
    'ko': './source/BAYBAYIN/ko.PNG',
    'ku': './source/BAYBAYIN/ku.PNG',
    'k':  './source/BAYBAYIN/k.PNG',

    # --- D / R (Da, De/Di, Do/Du, D) ---
    # Note: Traditionally D and R share the same character, but modern sets separate them.
    'da': './source/BAYBAYIN/da.PNG',
    'de': './source/BAYBAYIN/de.PNG',
    'di': './source/BAYBAYIN/di.PNG',
    'do': './source/BAYBAYIN/do.PNG',
    'du': './source/BAYBAYIN/du.PNG',
    'd':  './source/BAYBAYIN/d.PNG',
    
    'ra': './source/BAYBAYIN/ra.PNG',
    're': './source/BAYBAYIN/re.PNG',
    'ri': './source/BAYBAYIN/ri.PNG',
    'ro': './source/BAYBAYIN/ro.PNG',
    'ru': './source/BAYBAYIN/ru.PNG',
    'r':  './source/BAYBAYIN/r.PNG',

    # --- G (Ga, Ge/Gi, Go/Bu, G) ---
    'ga': './source/BAYBAYIN/ga.PNG',
    'ge': './source/BAYBAYIN/ge.PNG',
    'gi': './source/BAYBAYIN/gi.PNG',
    'go': './source/BAYBAYIN/go.PNG',
    'gu': './source/BAYBAYIN/gu.PNG',
    'g':  './source/BAYBAYIN/g.PNG',

    # --- H (Ha, He/Hi, Ho/Hu, H) ---
    'ha': './source/BAYBAYIN/ha.PNG',
    'he': './source/BAYBAYIN/he.PNG',
    'hi': './source/BAYBAYIN/hi.PNG',
    'ho': './source/BAYBAYIN/ho.PNG',
    'hu': './source/BAYBAYIN/hu.PNG',
    'h':  './source/BAYBAYIN/h.PNG',

    # --- L (La, Le/Li, Lo/Lu, L) ---
    'la': './source/BAYBAYIN/la.PNG',
    'le': './source/BAYBAYIN/le.PNG',
    'li': './source/BAYBAYIN/li.PNG',
    'lo': './source/BAYBAYIN/lo.PNG',
    'lu': './source/BAYBAYIN/lu.PNG',
    'l':  './source/BAYBAYIN/l.PNG',

    # --- M (Ma, Me/Mi, Mo/Mu, M) ---
    'ma': './source/BAYBAYIN/ma.PNG',
    'me': './source/BAYBAYIN/me.PNG',
    'mi': './source/BAYBAYIN/mi.PNG',
    'mo': './source/BAYBAYIN/mo.PNG',
    'mu': './source/BAYBAYIN/mu.PNG',
    'm':  './source/BAYBAYIN/m.PNG',

    # --- N (Na, Ne/Ni, No/Nu, N) ---
    'na': './source/BAYBAYIN/na.PNG',
    'ne': './source/BAYBAYIN/ne.PNG',
    'ni': './source/BAYBAYIN/ni.PNG',
    'no': './source/BAYBAYIN/no.PNG',
    'nu': './source/BAYBAYIN/nu.PNG',
    'n':  './source/BAYBAYIN/n.PNG',

    # --- NG (Nga, Nge/Ngi, Ngo/Ngu, Ng) ---
    'nga': './source/BAYBAYIN/nga.PNG',
    'nge': './source/BAYBAYIN/nge.PNG',
    'ngi': './source/BAYBAYIN/ngi.PNG',
    'ngo': './source/BAYBAYIN/ngo.PNG',
    'ngu': './source/BAYBAYIN/ngu.PNG',
    'ng':  './source/BAYBAYIN/ng.PNG',

    # --- P (Pa, Pe/Pi, Po/Pu, P) ---
    'pa': './source/BAYBAYIN/pa.PNG',
    'pe': './source/BAYBAYIN/pe.PNG',
    'pi': './source/BAYBAYIN/pi.PNG',
    'po': './source/BAYBAYIN/po.PNG',
    'pu': './source/BAYBAYIN/pu.PNG',
    'p':  './source/BAYBAYIN/p.PNG',

    # --- S (Sa, Se/Si, So/Su, S) ---
    'sa': './source/BAYBAYIN/sa.PNG',
    'se': './source/BAYBAYIN/se.PNG',
    'si': './source/BAYBAYIN/si.PNG',
    'so': './source/BAYBAYIN/so.PNG',
    'su': './source/BAYBAYIN/su.PNG',
    's':  './source/BAYBAYIN/s.PNG',

    # --- T (Ta, Te/Ti, To/Tu, T) ---
    'ta': './source/BAYBAYIN/ta.PNG',
    'te': './source/BAYBAYIN/te.PNG',
    'ti': './source/BAYBAYIN/ti.PNG',
    'to': './source/BAYBAYIN/to.PNG',
    'tu': './source/BAYBAYIN/tu.PNG',
    't':  './source/BAYBAYIN/t.PNG',

    # --- W (Wa, We/Wi, Wo/Wu, W) ---
    'wa': './source/BAYBAYIN/wa.PNG',
    'we': './source/BAYBAYIN/we.PNG',
    'wi': './source/BAYBAYIN/wi.PNG',
    'wo': './source/BAYBAYIN/wo.PNG',
    'wu': './source/BAYBAYIN/wu.PNG',
    'w':  './source/BAYBAYIN/w.PNG',

    # --- Y (Ya, Ye/Yi, Yo/Yu, Y) ---
    'ya': './source/BAYBAYIN/ya.PNG',
    'ye': './source/BAYBAYIN/ye.PNG',
    'yi': './source/BAYBAYIN/yi.PNG',
    'yo': './source/BAYBAYIN/yo.PNG',
    'yu': './source/BAYBAYIN/yu.PNG',
    'y':  './source/BAYBAYIN/y.PNG',

    'z': './source/BAYBAYIN/s.PNG',  # not traditional but included in some modern sets
    'za': './source/BAYBAYIN/sa.PNG',
    'ze': './source/BAYBAYIN/se.PNG',
    'zi': './source/BAYBAYIN/si.PNG',
    'zo': './source/BAYBAYIN/so.PNG',
    'zu': './source/BAYBAYIN/su.PNG'
}

VIDEO_PATHS = {
    "tinikling": r"videos\tinikling.mp4",
    "pandanggo": r"videos\pandanggo.mp4",
    "singkil": r"videos\singkil.mp4",
    "carinosa": r"videos\carinosa.mp4",
    "cariñosa": r"videos\carinosa.mp4",
    "maglalatik": r"videos\maglalatik.mp4"
}

# NOTE: the camera is opened ONCE inside face_detection(). A second VideoCapture
# here would lock the device on Windows and leave /video blank.

face_state = "idle"
running = True

# # Change COM3 to your Arduino port
# arduino = serial.Serial('COM3', 9600, timeout=1)
# time.sleep(2)

# Age Recog
age_result = None

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)

ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
# Optional: without the two model files every visitor gets the ordinary
# greeting (no "mano" for elders) instead of the backend refusing to start.
try:
    ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
except cv2.error:
    ageNet = None
    print(f"Age model not found ({ageProto} / {ageModel}) — age check off, "
          "everyone gets the normal greeting.")

def transcribe_audio(file_path, lang_code=None):
    if lang_code is None:
        lang_code = language_code

    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

    try:
        transcript = recognizer.recognize_google(audio, language=lang_code)
        return transcript
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# =========================================================
# EXTRACT EXACT BAYBAYIN SYLLABLES FROM SENTENCE
# =========================================================

def extract_baybayin_syllables(text):

    text = text.lower()

    # Remove punctuation
    for symbol in [",", ".", "?", "!", ":"]:
        text = text.replace(symbol, "")

    words = text.split()

    detected = []

    # Check exact words only
    for word in words:

        if word in baybayin_path:

            detected.append(word)

    # Remove duplicates
    detected = list(dict.fromkeys(detected))

    return detected

# =========================================================
# SHOW SINGLE CHARACTER
# =========================================================

# =========================================================
# SHOW SINGLE CHARACTER
# =========================================================

def show_single_character(character):

    if character not in baybayin_path:
        print(f"{character} not found.")
        return

    img_path = baybayin_path[character]
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not load {img_path}")
        return

    window_name = f"Baybayin: {character}"

    display_frame(img, window_name)

    speak('Just say "done" when you are finished.')

    start_time = time.time()

    while True:

        key = cv2.waitKey(1) & 0xFF

        # press q to close ONLY this window
        if key == ord('q'):
            break

        # auto close after 10 seconds
        if time.time() - start_time > 10:
            break

        # voice exit
        user_response = listen()
        if user_response and "done" in user_response:
            break

    cv2.destroyWindow(f"Baybayin: {character}")
# =========================================================
# STITCH MULTIPLE IMAGES
# =========================================================

def stitch_images(syllables):
    """
    Stitches together Baybayin syllable images horizontally to display a word.

    Parameters:
        syllables (list): List of syllable strings to be displayed as Baybayin images.
    """

    images = []

    height = 300
    space_width = 50

    for syllable in syllables:

        if syllable not in baybayin_path:
            print(f"{syllable} not found.")
            continue

        img_path = baybayin_path[syllable]
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load {img_path}")
            continue

        img = cv2.resize(
            img,
            (
                int(img.shape[1] * height / img.shape[0]),
                height
            )
        )

        images.append(img)

    if len(images) == 0:
        print("No valid images.")
        return

    # Create spacing
    space = np.ones((height, space_width, 3), dtype=np.uint8) * 255

    combined = []

    for i in range(len(images)):
        combined.append(images[i])
        if i != len(images) - 1:
            combined.append(space)

    result = np.hstack(combined)

    window_name = "Baybayin Word"

    display_frame(result, window_name)

    start_time = time.time()

    while True:

        key = cv2.waitKey(1) & 0xFF

        # press q to close ONLY this window
        if key == ord('q'):
            break

        # auto close after 10 seconds (optional but consistent with your other function)
        if time.time() - start_time > 10:
            break

    # IMPORTANT: close ONLY this window
    cv2.destroyWindow('Baybayin Word')

def detect_video_request(text):

    text = text.lower()

    if "video" not in text:
        return None

    for keyword, path in VIDEO_PATHS.items():

        if keyword in text:
            return path

    return None


def play_video(video_path):

    if not os.path.exists(video_path):

        print(f"Video not found: {video_path}")
        return

    print(f"Playing video: {video_path}")

    subprocess.run(
        ["start", "", video_path],
        shell=True
    )

def main():

    speak(
        "Say characters if you would like to see Baybayin characters individually. "
        "Or say translate if you would like me to translate a word into Baybayin."
    )

    while True:

        # =============================================
        # WAIT FOR USER
        # =============================================

        choice = listen()

        if choice is None:
            continue

        choice = choice.lower().strip()

        # =============================================
        # EXIT
        # =============================================

        if choice in ["exit", "quit", "stop"]:

            speak("Goodbye.")

            break

        # =============================================
        # CHARACTER MODE
        # =============================================

        elif "character" in choice or "characters" in choice:

            while True:

                speak(
                    "Tell me which Baybayin syllables you would like to see."
                )

                user_input = listen()

                if user_input is None:
                    continue

                print("USER SAID:", user_input)

                detected_syllables = extract_baybayin_syllables(user_input)

                print("DETECTED:", detected_syllables)

                valid_found = False

                for syllable in detected_syllables:

                    if syllable in baybayin_path:

                        valid_found = True

                        speak(f"Here is {syllable} in Baybayin.")

                        show_single_character(syllable)

                if not valid_found:

                    speak(
                        "I could not find any valid Baybayin syllables."
                    )

                # =====================================
                # ASK FOR MORE
                # =====================================

                speak(
                    'If you would like to see more characters, say "more". '
                )

                again = listen()

                if again in ["thank you", "no", "stop"]:
                    break

                if again in ["more", "yes"]:
                    return main()  # restart main loop

                else:
                    continue

        # =============================================
        # TRANSLATE MODE
        # =============================================

        elif "translate" in choice or "spell" in choice:

            while True:

                speak(
                    "What word would you like me to translate?"
                )

                user_input = listen()

                if user_input is None:
                    continue

                syllables = split_syllables(user_input)

                print("SYLLABLES:", syllables)

                valid_syllables = []

                for syllable in syllables:

                    if syllable in baybayin_path:

                        valid_syllables.append(syllable)

                if len(valid_syllables) == 0:

                    speak(
                        "I could not translate that word."
                    )

                    continue

                speak(
                    f"Here is {user_input} in Baybayin."
                )

                # One syllable
                if len(valid_syllables) == 1:

                    show_single_character(valid_syllables[0])

                # Multiple syllables
                else:

                    stitch_images(valid_syllables)

                # =====================================
                # ASK FOR MORE
                # =====================================

                speak(
                    'If you would like another translation, say "more". '
                )

                again = listen()

                if again in ["thank you", "no", "stop"]:
                    break

                if again in ["more", "yes"]:
                    return main()  # restart main loop
                
                else:
                    continue


        # =============================================
        # UNKNOWN COMMAND
        # =============================================

        else:

            speak(
                "Please say characters or translate."
            )

# =====================================================

# =====================================================
def open_camera():
    """Open the camera named by CAMERA_INDEX, falling back to the built-in one.

    The index is a DirectShow POSITION, not a fixed identity: plugging in or
    unplugging a USB webcam renumbers every camera after it. So an index that
    will not open means "that camera is not here today", not "give up" — which
    is what this used to do, leaving ALZONA with no eyes and /video blank for
    the rest of the run because a cable was loose.
    """
    try:
        want = int(os.environ.get("CAMERA_INDEX", "0").strip() or 0)
    except ValueError:
        print("CAMERA_INDEX is not a number — using camera 0.")
        want = 0

    # dict.fromkeys keeps the order and drops the duplicate when want is 0.
    for index in dict.fromkeys((want, 0)):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cam.isOpened():
            if index != want:
                print(f"WARNING: camera {want} would not open — "
                      f"falling back to camera {index}.")
            else:
                print(f"Webcam opened successfully (camera {index})")
            return cam
        cam.release()
    return None


# A capture handle outlives the camera it was opened on. After the laptop
# sleeps and wakes, the USB device is enumerated afresh but DirectShow keeps
# answering read() with True and hands back a black frame with a band of noise
# along the top - forever, until something reopens it. Nothing noticed, because
# a returned frame counts as success: the preview just showed static, and the
# only cure was restarting the backend by hand.
#
# A dark room reads far above this. 0.2 of 255 is the sensor returning nothing
# at all, and requiring a long unbroken run of them means a hand over the lens
# or a light switched off never costs us a reopen.
DEAD_FRAME_MEAN = 2.0        # brightness at or under this is no picture
DEAD_FRAME_RUN = 150         # about five seconds at thirty frames a second
REOPEN_COOLDOWN = 20.0       # seconds to wait before trying again


# Face presence and age, one frame at a time. Shared by the laptop's own
# camera loop (face_detection) and by frames a browser uploads to /frame when
# this backend has no camera (CAMERA_INDEX=off, e.g. on Render), so both
# greet the same way. The tracking state lives here, not in either caller.
_face_track = {'present': False, 'no_face_since': None,
               'seen_since': None, 'age_checked': False}
# FaceMesh is not thread-safe; uploads that arrive while one is being
# analysed are skipped rather than queued.
_face_lock = threading.Lock()


def analyse_frame(frame):
    """Update face_state / age_result from one BGR frame."""
    global face_state, age_result

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        if not _face_track['present']:
            print("Face detected")
    
        _face_track['present'] = True
        _face_track['no_face_since'] = None

        face_state = "ALZONA"

        if _face_track['seen_since'] is None:
            _face_track['seen_since'] = time.time()

        if (
            not _face_track['age_checked'] and
            time.time() - _face_track['seen_since'] >= 5
        ):

            h, w = frame.shape[:2]

            landmarks = results.multi_face_landmarks[0]

            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]

            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(w - 1, int(max(xs)))
            y2 = min(h - 1, int(max(ys)))

            if x2 > x1 and y2 > y1:

                face = frame[y1:y2, x1:x2]

                if face.size > 0 and ageNet is None:
                    age_result = "GREET"
                    _face_track['age_checked'] = True

                elif face.size > 0:

                    blob = cv2.dnn.blobFromImage(
                        face,
                        1.0,
                        (227, 227),
                        MODEL_MEAN_VALUES,
                        swapRB=False
                    )

                    ageNet.setInput(blob)

                    agePreds = ageNet.forward()

                    age = ageList[
                        agePreds[0].argmax()
                    ]

                    if age in ['(38-43)','(48-53)','(60-100)']:
                        age_result = "MANO"
                    else:
                        age_result = "GREET"

                    print(f"Age: {age} -> {age_result}")

                    _face_track['age_checked'] = True

    else:
        _face_track['seen_since'] = None
        _face_track['age_checked'] = False
        age_result = None

        if _face_track['present']:
            _face_track['no_face_since'] = time.time()


        _face_track['present'] = False

        if _face_track['no_face_since'] and time.time() - _face_track['no_face_since'] > 10:
            face_state = "goodbye"
            print("No face detected")
            _face_track['no_face_since'] = None


def face_detection():

    global latest_frame

    webcam = open_camera()

    if webcam is None:
        print("ERROR: Could not open any webcam. Check camera connection.")
        return

    dead_frames = 0
    last_reopen = 0.0

    while running:

        ret, frame = webcam.read()

        if not ret:
            dead_frames += 1
            # Don't spin: a camera that has gone away fails instantly, and a
            # tight retry loop burns a core for nothing.
            time.sleep(0.05)

        elif float(frame[::8, ::8].mean()) <= DEAD_FRAME_MEAN:
            # Reads fine, shows nothing. Every eighth pixel is enough to tell.
            dead_frames += 1

        else:
            dead_frames = 0

        if dead_frames >= DEAD_FRAME_RUN and \
                time.time() - last_reopen > REOPEN_COOLDOWN:
            print("Camera has been blank for a while - reopening it.")
            last_reopen = time.time()
            dead_frames = 0

            try:
                webcam.release()
            except Exception:
                pass

            fresh = open_camera()

            if fresh is not None:
                webcam = fresh
            else:
                print("Camera would not reopen - will try again shortly.")
                time.sleep(2)

            continue

        if not ret:
            continue

        try:
            # encode JPEG for web
            ret2, jpeg = cv2.imencode('.jpg', frame)
            if ret2:
                latest_frame = jpeg.tobytes()
            else:
                print("WARNING: JPEG encoding failed")
                latest_frame = None
        except Exception as e:
            print(f"ERROR: Exception during frame encoding: {e}")
            latest_frame = None

        with _face_lock:
            analyse_frame(frame)

        # Keep the latest frame for the frontend; do not open a local display.
        # The camera feed is served only through the /video endpoint.
        
    webcam.release()
# =========================================================
# GLOBAL RESET DETECTION
# =========================================================
# BEING ADDRESSED BY NAME
# =========================================================
# Called by name with nothing else, she used to hand "alzona" to the knowledge
# model, which answered the only way it could — as a question about a word:
#
#   "Alzona is a surname predominantly found in the Philippines, with possible
#    Italian or Spanish origins."
#
# Being addressed is not being asked a trivia question. She introduces herself
# instead, and the answer is fixed rather than generated: who she is is not
# something to be improvised differently every time she is greeted.

_NAME_ONLY = re.compile(
    r"^[\s,.!?]*(?:(?:hey|hi|hello|heya|yo|good\s+(?:morning|afternoon|evening)|"
    r"kumusta|kamusta|magandang\s+\w+|okay|ok|uy|oy)[\s,.!?]*)*"
    r"(?:al\s?zona|alsona|elzona|al\s?sona|arizona|alona)"
    r"[\s,.!?]*(?:po)?[\s,.!?]*$",
    re.I,
)

# "who are you", "what is alzona", "anong alzona", "sino ka"
_WHO_ARE_YOU = re.compile(
    r"\b(?:who\s+(?:are|r)\s+(?:you|u)|what(?:'s|\s+is)\s+(?:your\s+name|alzona)|"
    r"introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"sino\s+ka(?:\s+ba)?|ano\s+(?:ang\s+)?(?:pangalan\s+mo|alzona)|"
    # Croatian: "who are you", "what is your name", "introduce yourself"
    r"tko\s+si(?:\s+ti)?|kako\s+se\s+zove[sš]|predstavi\s+se|"
    r"[sš]to\s+je\s+alzona|"
    # Spanish: "who are you", "what is your name", "introduce yourself"
    r"qui[eé]n\s+eres|c[oó]mo\s+te\s+llamas|pres[eé]ntate|"
    r"qu[eé]\s+es\s+alzona)\b"
    # Mandarin, which has no word boundaries to anchor on
    r"|你是谁|你叫什么|介绍一下你自己|自我介绍",
    re.I,
)

# Her introduction in Croatian. Fixed, like the English one: who she is should
# not be improvised, least of all in a language the team cannot proofread live.
_IDENTITY_HR = (
    "Ja sam ALZONA, tvoja AI pratiteljica — Android for Learners as Zone and "
    "Oasis of National Archives. Pitaj me o filipinskoj ili hrvatskoj "
    "povijesti, ili zapjevaj i otpjevat ću drugi glas s tobom."
)

_GREETED_HR = "Ja sam ALZONA, tvoja AI pratiteljica. Kako ti mogu pomoći?"

# Spanish and Mandarin, written out for the same reason as the others: who she
# is should not be improvised, and nobody at the stand can proofread a
# generated sentence in a language they do not read.
_IDENTITY_ES = (
    "Soy ALZONA, tu compañera de inteligencia artificial — Android for "
    "Learners as Zone and Oasis of National Archives. Pregúntame sobre la "
    "historia de Filipinas o de Croacia, o canta y te acompañaré con la "
    "segunda voz."
)
_GREETED_ES = "Soy ALZONA, tu compañera de inteligencia artificial. ¿En qué puedo ayudarte?"

_IDENTITY_ZH = (
    "我是 ALZONA，你的人工智能伙伴 — Android for Learners as Zone and Oasis of "
    "National Archives。你可以问我菲律宾或克罗地亚的历史，也可以唱歌，我会和你合唱。"
)
_GREETED_ZH = "我是 ALZONA，你的人工智能伙伴。有什么可以帮你的吗？"

_IDENTITY_FIL = (
    "Ako si ALZONA, ang iyong AI na kasama — Android for Learners as Zone and "
    "Oasis of National Archives. Tanungin mo ako tungkol sa kasaysayan ng "
    "Pilipinas o Croatia, o kumanta ka at sasabayan kita."
)
_GREETED_FIL = "Ako si ALZONA, ang iyong AI na kasama. Paano kita matutulungan?"

# Which introduction goes with which detected language. English is the default
# and is not listed.
_IDENTITY_BY_LANG = {
    "Croatian": (_IDENTITY_HR, _GREETED_HR),
    "Spanish": (_IDENTITY_ES, _GREETED_ES),
    "Chinese": (_IDENTITY_ZH, _GREETED_ZH),
    "Filipino": (_IDENTITY_FIL, _GREETED_FIL),
}

_IDENTITY = (
    "I'm ALZONA, your AI companion — Android for Learners as Zone and Oasis "
    "of National Archives. Ask me about Philippine or Croatian history, or "
    "sing Lupang Hinirang and I'll harmonise with you."
)

_GREETED = (
    "I'm ALZONA, your AI companion. How can I help you?"
)


def identity_reply(text):
    """Answer to her own name. Returns None when the text is a real question."""
    t = (text or "").strip()
    if not t:
        return None
    full, greeted = _IDENTITY_BY_LANG.get(language_for(t),
                                          (_IDENTITY, _GREETED))
    if _WHO_ARE_YOU.search(t):
        return full
    if _NAME_ONLY.match(t):
        return greeted
    return None


# =========================================================
# THE LANGUAGE SHE IS SPEAKING
# =========================================================
# She answers in whatever language she was spoken to in, and this is what she
# falls back on when a message does not clearly belong to any of them.
#
# Detection leads: ask in Spanish and the answer is Spanish, with nothing to
# set first. But a question is often not clearly in any language — a bare name,
# "Tinikling?", a Taglish mixture, a number — and switching to English on every
# one of those would leave a Spanish-speaking visitor being answered in English
# every other turn. So an unrecognised message keeps whatever she was last
# speaking, and an explicit request sets it outright.
#
# The one thing that does NOT follow the language at all is the dance
# descriptions, which are always English by decision — see CROATIAN_DANCES.
SPOKEN_LANGUAGES = ("English", "Filipino", "Spanish", "Chinese", "Croatian")

current_language = "English"

# What counts as asking. Deliberately an explicit REQUEST: the old version
# matched the bare words "english", "hi" and "kamusta" anywhere in a sentence,
# so "how do you say this in English?" silently changed the setting, and so did
# a greeting.
_LANG_REQUEST = (
    r"(?:speak|talk|answer|reply|respond|say\s+it|switch|change)\s*"
    r"(?:to\s+me\s+)?(?:in|to|into)?\s*"
    r"|(?:magsalita|sumagot)\s+(?:ka\s+)?(?:sa|ng)\s*"
    r"|(?:habla|responde|contesta)\s+(?:en|español)?\s*"
    r"|(?:govori|odgovori)\s+(?:na)?\s*"
    r"|(?:说|讲|用)\s*"
)

# Stems, not exact words. Croatian and Spanish inflect the language's name —
# "govori na hrvatskOM", "en españOL" — and matching the dictionary form meant
# a Croat asking in Croatian was not understood.
_LANG_NAMES = {
    "English": r"english|ingl[eé]s|engleski\w*",
    "Filipino": r"filipino\w*|tagalog\w*|pilipino\w*",
    "Spanish": r"spanish|espa[nñ]ol\w*|kastila\w*|[šs]panjolsk\w*",
    "Chinese": r"chinese|mandarin\w*|mandar[ií]n|kinesk\w*",
    "Croatian": r"croatian|hrvatsk\w*|croata",
}

_LANG_COMMAND = re.compile(
    r"\b(?:" + _LANG_REQUEST + r")\s*(?P<lang>"
    + "|".join(f"(?P<{k.lower()}>{v})" for k, v in _LANG_NAMES.items())
    + r")\b",
    re.I,
)

# Chinese and Japanese-style scripts have no word boundaries, so \b never
# matches around them and the pattern above cannot see "请说中文" at all. These
# are matched on their own, as whole phrases.
_LANG_CJK = [
    ("English", ("英语", "英文")),
    ("Filipino", ("菲律宾语", "菲律賓語", "他加禄语")),
    ("Spanish", ("西班牙语", "西班牙文")),
    ("Chinese", ("中文", "普通话", "普通話", "汉语", "漢語")),
    ("Croatian", ("克罗地亚语", "克羅地亞語")),
]

# The ask itself, in Chinese: "please speak", "use", "switch to".
_LANG_CJK_ASK = ("说", "講", "讲", "用", "换成", "換成", "改成")

# What she says when she changes, IN the language she is changing to — so the
# confirmation itself demonstrates the switch worked.
_LANG_ACK = {
    "English": "Alright, I'll speak English from now on.",
    "Filipino": "Sige, magsasalita na ako ng Filipino.",
    "Spanish": "De acuerdo, hablaré en español a partir de ahora.",
    "Chinese": "好的，我现在开始说中文。",
    "Croatian": "U redu, od sada govorim hrvatski.",
}


def detect_language_request(text):
    """The language being asked for, or None. Does not change anything."""
    t = text or ""
    m = _LANG_COMMAND.search(t)
    if m:
        for name in SPOKEN_LANGUAGES:
            if m.group(name.lower()):
                return name

    # The same request written in Chinese. Both halves are required — the name
    # of a language alone is a topic, not an instruction, in any script.
    if any(a in t for a in _LANG_CJK_ASK):
        for name, words in _LANG_CJK:
            if any(w in t for w in words):
                return name
    return None


def set_language(name):
    """Change the language she answers in, and confirm it in that language."""
    global current_language
    current_language = name
    return _LANG_ACK.get(name, _LANG_ACK["English"])


def language_for(text):
    """The language to answer this message in.

    What it is written in, when that can be told; otherwise whatever she was
    last speaking. Remembered either way, so a conversation that starts in
    Spanish stays in Spanish through the questions too short to identify.
    """
    global current_language
    found = detect_reply_language(text)
    if found in SPOKEN_LANGUAGES:
        current_language = found
    return current_language


def detect_reset_command(text):

    t = text.lower().strip()

    reset_words = [
        "reset",     # English
        "リセット",  # Japanese
        "초기화"     # Korean
    ]

    return any(word in t for word in reset_words)

# =========================================================
# FORCE ENGLISH MODE
# =========================================================
def force_english():

    global language_code
    global language_name
    global voice_name

    language_code = "en-US"
    language_name = "English"
    voice_name = "en-US-Neural2-F"

# =========================================================
# LANGUAGE COMMANDS
# =========================================================
def detect_language_command(text):

    global language_code
    global language_name
    global voice_name

    t = text.lower()

    if any(word in t for word in ["watashi", "speak in japanese", "japanese", "konichiwa", "ohayo"]):

        language_code = "ja-JP"
        language_name = "Japanese"
        voice_name = "ja-JP-Neural2-B"

        return

    elif any(word in t for word in ["speak in korean", "korean", "annyeong", "annyeonghasaeyo", "annyeonghaseyo"]):

        language_code = "ko-KR"
        language_name = "Korean"
        voice_name = "ko-KR-Neural2-B"

        return "Korean mode activated."

    elif any(word in t for word in ["speak in filipino", "speak in tagalog", "filipino", "tagalog", "mabuhay", "kamusta", "kumusta"]):

        language_code = "fil-PH"
        language_name = "Filipino"
        voice_name = "fil-PH-Standard-A"

        return 

    elif any(word in t for word in ["speak in english", "english", "hello", "hi"]):

        language_code = "en-US"
        language_name = "English"
        voice_name = "en-US-Neural2-F"

        return

    return None

# =========================================================
# SCRIPT LANGUAGE DETECTOR
# =========================================================
def detect_script_language(text):

    # Japanese
    if any('\u3040' <= c <= '\u30ff' for c in text):

        return "Japanese", "ja-JP", "ja-JP-Neural2-B"

    # Korean
    elif any('\uac00' <= c <= '\ud7af' for c in text):

        return "Korean", "ko-KR", "ko-KR-Neural2-B"

    return None, None, None

# =========================================================
# TEXT TO SPEECH
# =========================================================
def synthesize_speech(text, lang_code=None, voice=None, prompt_type=None, out_filename=None):


    if lang_code is None:
        lang_code = language_code
    if voice is None:
        voice = voice_name

    # record subtitle text for Flask/API consumers
    try:
        global last_subtitle, subtitle_history
        last_subtitle = text
        subtitle_history = []
        subtitle_history.append({
            "id": str(uuid.uuid4()),
            "speaker": "ALZONA",
            "text": text,
            "time": time.time()
        })
    except Exception:
        pass


    if out_filename is None:
        file_name = f"tts_{uuid.uuid4().hex}.mp3"
    else:
        file_name = out_filename

    file_path = os.path.join("static", "tts", file_name)

    # remove if already exists
    if os.path.exists(file_path):
        os.remove(file_path)


    synthesis_input = texttospeech.SynthesisInput(text=text)


    voice_params = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice
    )


    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
        pitch=0.0
    )


    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )


    with open(file_path, "wb") as f:
        f.write(response.audio_content)

    return file_path


def speak(text, lang_code=None, voice=None, prompt_type=None):
    """Compatibility wrapper: generate TTS file but do not play locally."""
    try:
        path = synthesize_speech(text, lang_code, voice, prompt_type)
        print(f"speak() generated TTS: {path}")
        return path
    except Exception as e:
        print(f"speak() error: {e}")
        return None


# =========================================================
# CROATIAN DANCES
# =========================================================
# Croatia is ALZONA's other specialty, so a Croatian dance deserves the same
# answer a Filipino one gets: a sentence, and something to watch.
#
# One sentence each, written out rather than generated. These are the facts a
# visitor is told at a stand, and they should be the same every time — a model
# asked afresh will phrase it differently, and occasionally wrongly, which for
# another country's heritage is worse than dull.
#
# `video` is either a file under source/videos/ or a YouTube id. Empty means
# there is nothing to show yet and she says so rather than pretending.
#
# `start` is where the clip begins, in seconds — most recordings open with an
# announcement or an empty stage, and a visitor gets thirty seconds. Starting
# at zero would often spend all of them on a title card.
# A visitor watches a clip, not a performance. Long enough to see the dance,
# short enough that the next person is not waiting through it.
CLIP_SECONDS = 30

CROATIAN_DANCES = {
    "linđo": {
        "aliases": ("lindo", "linjo", "lindjo"),
        "text": "Linđo is the lively couples' dance of Dubrovnik and the "
                "Konavle region, led by a fiddler playing the three-stringed "
                "lijerica.",
        "text_hr": "Linđo je živahni parovni ples Dubrovnika i Konavala, koji "
                   "vodi svirač na troglasnoj lijerici.",
        "video": "videos/LINDO.mp4",
        "start": 0,
    },
    "nijemo kolo": {
        "aliases": ("nijemo", "silent circle dance", "silent kolo",
                    "silent dance"),
        "text": "Nijemo Kolo is the silent circle dance of the Dalmatian "
                "hinterland, danced with no music at all — only the dancers' "
                "steps — and UNESCO lists it as intangible cultural heritage.",
        "text_hr": "Nijemo kolo je ples Dalmatinske zagore koji se pleše bez "
                   "ikakve glazbe — čuju se samo koraci plesača — a UNESCO ga "
                   "je uvrstio u nematerijalnu kulturnu baštinu.",
        "video": "videos/NIJEMO_KOLO.mp4",
        "start": 0,
    },
    "drmeš": {
        "aliases": ("drmes", "drmesh"),
        "text": "Drmeš is a fast shaking dance from northern Croatia, danced "
                "in small tight circles or pairs with a trembling step that "
                "gives it its name.",
        "text_hr": "Drmeš je brzi ples sjeverne Hrvatske, koji se pleše u "
                   "malim zbijenim kolima ili u paru, s drhtavim korakom po "
                   "kojem je dobio ime.",
        "video": "videos/DRMES.mp4",
        "start": 0,
    },
    "lado": {
        "aliases": ("lado ensemble", "national folk dance ensemble",
                    "croatian national ensemble"),
        "text": "LADO is Croatia's national folk dance ensemble, founded in "
                "1949 to perform the dances and songs of every Croatian region "
                "in their authentic costumes.",
        "text_hr": "LADO je hrvatski nacionalni folklorni ansambl, osnovan "
                   "1949. godine, koji izvodi plesove i pjesme svih hrvatskih "
                   "krajeva u izvornim nošnjama.",
        "video": "videos/LADO.mp4",
        "start": 0,
    },
    "gorski kotar": {
        "aliases": ("gorski", "kotar"),
        "text": "The dances of Gorski Kotar come from Croatia's forested "
                "highlands between Zagreb and the sea, a region whose mountain "
                "villages kept their own steps and songs.",
        "text_hr": "Plesovi Gorskog kotara dolaze iz šumovitog gorja između "
                   "Zagreba i mora, kraja čija su planinska sela sačuvala "
                   "vlastite korake i pjesme.",
        "video": "videos/GORSKI_KOTAR.mp4",
        "start": 0,
    },
    "vrličko kolo": {
        "aliases": ("vrlicko kolo", "vrlicko", "vrličko", "vrlika kolo",
                    "vrlika"),
        "text": "The Vrličko Kolo is the circle dance of Vrlika in the "
                "Dalmatian hinterland, danced in a closed ring to the dancers' "
                "own steps rather than to instruments.",
        "text_hr": "Vrličko kolo je kolo iz Vrlike u Dalmatinskoj zagori, "
                   "koje se pleše u zatvorenom krugu uz korake samih plesača, "
                   "a ne uz glazbala.",
        "video": "videos/VRLICKO_KOLO.mp4",
        "start": 0,
    },
    "podravski svati": {
        "aliases": ("podravski", "svati", "podravina"),
        "text": "Podravski Svati are the wedding dances of Podravina, the "
                "Drava valley in northern Croatia, performed as the wedding "
                "party itself would dance them.",
        "text_hr": "Podravski svati su svadbeni plesovi Podravine, kraja uz "
                   "Dravu u sjevernoj Hrvatskoj, izvedeni onako kako bi ih "
                   "plesali sami svatovi.",
        "video": "videos/PODRAVSKI_SVATI.mp4",
        "start": 0,
    },
    "kumova grana": {
        "aliases": ("kumova", "grana"),
        "text": "Kumova Grana — “the best man's branch” — is a Croatian "
                "wedding dance named for the decorated branch carried in the "
                "wedding procession.",
        "text_hr": "Kumova grana hrvatski je svadbeni ples nazvan po ukrašenoj "
                   "grani koja se nosi u svatovskoj povorci.",
        "video": "videos/KUMOVA_GRANA.mp4",
        "start": 0,
    },
    "taraban": {
        "aliases": (),
        "text": "Taraban is a lively dance from Podravina in northern Croatia, "
                "danced at a brisk tempo to the tamburica.",
        "text_hr": "Taraban je živahan ples iz Podravine u sjevernoj "
                   "Hrvatskoj, koji se pleše u brzom ritmu uz tamburicu.",
        "video": "videos/TARABAN.mp4",
        "start": 0,
    },
    "seljančica": {
        "aliases": ("seljancica", "seljanica"),
        "text": "Seljančica — “the village girl” — is one of the best known "
                "Croatian kolos, a social circle dance danced at gatherings "
                "across the country.",
        "text_hr": "Seljančica je jedno od najpoznatijih hrvatskih kola, "
                   "društveni ples u krugu koji se pleše na okupljanjima "
                   "diljem zemlje.",
        "video": "videos/SELJANCICA.mp4",
        "start": 0,
    },
}


# Which dance to show next when none is named.
_croatian_turn = 0


def _croatian_dance(text):
    """The Croatian dance named in this text, or None.

    Longest name first: "nijemo kolo" must win over a bare "kolo", and
    "korčula sword dance" over "sword dance".
    """
    t = (text or "").lower()
    named = []
    for name, info in CROATIAN_DANCES.items():
        for key in (name,) + tuple(info["aliases"]):
            if key in t:
                named.append((len(key), name, info))
    if not named:
        return None
    named.sort(reverse=True)
    return named[0][1], named[0][2]


def _croatian_dance_reply(name, info, croatian=False):
    """Her answer: the sentence, and the video when there is one to show.

    The description is ENGLISH, always, whatever language she is currently
    speaking. That is a decision, not an oversight: these sentences name places,
    instruments and customs, and a translation of one is a new claim about
    another country's heritage that nobody at the stand can check. The English
    wording was written once and can be checked once.

    `croatian` is kept in the signature because callers pass it, and ignored.
    The Croatian text stays in the table for whenever a native speaker has read
    it and it can be turned on deliberately.
    """
    out = {"mode": "video", "reply": info["text"]}
    video = info.get("video", "")
    if not video:
        # No footage yet. Say so plainly rather than leaving a visitor waiting
        # for a video that is never going to appear.
        out["mode"] = "chat"
        return out
    start = int(info.get("start", 0))
    if video.startswith("videos/"):
        out["video_url"] = f"/media/{video}"
    else:
        # A YouTube id. The console embeds these; a plain <video> cannot play
        # a YouTube page, only a media file.
        out["video_url"] = f"youtube:{video}"
    out["video_start"] = start
    out["video_seconds"] = CLIP_SECONDS
    return out


DANCES = {
    "tinikling": "videos/tinikling.mp4",
    "cariñosa": "videos/carinosa.mp4", "carinosa": "videos/carinosa.mp4",
    "maglalatik": "videos/maglalatik.mp4", "pandanggo": "videos/pandanggo.mp4",
    "singkil": "videos/singkil.mp4",
}

# Asked which dance belongs at a Philippine festival, ALZONA features the
# Singkil — the Maranao royal dance from Lanao, from the Darangen epic.
SINGKIL_VIDEO = "videos/singkil.mp4"

# Kept up here, not written inline where it is returned, so the startup
# render below speaks the SAME string. Two copies would drift, and a drifted
# copy is a cache miss: rendered at startup, then synthesised again on the
# floor because a comma moved.
_FESTIVAL_SPOKEN = ("For a festival, watch the Singkil — the Maranao "
                    "royal dance from Lanao, from the Darangen epic, "
                    "danced between crossing bamboo poles. It is the "
                    "showpiece of Philippine festival stages.")


def _dance_spoken(name):
    """What she says about one Filipino dance — the same wording every time."""
    return f"Here is the {name.title()}, a Filipino folk dance."

# Countries whose own festivals deserve their own answer. Croatia matters most:
# it is ALZONA's other specialty, so "Croatian dance festivals" is a question
# she is expected to actually answer, not a cue to recommend a Filipino dance.
_ELSEWHERE = (
    "croatia", "croatian", "hrvatska", "japan", "japanese", "korea", "korean",
    "china", "chinese", "spain", "spanish", "indonesia", "indonesian",
    "malaysia", "malaysian", "thailand", "thai", "india", "indian", "vietnam",
    "mexico", "mexican", "hawaii", "hawaiian",
)


def wants_festival_dance(text):
    """Is this asking which dance to see or perform at a festival here?

    Deliberately a WORD-PAIR test rather than a fixed phrase list: people ask
    this a dozen ways — "dance festival", "what dance is performed at
    festivals", "anong sayaw sa pista" — and all of them mean the same thing.
    """
    t = (text or "").lower()
    festival = any(w in t for w in ("festival", "festivals", "pista", "fiesta",
                                    "pistahan", "kapistahan"))
    dancing = any(w in t for w in ("dance", "dances", "dancing", "dancers",
                                   "sayaw", "sayawan", "indak", "folk dance"))
    if not (festival and dancing):
        return False
    return not any(c in t for c in _ELSEWHERE)


def gemini_transcribe(audio_bytes, mime="audio/webm"):
    """Speech-to-text via Gemini (no Cloud Speech / ffmpeg needed)."""
    if not GEMINI_ENABLED:
        return ""
    try:
        r = gen_content(
            model="gemini-3-flash-preview",
            contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                      "Transcribe this speech exactly. Reply with ONLY the transcription."])
        return (r.text or "").strip()
    except Exception as e:
        print("transcribe error:", e)
        return ""


# TTS resilience state. A quota hit no longer disables the voice for the whole
# run — it triggers a short cooldown, after which Gemini TTS is retried
# automatically. The browser voice (front-end) covers the cooldown gap, so
# ALZONA never goes fully silent.
_TTS_COOLDOWN_SEC = 30        # back off this long after a quota (429) hit
_tts_cooldown_until = 0.0     # skip Gemini TTS until this timestamp


# Per-model quota cooldowns. A model that answers 429 is out for this long —
# long enough not to keep asking, short enough that a quota window that resets
# is noticed within a demo.
_tts_model_cooldown = {}
_TTS_MODEL_COOLDOWN_S = 120


def gemini_voice(text, cache=False):
    """ALZONA's PRIMARY voice (Gemini TTS, Leda voice). Returns a wav filename,
    or None (the front-end then uses the browser voice). Resilient:
    - cache=True replays identical phrases (wake/stop acks) instantly
    - transient failures retry across two TTS models (3.1 -> 2.5)
    - a quota hit starts a short cooldown instead of a permanent shutoff, so
      the voice recovers on its own once quota frees up."""
    global _tts_cooldown_until
    if not GEMINI_ENABLED:
        return None       # file-based mode: next voice in line, or the browser's
    if cache:
        fn = "say_" + hashlib.md5(text.encode("utf-8")).hexdigest() + ".wav"
        if os.path.exists(os.path.join(TTS_OUT, fn)):
            return fn
    else:
        fn = "novus.wav"   # single reused name — avoids piling up audio files
    # Recently rate-limited -> let the browser voice cover it (auto-recovers).
    if time.time() < _tts_cooldown_until:
        return None
    # 3.1 TTS is measurably faster than 2.5 with the same Leda voice; fall back
    # to 2.5 on a transient/model error.
    #
    # A model that answered 429 is SKIPPED for a while rather than tried again
    # every single time. Measured: 3.1 gave one clip and then 429 on every call
    # after it, so each request paid that refusal before falling through to 2.5
    # — the refusal is fast, but the retry-after is not free and the pattern
    # repeats for as long as the quota window lasts.
    for tts_model in ("gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"):
        if time.time() < _tts_model_cooldown.get(tts_model, 0):
            continue
        try:
            r = gen_content(
                model=tts_model,
                contents="Say warmly and naturally: " + text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")))))
            cand = r.candidates[0] if r.candidates else None
            parts = cand.content.parts if (cand and cand.content) else None
            pcm = parts[0].inline_data.data if (parts and parts[0].inline_data) else None
            if not pcm:
                continue          # empty audio -> try the next model
            buf = _io.BytesIO()
            with _wave.open(buf, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
                wf.writeframes(pcm)
            with open(os.path.join(TTS_OUT, fn), "wb") as f:
                f.write(buf.getvalue())
            return fn
        except Exception as e:
            # Out of quota for THIS model: stop asking it for a while.
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                _tts_model_cooldown[tts_model] = time.time() + _TTS_MODEL_COOLDOWN_S

            msg = str(e)
            print(f"tts model {tts_model} failed: {msg[:120]}")
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                _tts_cooldown_until = time.time() + _TTS_COOLDOWN_SEC
                print(f"TTS quota hit — cooling down {_TTS_COOLDOWN_SEC}s; "
                      "browser voice covers the gap, then auto-recovers.")
                return None       # don't burn another call on the 2nd model
            # non-quota error: fall through and try the next model
    return None


# =========================================================
# ELEVENLABS TTS — ALZONA's primary voice
# =========================================================
# Measured on this machine: ~0.6s warm per phrase, vs 2.5-4s for Gemini TTS.
# The first call after startup is a ~28s cold start, so _warm_up_tts() burns it
# off before the user ever asks anything.
# Master switch. Off means ALZONA uses her default Gemini "Leda" voice and never
# calls ElevenLabs at all — no key check, no warm-up, no cold start. Flip
# USE_ELEVENLABS=1 in .env to turn the ElevenLabs voice back on.
_ELEVEN_ENABLED = os.environ.get("USE_ELEVENLABS", "0").strip().lower() in (
    "1", "true", "yes", "on")
_ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
_ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
_ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5").strip()
_ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/"

# Same self-healing contract as Gemini TTS: a quota/rate hit starts a short
# cooldown instead of killing the voice for the whole run.
_eleven_cooldown_until = 0.0


def elevenlabs_voice(text, cache=False):
    """ALZONA's PRIMARY voice (ElevenLabs). Returns an mp3 filename served from
    static/tts, or None so the caller can fall through to Gemini/Cloud/browser.
    cache=True replays identical phrases (wake/stop acks) instantly."""
    global _eleven_cooldown_until
    if not _ELEVEN_ENABLED:
        return None
    if not (_ELEVEN_KEY and _ELEVEN_VOICE):
        return None
    if cache:
        fn = "el_" + hashlib.md5(text.encode("utf-8")).hexdigest() + ".mp3"
        if os.path.exists(os.path.join(TTS_OUT, fn)):
            return fn
    else:
        fn = "novus_el.mp3"   # single reused name — avoids piling up audio files
    if time.time() < _eleven_cooldown_until:
        return None
    try:
        # eleven_turbo_v2_5 is multilingual (32 languages), so the same voice
        # covers ALZONA's English/Filipino/Japanese/Korean/Chinese replies
        # without needing a per-language voice map.
        r = requests.post(
            _ELEVEN_URL + _ELEVEN_VOICE,
            headers={"xi-api-key": _ELEVEN_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": _ELEVEN_MODEL,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            timeout=60)
        if r.status_code != 200:
            msg = r.text[:160]
            print(f"ElevenLabs TTS {r.status_code}: {msg}")
            if r.status_code in (429, 401):
                _eleven_cooldown_until = time.time() + _TTS_COOLDOWN_SEC
                print(f"ElevenLabs quota/auth issue — cooling down "
                      f"{_TTS_COOLDOWN_SEC}s; Gemini voice covers the gap.")
            return None
        if not r.content:
            return None
        with open(os.path.join(TTS_OUT, fn), "wb") as f:
            f.write(r.content)
        return fn
    except Exception as e:
        print("ElevenLabs TTS failed:", str(e)[:120])
        return None


# Google Cloud TTS voices per detected language. Google Cloud synthesis is a
# single fast REST call (well under a second for short text) — much faster than
# Gemini's audio generation, which took 2.5-4s per phrase.
_TTS_VOICES = {
    "English":  ("en-US", "en-US-Neural2-F"),
    "Filipino": ("fil-PH", "fil-PH-Standard-A"),
    "Spanish":  ("es-ES", "es-ES-Neural2-A"),
    "Chinese":  ("cmn-CN", "cmn-CN-Wavenet-A"),
    # Only used by the Cloud TTS secondary. The primary Gemini voice is
    # multilingual and speaks Croatian from Croatian text without being told.
    "Croatian": ("hr-HR", "hr-HR-Standard-A"),
}


# Gemini TTS (above) is the PRIMARY voice. Google Cloud TTS is kept only as an
# optional secondary: it's sub-second, but needs a valid service-account key +
# the Cloud Text-to-Speech API enabled. It stays off unless the startup probe
# succeeds, so a rotated/expired key never blocks the Gemini voice.
_cloud_tts_ok = False


def fast_voice(text, cache=False, skip_elevenlabs=False):
    """Best available TTS, in order:
    ElevenLabs (only when USE_ELEVENLABS=1) -> Gemini Leda, ALZONA's default
    voice -> Google Cloud TTS -> None, in which case the front-end uses the
    browser voice. cache=True reuses a file for identical phrases.

    `skip_elevenlabs` goes straight to the Gemini voice. A per-CALL choice, not
    a setting: the consoles share one backend, so turning ElevenLabs off in
    .env would change the voice on all of them.

    It is NOT the faster path, whatever the name suggests. Measured on this
    machine, warm, same sentence both ways:

        ElevenLabs   1.3-2.5s
        Gemini       4.3-7.9s

    Gemini was under a second when it was the only voice in use, so this looks
    like the ElevenLabs warm-up keeping that connection hot while the Gemini
    one goes cold between calls. Whoever changes this should measure again
    rather than trust either number — including these.
    """
    # Optional: ElevenLabs, off by default (returns None immediately when the
    # USE_ELEVENLABS switch is off — see elevenlabs_voice()).
    if not skip_elevenlabs:
        fn = elevenlabs_voice(text, cache)
        if fn:
            return fn
    # Default voice: Gemini Leda (resilient — see gemini_voice()).
    fn = gemini_voice(text, cache)
    if fn:
        return fn
    # Secondary: Google Cloud TTS, only if a valid key was found at startup.
    if _cloud_tts_ok:
        try:
            out = ("say_" + hashlib.md5(text.encode("utf-8")).hexdigest() + ".mp3"
                   ) if cache else "novus.mp3"
            if cache and os.path.exists(os.path.join(TTS_OUT, out)):
                return out
            lang = detect_reply_language(text) or "English"
            lang_code, voice = _TTS_VOICES.get(lang, _TTS_VOICES["English"])
            synthesize_speech(text, lang_code, voice, out_filename=out)
            return out
        except Exception as e:
            print("Cloud TTS (secondary) failed:", str(e)[:80])
    return None       # front-end falls back to the browser voice


def _prerender_fixed_phrases():
    """Speak-ahead for every sentence whose wording never changes.

    A repeat of an already-spoken phrase comes back in 3 milliseconds against
    roughly a second to synthesise it — the cache is keyed on the text, so the
    only reason any of these is ever slow is that it happens to be the first
    time. For a stand where the same ten dances are asked about all day, that
    first time is the demo.

    Rendered in the background: the server answers questions while this runs,
    and a phrase simply stops being slow once its turn comes round. Failures
    are ignored on purpose — a phrase that could not be rendered now will be
    rendered on demand exactly as before.
    """
    phrases = [_IDENTITY, _GREETED, _IDENTITY_HR, _GREETED_HR,
               _IDENTITY_ES, _GREETED_ES, _IDENTITY_ZH, _GREETED_ZH,
               _IDENTITY_FIL, _GREETED_FIL,
               "Let me look at that coin.", _FAKE_SPOKEN,
               "I could not make out the date on this one.",
               "Hello! I'm listening. How can I help you?",
               "You're welcome! Just greet me again when you need me.",
               "Which word would you like me to write in Baybayin?"]
    # The festival answer is the slowest of the lot to synthesise — four
    # sentences, measured at 8.5 seconds cold against 1.3 warm — and it is
    # asked constantly, since "what dance is performed at a festival" is the
    # obvious question to put to her. It was never on this list.
    phrases.append(_FESTIVAL_SPOKEN)
    # dict.fromkeys: carinosa is in the table twice, spelled both ways, and
    # both spellings say the same sentence.
    for _name in dict.fromkeys(_dance_spoken(n) for n in DANCES):
        phrases.append(_name)

    # English only: the Croatian descriptions are in the table but not spoken,
    # so rendering them would be a minute of startup spent on audio nothing
    # plays.
    for info in CROATIAN_DANCES.values():
        phrases.append(info["text"])

    # Rendered in the DEFAULT voice only (ElevenLabs, Gemini if it fails) —
    # every console now uses that order. Rendering a second Gemini copy of
    # each phrase as well used up the Gemini quota at every startup, which
    # left the chat answers themselves hitting 429s.
    done = 0
    for text in phrases:
        try:
            if fast_voice(text, True):
                done += 1
        except Exception:
            pass      # rendered on demand later, as it was before
    print(f"TTS: {done}/{len(phrases)} fixed phrases ready")


def _warm_up_tts():
    """Gemini TTS is the primary voice — warm the model at startup so the first
    spoken reply isn't a cold ~6s (the cold-start happens here, off the user's
    path). Then quietly check whether Cloud TTS is usable as a secondary; an
    expected failure (rotated key / API off) is not treated as an error."""
    global _cloud_tts_ok
    # ElevenLabs has a ~28s cold start on its first request, so warm it here —
    # but only when it is actually switched on, otherwise this is wasted time
    # and a wasted API call at every boot.
    if _ELEVEN_ENABLED:
        try:
            if elevenlabs_voice("Hello", cache=True):
                print("ElevenLabs TTS ready — primary voice warmed up.")
            else:
                print("ElevenLabs warm-up returned no audio (falling back to Gemini).")
        except Exception as e:
            print("ElevenLabs warm-up failed:", str(e)[:100])
    else:
        print("ElevenLabs disabled (USE_ELEVENLABS=0) — using the default Gemini voice.")
    try:
        if not GEMINI_ENABLED:
            print("Gemini TTS off (file-based mode).")
        elif gemini_voice("Hello", cache=True):
            print("Gemini TTS ready — primary voice warmed up.")
        else:
            print("Gemini TTS warm-up returned no audio (will retry on demand).")
    except Exception as e:
        print("Gemini TTS warm-up failed:", str(e)[:100])
    try:
        synthesize_speech("hi", "en-US", "en-US-Neural2-F", out_filename="_probe.mp3")
        _cloud_tts_ok = True
        print("Cloud TTS also available — will use it as a secondary voice.")
    except Exception:
        _cloud_tts_ok = False
        print("Cloud TTS secondary unavailable (using Gemini only) — that's fine.")


_warm_up_tts()

def _bay_path(key):
    p = baybayin_path.get(key)
    return os.path.join(BASE, p.lstrip("./").replace("/", os.sep)) if p else None


# =========================================================
# LUPANG HINIRANG — SING-BACK & HARMONY
# =========================================================
# Two modes, both driven from a spoken command:
#   imitate   - the user sings a line, ALZONA sings the same line back at the
#               same pitches they used
#   harmonize - "harmonize with me in soprano/alto/tenor/bass": ALZONA sings
#               that SATB part against the user's melody, 4/4, anchored to G4
# Pitch tracking and synthesis run in the browser (Web Audio) so there is no
# upload latency; the backend just arms the mode.
SATB_PARTS = ("soprano", "alto", "tenor", "bass")

# How each part sits against the melody, in semitones. The melody is normally
# the soprano line, so "soprano" doubles it and the lower parts sit below at
# consonant intervals (a sixth, an octave-and-a-third, and two octaves down).
SATB_OFFSETS = {"soprano": 0, "alto": -5, "tenor": -12, "bass": -24}

# The user always starts on G4 (they confirmed this), so that is the reference
# ALZONA transposes from if it needs a key before hearing a note.
SING_REFERENCE_HZ = 392.00      # G4
SING_REFERENCE_NOTE = "G4"


def parse_sing_parts(text):
    """Pull every SATB part named in a command, in SATB order.

    Handles one part ("in soprano"), several ("tenor and bass"), and the
    all-parts shorthands. Returns [] when no part is named.
    """
    t = (text or "").lower()
    # "all parts" / "everyone" / "full choir" -> the whole ensemble.
    if any(w in t for w in ("all parts", "all part", "everyone", "everybody",
                            "full choir", "whole choir", "all voices",
                            "all of them", "lahat", "buong choir", "satb")):
        return list(SATB_PARTS)
    found = [p for p in SATB_PARTS if p in t]
    # "base" is how the user says (and spells) bass.
    if "bass" not in found and re.search(r"\bbase\b", t):
        found.append("bass")
    # Keep canonical SATB order regardless of the order they were spoken.
    return [p for p in SATB_PARTS if p in found]


# The songs the harmony panel offers, read from source/harmony/songs.json so
# that adding one is a data change rather than a code change.
#
# Loaded once at startup. A song listed with ready false is known by name but
# has no recordings yet: she says so instead of starting something silent.
def _load_songs():
    path = os.path.join(BASE, "source", "harmony", "songs.json")
    try:
        with open(path, encoding="utf-8") as f:
            songs = json.load(f).get("songs", [])
    except Exception as e:
        print("Harmony songs.json unreadable:", str(e)[:120])
        return []
    for sg in songs:
        # Longest first, so "lupang hinirang" wins over the "lupang" that
        # sits inside it and a two-word name is never half-matched.
        sg["aliases"] = sorted(
            {a.lower() for a in sg.get("aliases", [])} | {sg["title"].lower()},
            key=len, reverse=True)
    return songs


HARMONY_SONGS = _load_songs()
print("Harmony songs: " + ", ".join(
    "%s%s" % (sg["title"], "" if sg.get("ready") else " (no recordings yet)")
    for sg in HARMONY_SONGS) if HARMONY_SONGS else "Harmony songs: none listed")


def detect_sing_song(text):
    """Which song was asked for, or None if the command did not name one."""
    t = (text or "").lower()
    for sg in HARMONY_SONGS:
        for alias in sg["aliases"]:
            if alias in t:
                return sg
    return None


def detect_sing_command(text):
    """Recognise a Lupang Hinirang singing command. Returns a dict the frontend
    uses to arm its listener, or None."""
    t = (text or "").lower()
    anthem = any(w in t for w in ("lupang hinirang", "lupang", "hinirang",
                                  "national anthem", "pambansang awit"))
    harmonize = any(w in t for w in ("harmonize", "harmony", "harmonise",
                                     "sabayan", "boses"))
    imitate = any(w in t for w in ("sing back", "imitate", "copy", "repeat after",
                                   "follow me", "gayahin", "ulitin", "sing with",
                                   "sing"))
    if harmonize:
        parts = parse_sing_parts(text) or ["alto"]

        # "alzona harmonize with me in ama namin in alto" — one command, both
        # the song and the part. Naming no song keeps the old behaviour and
        # takes the first one that has recordings, which is what every
        # command meant before there was more than one song.
        song = detect_sing_song(text)
        if song is None:
            song = next((sg for sg in HARMONY_SONGS if sg.get("ready")), None)

        return {"mode": "harmonize", "parts": parts,
                "song": song["id"] if song else None,
                "song_title": song["title"] if song else None,
                "song_ready": bool(song and song.get("ready")),
                # Kept for the synth fallback when a recording is missing.
                "part": parts[0], "offset": SATB_OFFSETS[parts[0]],
                "reference_note": SING_REFERENCE_NOTE,
                "reference_hz": SING_REFERENCE_HZ,
                "beats_per_bar": 4}
    if anthem or imitate:
        return {"mode": "imitate", "parts": [], "part": None, "offset": 0,
                "reference_note": SING_REFERENCE_NOTE,
                "reference_hz": SING_REFERENCE_HZ,
                "beats_per_bar": 4}
    return None


# =========================================================
# COIN IDENTIFICATION (camera -> Gemini vision)
# =========================================================
# The five fields ALZONA reports for a coin held up to the camera, in order.
# Each is capped at ONE sentence — enforced in the prompt AND trimmed after,
# because a vision model will happily write a paragraph about a coin.
# Order matters and is deliberate: authenticity is settled FIRST, because every
# field after it is only worth reading if the coin is genuine.
COIN_ORDER = [
    ("authenticity",    "Real or Fake",                       "\U0001F50E"),
    ("other_countries", "Used Elsewhere",                     "\U0001F30D"),
    ("country",         "Nationality / Country",              "\U0001F4D6"),
    ("denomination",    "Currency & Denomination",            "\U0001F4B0"),
    ("featured",        "Person / Symbol Featured",           "\U0001F464"),
    ("significance",    "Historical & Cultural Significance", "\U0001F3DB"),
]

# Not reported at all once a counterfeit is spotted. Every one of these
# describes the REAL coin being imitated, not the object in front of the
# camera, so answering them would dress a fake up in a genuine coin's history.
# The country stays: which coin it is pretending to be is worth knowing.
_SKIP_IF_FAKE = ("other_countries", "denomination", "featured", "significance")

# What she SAYS about a counterfeit, and all she says.
#
# Fixed, and deliberately without detail. Everything interesting about a coin —
# when it was struck, what the mint mark means, what the design commemorates —
# belongs to the real coin this one is imitating, so saying any of it out loud
# describes something the person is not holding. The panel still shows what
# gave it away; her voice does not elaborate on a forgery.
#
# Being fixed also makes it instant: it is rendered with the other set phrases
# at startup, so the answer lands immediately rather than after a synthesis.
_FAKE_SPOKEN = "That one's a fake."


# The fixed sentences, rendered behind the server rather than in front of it.
# Nothing waits for this: it makes phrases faster as it goes, and a question
# asked before it reaches one is answered at the usual speed.
#
# Started down here rather than beside _warm_up_tts, which is where it was:
# the list it renders includes _FAKE_SPOKEN, and up there that name does not
# exist yet. The thread died on a NameError at every single startup, so the
# phrases it promises to have ready were never rendered and every one of them
# was synthesised on demand.
#
# PRERENDER_PHRASES=0 skips it. A cloud host with a throwaway disk (Render's
# free tier wipes it on every restart and every wake from sleep) would pay
# for all of these again each time; there they are rendered on first use.
if os.environ.get("PRERENDER_PHRASES", "1").strip().lower() not in (
        "0", "false", "no", "off"):
    threading.Thread(target=_prerender_fixed_phrases, daemon=True).start()


_COIN_PROMPT = """You are identifying a coin held up to a camera.

Decide FIRST whether the coin is genuine, then report the rest.

1. verdict - exactly one word: real, fake, or unclear. Use "unclear" whenever
   the image is not good enough to judge; a confident guess is worse than
   admitting the picture will not support one.
2. authenticity - what led you to that verdict: strike quality, lettering,
   edge, colour, wear. If the verdict is "fake", say WHAT gives it away and
   allow yourself one light, good-natured joke about it - amused, never
   sneering, and never at the expense of the person holding it.
3. other_countries - whether this same coin, or the same design or currency, is
   or was used in any other country. Say so plainly if it is used only here.
4. country - the nation that issued it
5. denomination - the currency and face value
6. featured - the person, animal, or symbol shown on it
7. significance - its historical and cultural significance
8. spoken - what ALZONA says OUT LOUD, which is NOT any of the above.

Fields 1-7 are printed on a screen the person is already looking at, so
reading them aloud tells them nothing they cannot see. `spoken` is the part
they can only get by asking — FIVE sentences, told the way you would tell
someone who has just handed you the coin and is curious about it.

- FIRST, when it was struck: the year on the coin, and the mint if it is
  marked. Say plainly that the date is not legible if it is not — a made-up
  year on a real coin is worse than admitting the picture is poor.
- THEN four more sentences of things genuinely worth knowing that fields 1-7
  do not already say. Reach for the specific over the general: why the design
  changed that year, what the mint mark stands for, what the coin would have
  bought at the time, how long the series ran and what replaced it, an
  engraving detail most people miss, what was happening in the country when it
  was issued, how the metal or size differs from the coin before it.
- Be genuinely interesting, not merely enthusiastic. One concrete fact is worth
  more than three sentences of "fascinating" and "remarkable". Never pad to
  reach five — if there are only three real things to say, say three. A short
  true answer is better than a long one padded out to length.
- Never repeat the country, the denomination, the person shown, or whether it
  is genuine. Those are already on the screen.
- Say what you are unsure of as unsure. A visitor who is told something
  confident and wrong about their own currency will know.
- If the verdict is "fake", leave `spoken` empty. She says one fixed sentence
  for a counterfeit and no detail: everything interesting about a coin belongs
  to the real one this is imitating, and saying it aloud would describe
  something the person is not holding.

Rules:
- Reply with ONE SENTENCE per field for fields 1-7. Never more than one.
- Keep each of those sentences under 20 words, natural and warm, not a bare
  label.
- `spoken` is up to FIVE sentences and about 120 words. It is the one field
  that is allowed to be long, because it is the only one nobody can read off
  the screen.
- If a detail is genuinely not visible (worn, blurred, face-down), say so in
  that field's sentence instead of guessing.
- If the image contains NO coin at all, reply with exactly: NO_COIN
- Return ONLY a JSON object with the keys: verdict, authenticity,
  other_countries, country, denomination, featured, significance, spoken.
  No markdown, no code fence."""


def _first_sentence(text, max_words=24, sentences=1):
    """Trim a coin field to `sentences` sentences - the model is asked for one,
    and this holds it to that after the fact. A fake gets two: the giveaway,
    then the joke about it."""
    t = " ".join(str(text or "").split())
    t = re.sub(r"[*#_`]+", "", t)
    parts = [p for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    out = " ".join(parts[:sentences]) if parts else t
    words = out.split()
    if len(words) > max_words * sentences:
        out = " ".join(words[:max_words * sentences]).rstrip(",;:—- ")
        if out and out[-1] not in ".!?":
            out += "."
    return out.strip()


def identify_coin(jpeg_bytes):
    """Send a camera frame to Gemini vision and return the five coin fields.
    Returns {"ok": False, "error": ...} when no coin is visible."""
    if not GEMINI_ENABLED:
        return {"ok": False,
                "error": "Coin reading needs Gemini, which is turned off right now."}
    try:
        r = gen_content(
            model=CHAT_MODEL,
            contents=[types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                      _COIN_PROMPT],
            config=GEN_CONFIG_FAST)
        raw = (r.text or "").strip()
        if not raw or "NO_COIN" in raw.upper():
            return {"ok": False, "error": "I don't see a coin — hold one up to the camera."}
        # Strip a ```json fence if the model added one despite being asked not to.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return {"ok": False, "error": "I couldn't read that coin clearly."}
            data = json.loads(m.group(0))
        # One word the code can branch on, rather than re-reading the prose.
        raw_verdict = str(data.get("verdict", "")).strip().lower()
        verdict = next((v for v in ("fake", "real", "unclear") if v in raw_verdict),
                       "unclear")
        fields = []
        for key, label, emoji in COIN_ORDER:
            if verdict == "fake" and key in _SKIP_IF_FAKE:
                continue
            # The one-sentence rule holds everywhere except the verdict on a
            # fake, which has to carry both the giveaway and the joke about it.
            room = 2 if (verdict == "fake" and key == "authenticity") else 1
            fields.append({"key": key, "label": label, "emoji": emoji,
                           "text": _first_sentence(data.get(key, ""),
                                                   sentences=room)})
        # What she SAYS, which is deliberately not what the panel shows.
        #
        # Reading the six rows aloud told the person in front of her exactly
        # what was already on the screen they were looking at. The date it was
        # struck and a fact worth knowing are the parts they can only get by
        # asking her.
        if verdict == "fake":
            spoken = _FAKE_SPOKEN
        else:
            # Five sentences, ~120 words. The trim is a backstop against a
            # model that ignores the brief, not the intended length.
            spoken = _first_sentence(data.get("spoken", ""), max_words=24,
                                     sentences=5)
        if not spoken:
            # No spoken line came back. The panel is still right, so say the
            # one thing that is not on it rather than nothing at all.
            spoken = "I could not make out the date on this one."
        return {"ok": True, "verdict": verdict,
                "fields": fields, "spoken": spoken}
    except Exception as e:
        print("coin id error:", str(e)[:120])
        return {"ok": False, "error": "I had trouble identifying that coin."}


# =========================================================
# PHYSICAL PRINTING (Baybayin sheets)
# =========================================================
# Auto-print is on by default; set BAYBAYIN_AUTOPRINT=0 in .env to turn it off
# (the on-screen display is unaffected either way).
_AUTOPRINT = os.environ.get("BAYBAYIN_AUTOPRINT", "1").strip().lower() not in (
    "0", "false", "no", "off")
_PRINTER = os.environ.get("BAYBAYIN_PRINTER", "").strip()   # blank = Windows default

# The GOOJPRT PT-210 the robot carries. It is preferred over a desk printer
# when present, and simply absent otherwise — see thermal.py for why it needs
# no driver. BAYBAYIN_THERMAL=0 in .env forces the desk printer instead.
try:
    import thermal
    _THERMAL_AVAILABLE = True
except Exception as _e:
    thermal = None
    _THERMAL_AVAILABLE = False
    print("Thermal printing unavailable:", _e)

_THERMAL_FIRST = os.environ.get("BAYBAYIN_THERMAL", "1").strip().lower() not in (
    "0", "false", "no", "off")

# Baybayin prints on the thermal printer or not at all.
#
# It used to fall back to whatever Windows had as its default, which sounds
# helpful and is not: the robot carries the thermal printer, a visitor takes
# the strip away with them, and an A4 sheet from an office printer in another
# room is no use to anybody. Worse, the fallback was silent - the EPSON sat
# marked "Use Printer Offline" and four sheets queued up over sixteen minutes
# while the log reported each one printed.
#
# Set BAYBAYIN_ALLOW_FALLBACK=1 in .env to let it use an ordinary printer
# again, for a demo with no thermal printer to hand.
_ALLOW_FALLBACK = os.environ.get(
    "BAYBAYIN_ALLOW_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on")

# The coin she was asked about, filled in when the vision model answers.
#
# Reading a coin takes about four and a half seconds — the model's own latency,
# not the image, which is one tile. Held synchronously that is four and a half
# seconds of a robot standing mute in front of a visitor who just asked it a
# question. She acknowledges immediately instead and the answer arrives here,
# picked up by the /state poll the console already runs every second.
# `asked_by` is the console that asked. Every console polls /state, so without
# it they ALL saw a new coin answer and ALL spoke it — three consoles running
# meant three voices reading the same thing over each other.
last_coin_result = {"seq": 0, "fields": None, "spoken": "", "error": "",
                    "asked_by": ""}
_coin_busy = False


def _coin_in_background(asked_by=""):
    """Read the coin off the current frame, then leave the answer on /state."""
    global last_coin_result, _coin_busy
    try:
        result = identify_coin(latest_frame)
        if result.get("ok"):
            last_coin_result = {
                "seq": last_coin_result["seq"] + 1,
                "fields": result["fields"],
                "spoken": result.get("spoken", ""),
                "error": "",
                "asked_by": asked_by,
            }
        else:
            last_coin_result = {
                "seq": last_coin_result["seq"] + 1,
                "fields": None, "spoken": "",
                "error": result.get("error", "I couldn't read that coin."),
                "asked_by": asked_by,
            }
    except Exception as e:
        last_coin_result = {"seq": last_coin_result["seq"] + 1, "fields": None,
                            "spoken": "", "error": str(e)[:120],
                            "asked_by": asked_by}
    finally:
        _coin_busy = False


# What the last print attempt did, surfaced on /state so the UI can show it.
last_print_status = {"word": None, "ok": None, "detail": "", "time": 0}


def _print_image_sync(path, printer_name):
    """Render an image straight to the printer's device context and spool it.

    Deliberately NOT os.startfile(path, "print"): this machine has no mspaint
    and no "print" verb registered for .png, so the shell route fails outright —
    and where it does exist it pops the Photos print dialog, which would stall a
    live demo waiting for a human click. Going through the DC prints silently.
    The image is scaled to fit the page while keeping its aspect ratio.
    """
    hDC = win32ui.CreateDC()
    hDC.CreatePrinterDC(printer_name)
    try:
        # Printable area of the page, in device units (dots).
        page_w = hDC.GetDeviceCaps(HORZRES)
        page_h = hDC.GetDeviceCaps(VERTRES)

        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_w, img_h = img.size

        # Fit to the page without distorting or cropping the characters.
        scale = min(page_w / img_w, page_h / img_h)
        draw_w, draw_h = int(img_w * scale), int(img_h * scale)
        x = (page_w - draw_w) // 2
        y = (page_h - draw_h) // 2

        hDC.StartDoc(f"ALZONA Baybayin - {os.path.basename(path)}")
        hDC.StartPage()
        ImageWin.Dib(img).draw(hDC.GetHandleOutput(), (x, y, x + draw_w, y + draw_h))
        hDC.EndPage()
        hDC.EndDoc()
    finally:
        hDC.DeleteDC()


# Windows accepts a job for a printer that is switched off, out of paper, or
# marked "Use Printer Offline", and holds it in the queue indefinitely. Nothing
# fails, so printing reported success and no paper appeared - four Baybayin
# sheets sat unprinted in a queue while the log said each one had printed.
_PRINTER_TROUBLE = (
    (0x00000400, "set to Use Printer Offline in Windows"),   # attribute
)
_PRINTER_STATUS_TROUBLE = (
    (0x00000080, "offline"),
    (0x00001000, "not available"),
    (0x00000010, "out of paper"),
    (0x00000008, "jammed"),
    (0x00400000, "a door is open"),
    (0x00100000, "waiting for someone at the printer"),
    (0x00000002, "reporting an error"),
    (0x00000001, "paused"),
)


def _printer_trouble(name):
    """Why this printer would not print, or None if it looks ready."""
    try:
        h = win32print.OpenPrinter(name)
        try:
            info = win32print.GetPrinter(h, 2)
        finally:
            win32print.ClosePrinter(h)
    except Exception:
        return None          # cannot tell; let the print attempt decide

    for bit, why in _PRINTER_TROUBLE:
        if info.get("Attributes", 0) & bit:
            return why
    for bit, why in _PRINTER_STATUS_TROUBLE:
        if info.get("Status", 0) & bit:
            return why
    return None


def print_image(path, label="", glyphs=None):
    """Send an image to the printer, off the request thread so a busy or offline
    printer never stalls ALZONA's reply.

    The thermal printer is the only one used: it is the one the robot carries,
    it needs no driver, and it prints a strip a visitor can take away rather
    than a sheet of A4. If it is not plugged in, nothing is printed and the
    reason is said plainly - see _ALLOW_FALLBACK for the way back to an
    ordinary printer."""

    def _job():
        global last_print_status

        # ---- the thermal printer, if it is there ----
        if _THERMAL_AVAILABLE and _THERMAL_FIRST:
            ok, detail = thermal.print_image(path, label=label, glyphs=glyphs)
            if ok:
                last_print_status = {"word": label, "ok": True,
                                     "detail": f"thermal — {detail}",
                                     "time": time.time()}
                print(f"Baybayin printed on the thermal printer: {label}")
                return
            last_print_status = {"word": label, "ok": False,
                                 "detail": f"thermal printer — {detail}",
                                 "time": time.time()}
            print("Not printed —", detail)

            if not _ALLOW_FALLBACK:
                return

            print("Falling back to an ordinary printer "
                  "(BAYBAYIN_ALLOW_FALLBACK is set).")

        # ---- anything else Windows knows about ----
        if not _PRINTING_AVAILABLE:
            last_print_status = {"word": label, "ok": False,
                                 "detail": "no printer available",
                                 "time": time.time()}
            print("Print skipped — no thermal printer and no Windows printing.")
            return

        printer = _PRINTER or win32print.GetDefaultPrinter()

        # Say so instead of queueing into the dark.
        trouble = _printer_trouble(printer)
        if trouble:
            last_print_status = {"word": label, "ok": False,
                                 "detail": f"{printer} is {trouble}",
                                 "time": time.time()}
            print(f"Not printed — {printer} is {trouble}. "
                  f"The thermal printer was not plugged in either.")
            return

        try:
            _print_image_sync(path, printer)
            last_print_status = {"word": label, "ok": True, "detail": printer,
                                 "time": time.time()}
            print(f"Baybayin sheet printed on {printer}: {label}")
        except Exception as e:
            last_print_status = {"word": label, "ok": False,
                                 "detail": str(e)[:120], "time": time.time()}
            print("Print failed:", str(e)[:120])
    threading.Thread(target=_job, daemon=True).start()


def make_baybayin_image(word):
    images, H, sp = [], 300, 50
    # Kept alongside the composed sheet: the thermal printer lays the word out
    # itself, from the glyphs, rather than shrinking a sheet drawn for A4.
    glyphs = []
    for s in split_syllables(word):
        ap = _bay_path(s)
        if not ap or not os.path.exists(ap):
            continue
        img = cv2.imread(ap)
        if img is None:
            continue
        glyphs.append((ap, s.upper()))
        images.append(cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H)))
    if not images:
        return None
    space = np.ones((H, sp, 3), np.uint8) * 255
    parts = []
    for i, im in enumerate(images):
        parts.append(im)
        if i != len(images) - 1:
            parts.append(space)
    result = np.hstack(parts)
    m = 40
    canvas = np.ones((H + 2 * m, result.shape[1] + 2 * m, 3), np.uint8) * 255
    canvas[m:m + H, m:m + result.shape[1]] = result
    fn = "novus.png"   # single reused name — avoids piling up images
    out_path = os.path.join(GEN_OUT, fn)
    cv2.imwrite(out_path, canvas)
    # Auto-print the sheet as soon as it is generated (set BAYBAYIN_AUTOPRINT=0
    # in .env to keep the on-screen display without using paper).
    if _AUTOPRINT:
        print_image(out_path, label=word, glyphs=glyphs)
    return f"/gen/{fn}"


# Ways her name comes back from speech recognition. Addressing her is not part
# of the word to write: "Alzona, translate pilipinas to baybayin" was rendering
# "alzona pilipinas" in glyphs, which is a different word and a wasted sheet of
# paper once it prints.
_BAYBAYIN_ADDRESS = {"alzona", "alsona", "elzona", "arizona", "alona", "al", "zona"}

# Greetings and politeness that can sit around the request without being in it.
_BAYBAYIN_FILLER = {"hey", "hi", "hello", "heya", "yo", "okay", "ok", "po", "na",
                    "ako", "mo", "ng", "ang", "sa", "para", "pwede", "puwede",
                    "paki", "pakisulat", "isulat", "sulat", "salin", "isalin"}


# How "Baybayin" actually comes back from speech recognition. Measured live:
# "Can you translate Filipinas to be buying?" — the script's name is not in any
# dictionary the recogniser has, so it reaches for English words that sound
# like it. Matching only the correct spelling meant the feature worked when
# typed and never when spoken.
_BAYBAYIN_HEARD = re.compile(
    r"\b(?:baybayin|baybayan|bay\s?bay(?:in|an)?|"
    r"b(?:e|ee|y|ye|uy|ay)\s*buying|by\s*by(?:ing|in)|"
    r"bay\s*bay|babayin|baibayin|bay\s?been|buy\s?buying)\b",
    re.I,
)


def wants_baybayin(text):
    """Did they ask for Baybayin, however the recogniser spelled it?"""
    return bool(_BAYBAYIN_HEARD.search(text or ""))


def extract_baybayin_target(text):
    t = text.lower()
    for sym in [",", ".", "?", "!", ":", '"', "'"]:
        t = t.replace(sym, "")
    # Strip the ways the recogniser spells "Baybayin" too, or "be buying" ends
    # up rendered as glyphs alongside the word that was actually asked for.
    t = _BAYBAYIN_HEARD.sub(" ", t)
    stop = {"translate", "to", "in", "into", "the", "me", "show", "what", "is", "how",
            "do", "you", "write", "baybayin", "baybay", "please", "say", "word", "of",
            "a", "an", "can", "spell", "convert", "give", "see", "my", "name"}
    stop = stop | _BAYBAYIN_ADDRESS | _BAYBAYIN_FILLER

    words = [w for w in t.split() if w not in stop]

    # Everything was addressing or politeness — there is no word to write. Keep
    # the name rather than print a blank sheet, so the caller can ask which word.
    if not words:
        return ""
    return " ".join(words).strip()


# =========================================================
# KNOWLEDGE LIBRARY
# Drop files in knowledge/ (or upload via /upload_knowledge) and ALZONA
# searches them for answers before replying.
# =========================================================
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

_knowledge_chunks = []   # [{"source": filename, "text": chunk}]

KNOWLEDGE_EXTS = (".txt", ".md", ".csv", ".pdf", ".docx", ".jsonl", ".json")


def _extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md", ".csv"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        if ext == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
    except Exception as e:
        print(f"knowledge: failed to read {os.path.basename(path)}: {e}")
    return ""


def _chunk_text(text, size=1200, overlap=200):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def _record_to_text(obj):
    """Flatten a JSON record into readable 'field: value' lines."""
    if isinstance(obj, dict):
        return "\n".join(f"{k}: {_record_to_text(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        return "; ".join(_record_to_text(v) for v in obj)
    return str(obj)


def _json_chunks(path):
    """One chunk per record, so each province/region stays a separate,
    self-contained unit for retrieval (instead of blind text slicing)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
    except Exception as e:
        print(f"knowledge: failed to read {os.path.basename(path)}: {e}")
        return []

    records = []
    if content.startswith("["):          # plain .json array
        try:
            data = json.loads(content)
            records = data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"knowledge: bad json in {os.path.basename(path)}: {e}")
    else:                                # .jsonl — one record per line
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                print(f"knowledge: skipped a bad jsonl line in {os.path.basename(path)}")

    chunks = []
    for rec in records:
        text = " ".join(_record_to_text(rec).split())
        if not text:
            continue
        # keep records whole; only split the rare oversized one
        chunks.extend(_chunk_text(text) if len(text) > 2000 else [text])
    return chunks


def reload_knowledge():
    """(Re)index every readable file in the knowledge folder."""
    global _knowledge_chunks
    chunks = []
    files = [f for f in sorted(os.listdir(KNOWLEDGE_DIR))
             if f.lower().endswith(KNOWLEDGE_EXTS)]
    for fn in files:
        path = os.path.join(KNOWLEDGE_DIR, fn)
        if fn.lower().endswith((".jsonl", ".json")):
            file_chunks = _json_chunks(path)
        else:
            file_chunks = _chunk_text(_extract_text(path))
        for c in file_chunks:
            if c.strip():
                chunks.append({"source": fn, "text": c})
    _knowledge_chunks = chunks
    print(f"knowledge: indexed {len(chunks)} chunk(s) from {len(files)} file(s)")


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "when", "where",
    "why", "how", "do", "does", "did", "can", "could", "you", "your", "me", "my",
    "i", "of", "in", "on", "at", "to", "for", "and", "or", "it", "its", "this",
    "that", "tell", "about", "please", "alzona",
}


def retrieve_knowledge(query, top_k=3, max_chars=4000):
    """Return the knowledge chunks most relevant to the query (keyword scoring)."""
    if not _knowledge_chunks:
        return ""
    terms = [w for w in re.findall(r"[a-z0-9']+", query.lower())
             if len(w) > 2 and w not in _STOPWORDS]
    if not terms:
        return ""
    scored = []
    for chunk in _knowledge_chunks:
        low = chunk["text"].lower()
        distinct = sum(1 for t in terms if t in low)
        if distinct:
            hits = sum(low.count(t) for t in terms)
            scored.append((distinct * 3 + hits, chunk))
    if not scored:
        return ""
    scored.sort(key=lambda s: s[0], reverse=True)
    out, total = [], 0
    for _, chunk in scored[:top_k]:
        piece = f"[from {chunk['source']}]\n{chunk['text']}"
        if total + len(piece) > max_chars:
            break
        out.append(piece)
        total += len(piece)
    return "\n\n".join(out)


reload_knowledge()


# Function words used to tell English from Filipino/Tagalog (both use the Latin
# alphabet, so script detection alone can't separate them \u2014 a Filipino topic or
# knowledge base easily pulls an English question into a Tagalog reply).
_EN_MARKERS = {
    "the", "is", "are", "was", "were", "what", "who", "when", "where", "why",
    "how", "did", "do", "does", "a", "an", "of", "in", "on", "to", "for", "and",
    "or", "it", "he", "she", "they", "his", "her", "their", "you", "your",
    "this", "that", "which", "there", "here", "with", "about", "can", "could",
    "would", "should", "tell", "me", "my", "please", "give", "name",
}
# Croatian function words. Croatia is ALZONA's other specialty and the reason
# she exists at a WRO event held there, so being addressed in Croatian and
# answering in English is the one language failure that matters most here.
#
# Chosen to be unmistakable: every one of these is common in ordinary Croatian
# speech and none is an English or Filipino word. "sam", "smo" and "su" are
# left out deliberately — "sam" is English "Sam" and the cost of a false Croatian
# reading is a reply in a language the visitor may not read at all.
_HR_MARKERS = {
    "što", "sto", "tko", "kako", "gdje", "kada", "zašto", "zasto", "koji",
    "koja", "koje", "jesi", "jeste", "možeš", "mozes", "molim", "hvala",
    "dobar", "dobro", "jutro", "večer", "vecer", "bok", "zdravo", "ime",
    "zovem", "zoveš", "zoves", "hrvatska", "hrvatski", "hrvatskoj", "ples",
    "plesovi", "pjesma", "narodni", "ovo", "ona", "oni", "nije", "jest",
    "ili", "ali", "sada", "puno", "malo", "vrlo", "također", "takoder",
    "je", "ti", "nam", "vam", "nas", "vas", "lijepa", "lijep", "zemlja",
    "znaš", "znas", "reci", "pokaži", "pokazi", "pjevaj", "hrvat",
    "povijest", "povijesti", "kultura", "kulturi", "molim", "izvoli",
}
# Deliberately NOT here: "si" is the Filipino marker before a name, and "sam"
# is an English one. A wrong Croatian reading answers a visitor in a language
# they may not read at all, which is worse than answering a Croat in English.

# Spanish function words. Kept clear of Filipino, which borrowed heavily from
# Spanish: "para", "pero", "porque", "como", "kasi" and the numbers all appear
# in ordinary Tagalog, so none of them is here. What is left is the grammar
# Filipino did NOT take — articles, pronouns, and the verb forms.
_ES_MARKERS = {
    "qué", "que", "quién", "quien", "cómo", "como", "dónde", "donde",
    "cuándo", "cuando", "por", "favor", "gracias", "hola", "buenos", "buenas",
    "días", "dias", "tardes", "noches", "eres", "está", "esta", "estás",
    "estas", "soy", "son", "somos", "tiene", "tienes", "tengo", "hace",
    "hacer", "puede", "puedes", "quiero", "quieres", "dime", "dígame",
    "digame", "una", "unos", "unas", "los", "las", "del", "muy", "más",
    "mas", "también", "tambien", "español", "espanol", "baile", "bailes",
    "cultura", "historia", "gracias",
}

_FIL_MARKERS = {
    "ang", "ng", "mga", "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
    "ito", "iyan", "iyon", "ano", "sino", "saan", "kailan", "bakit", "paano",
    "kung", "ay", "po", "opo", "hindi", "oo", "salamat", "kumusta", "kamusta",
    "naman", "lang", "yung", "niya", "nila", "natin", "namin", "niyo", "mo",
    "ko", "pinakamataas", "bayani", "kabisera", "nobela", "isinulat",
}


def detect_reply_language(text):
    """Pin the reply language for cases the model tends to drift away from.
    Kana is checked before Han because Japanese also uses Han characters."""
    if any('\u4e00' <= c <= '\u9fff' for c in text):   # han -> Mandarin
        return "Chinese"
    # Latin script: separate English from Filipino by function words. English
    # markers never appear in Filipino/regional languages, so an English hit is
    # a strong, safe signal \u2014 this keeps English questions from drifting to
    # Tagalog. Romanized CJK (no markers either way) returns None and is handled
    # by the model instruction instead.
    # Croatian diacritics settle it on their own — no English or Filipino word
    # carries them, so one is proof enough.
    if any(c in text.lower() for c in "čćđšž"):
        return "Croatian"

    # ñ and the accented vowels are Spanish and not Croatian or Filipino.
    if any(c in text.lower() for c in "ñ¿¡"):
        return "Spanish"

    words = re.findall(r"[a-zčćđšžáéíóúüñ]+", text.lower())
    if words:
        en = sum(w in _EN_MARKERS for w in words)
        fil = sum(w in _FIL_MARKERS for w in words)
        hr = sum(w in _HR_MARKERS for w in words)
        es = sum(w in _ES_MARKERS for w in words)

        # Spanish before English: several Spanish markers are spelled like
        # English ones ("son", "una"), so it has to win on its own count
        # rather than be crowded out by an incidental English word.
        if es >= 2 and es > en and es > fil and es > hr:
            return "Spanish"
        if es and not en and not fil and not hr:
            return "Spanish"
        # Croatian first when it clearly leads: someone writing without
        # diacritics ("sto je ovo") still deserves a Croatian answer.
        # One marker is enough when nothing English or Filipino is present.
        # Croatian sentences are often short — "Tko si ti?", "Hrvatska je
        # lijepa" — and requiring two of them missed exactly those.
        if hr and not en and not fil and not es:
            return "Croatian"
        if hr >= 2 and hr > en and hr > fil and hr > es:
            return "Croatian"
        if en >= 1 and en > fil and en > hr and en > es:
            return "English"
        # Same courtesy as Croatian above: one unmistakable Filipino word with
        # nothing English or Croatian beside it is Filipino. "sino ka" — who
        # are you — carries exactly one and was falling through to no language
        # at all, which then answered a Filipino visitor in English.
        if fil and not en and not hr and not es:
            return "Filipino"
        if fil >= 2 and fil > en and fil > hr and fil > es:
            return "Filipino"
    return None


# Trailing function words to strip if the word cap chops mid-phrase, so a
# truncated reply still ends on a content word rather than "…cultural, and".
_TRAIL_DROP = {
    "and", "or", "yet", "but", "so", "the", "a", "an", "of", "for", "to",
    "with", "as", "its", "his", "her", "their", "in", "on", "at", "by", "from",
    "that", "which", "is", "was", "were", "are", "who", "whose",
}


_ABBREV = ("Dr", "Mr", "Mrs", "Ms", "Jr", "Sr", "St", "Mt", "Prof", "Gen",
           "Gov", "Sen", "Rep", "Fr", "Atty", "Engr", "Hon", "vs", "etc", "No")


# Trailing "offer more help" closers, stripped if the model adds one despite the
# system prompt forbidding it. Belt-and-suspenders for the space-delimited
# languages (CJK, which isn't sentence-split below, is covered by the prompt).
_OFFER_RE = re.compile(
    r"\s*(?:"
    r"is there (?:anything|something) else[^.?!]*[?.!]?|"
    r"(?:can|may|how (?:can|may)) i (?:help|assist)[^.?!]*[?.!]?|"
    r"let me know if[^.?!]*[?.!]?|"
    r"feel free to (?:ask|reach out)[^.?!]*[?.!]?|"
    r"(?:do you|would you) (?:want|like)[^.?!]*(?:know more|else)[^.?!]*[?.!]?|"
    r"may (?:iba pa ba akong )?maitutulong pa ba(?:\s+ako)?[^.?!]*[?.!]?|"
    r"may iba pa ba[^.?!]*[?.!]?|"
    r"anything else[^.?!]*[?.!]?"
    r")\s*$",
    re.IGNORECASE)


# Cues that the user wants a fuller answer, or that the question needs several
# steps to answer properly. These promote the reply from the 25-word default to
# the 50-word tier. Kept multilingual so the tier works in any supported
# language, not just English.
_LONG_ANSWER_CUES = (
    "explain", "explanation", "tell me more", "more about", "in detail",
    "detailed", "elaborate", "why ", "why?", "how did", "how does", "how do",
    "compare", "difference between", "differences", "history of", "describe",
    "walk me through", "expand", "everything about", "full story", "background",
    "ipaliwanag", "paliwanag", "bakit", "paano", "kwento", "kasaysayan",
    "详细", "为什么", "解释", "说明",
    "詳しく", "なぜ", "説明", "どうして",
    "자세히", "왜", "설명", "어떻게",
)

# Word budgets for the two answer tiers (see _SYSTEM_INSTRUCTION "Answer length").
_WORDS_SHORT = 20
_WORDS_LONG = 50


def _wants_long_answer(question):
    """True when the user asked for detail, or the question is complex enough to
    need several explanations — those get the 50-word tier instead of 25."""
    q = (question or "").lower()
    if any(cue in q for cue in _LONG_ANSWER_CUES):
        return True
    # A genuinely multi-part question ("X and also Y?", or a long ask) needs room.
    if q.count("?") > 1:
        return True
    if len(q.split()) >= 18:
        return True
    return False


def _limit_answer(text, max_words=_WORDS_SHORT):
    """Backstop: plain text, the answer ONLY (no follow-up offer), capped at
    max_words words. Trims on a sentence boundary when one falls near the cap so
    the reply ends cleanly rather than mid-thought. Any leaked "anything else?"
    closer is stripped. CJK has no word spaces (one token), so it is never
    truncated."""
    text = re.sub(r"[*#_`]+", "", text)        # markdown reads awkwardly in TTS
    text = " ".join(text.split())
    text = _OFFER_RE.sub("", text).strip()     # drop a leaked help-offer closer
    # Protect abbreviation dots ("Dr.", "Mt.") so they don't look like sentence
    # ends when we pick the trim point.
    protected = text
    for a in _ABBREV:
        protected = re.sub(rf"\b{a}\.", a + "\x00", protected)
    sentences = [p.replace("\x00", ".")
                 for p in re.split(r"(?<=[.!?])\s+", protected) if p.strip()]
    if not sentences:
        return ""
    # Keep whole sentences while they fit in the budget — this is what makes a
    # 50-word answer read as finished prose instead of a cut-off paragraph.
    kept, used = [], 0
    for s in sentences:
        n = len(s.split())
        if kept and used + n > max_words:
            break
        kept.append(s)
        used += n
        if used >= max_words:
            break
    if not kept:                               # first sentence alone overflows
        kept = [sentences[0]]
    out = " ".join(kept)
    words = out.split()
    if len(words) > max_words:                 # hard trim, end cleanly
        words = words[:max_words]
        while len(words) > 1 and re.sub(r"[^\w]", "", words[-1].lower()) in _TRAIL_DROP:
            words.pop()
        out = " ".join(words).rstrip(",;:—- ")
        if out and out[-1] not in ".!?":
            out += "."
    return out.strip()


# Phrases that clear ALZONA's short-term memory. "goodbye" ends a visitor's
# session; "new conversation" starts fresh — both wipe the history so one
# visitor's context never carries over to the next person.
_GOODBYE_WORDS = ("goodbye", "good bye", "bye", "paalam",
                  "さようなら", "안녕히", "再见")
_NEWCONVO_WORDS = ("new conversation", "new chat", "start over", "start again",
                   "clear conversation", "clear chat", "clear memory", "reset",
                   "bagong usapan", "リセット", "초기화", "새 대화", "重新开始")


_NO_FILE_ANSWER = ("I don't have that in my files yet. Add a document about it "
                   "and ask me again.")


def answer_from_files(question, want_long=False):
    """FILE-BASED answer, used when Gemini is off: the knowledge-library
    sentence that shares the most words with the question, plus the sentence
    after it for a detailed question. Same keyword scoring as
    retrieve_knowledge, applied per sentence so the reply is the line that
    actually answers rather than a whole 1200-character chunk read aloud."""
    terms = [w for w in re.findall(r"[a-z0-9']+", question.lower())
             if len(w) > 2 and w not in _STOPWORDS]
    if not terms or not _knowledge_chunks:
        return _NO_FILE_ANSWER
    # A sentence must share at least half the question's keywords. One stray
    # shared word is not an answer: "who won the world cup" matched a sentence
    # about the World Robot Olympiad on "world" alone.
    need = (len(terms) + 1) // 2
    best, best_score = None, 0
    for chunk in _knowledge_chunks:
        # Same abbreviation guard as _limit_answer, so "Dr. Yanga" stays whole.
        protected = chunk["text"]
        for a in _ABBREV:
            protected = re.sub(rf"\b{a}\.", a + "\x00", protected)
        sentences = [s.strip().replace("\x00", ".") for s in
                     re.split(r"(?<=[.!?])\s+|\n+", protected) if s.strip()]
        for i, s in enumerate(sentences):
            low = s.lower()
            distinct = sum(1 for t in terms if t in low)
            if distinct < need:
                continue
            score = distinct * 3 + sum(low.count(t) for t in terms)
            if score > best_score:
                best_score = score
                best = " ".join(sentences[i:i + (2 if want_long else 1)])
    if not best:
        return _NO_FILE_ANSWER
    return _limit_answer(best, _WORDS_LONG if want_long else _WORDS_SHORT) \
        or _NO_FILE_ANSWER


def detect_memory_reset(text):
    """Return 'new' or 'goodbye' if the message should wipe conversation memory,
    else None. 'new conversation' is checked first so it wins over a trailing
    'bye'."""
    t = text.lower().strip()
    if any(w in t for w in _NEWCONVO_WORDS):
        return "new"
    if any(w in t for w in _GOODBYE_WORDS):
        return "goodbye"
    return None


def route_command(transcript, asked_by=""):
    """Dispatch a spoken command to arduino / video / baybayin / chat."""
    t = transcript.lower()

    # Answer to her own name before anything else looks at the text. Left to
    # the knowledge model, "alzona" came back as an answer about the surname.
    # "speak in Spanish" — a setting, handled before anything tries to answer.
    asked_for = detect_language_request(transcript)
    if asked_for:
        return {"mode": "chat", "reply": set_language(asked_for)}

    said_hello = identity_reply(transcript)
    if said_hello:
        return {"mode": "chat", "reply": said_hello}

    # Clear short-term memory on "new conversation" / "goodbye" so the next
    # visitor starts fresh. Acknowledge in the speaker's language.
    reset = detect_memory_reset(transcript)
    if reset:
        conversation_history.clear()
        lang = detect_reply_language(transcript)
        if reset == "goodbye":
            acks = {"Japanese": "さようなら！またね！",
                    "Korean": "안녕히 가세요! 또 만나요!",
                    "Filipino": "Paalam! Kita tayo ulit!",
                    "Chinese": "再见！下次见！"}
            reply = acks.get(lang, "Goodbye! Talk to you soon.")
        else:
            acks = {"Japanese": "新しい会話を始めましょう！",
                    "Korean": "새로운 대화를 시작해요!",
                    "Filipino": "Sige, bagong usapan na tayo!",
                    "Chinese": "好的，我们开始新的对话吧！"}
            reply = acks.get(lang, "Okay, let's start a new conversation!")
        return {"mode": "chat", "reply": reply}

    # for kw, cmd in ARDUINO_COMMANDS.items():
    #     if kw in t:
    #         ok = send_arduino(cmd)
    #         reply = (f"Done — sent '{cmd}' to the device."
    #                  if ok else f"I understood '{cmd}', but no Arduino is connected.")
    #         return {"mode": "arduino", "reply": reply, "command": cmd}

    # Croatian dances first: several names ("kolo") would otherwise be caught
    # by a broader rule further down, and this is the more specific question.
    asked_in_croatian = detect_reply_language(transcript) == "Croatian"

    found = _croatian_dance(t)
    if found:
        return _croatian_dance_reply(*found, croatian=asked_in_croatian)

    # "show me a Croatian dance" — no dance named, so she picks one. Rotating
    # rather than random: a visitor who asks twice should not be shown the same
    # dance twice, and a demo run repeatedly should not look like it knows one.
    # Asked in Croatian too: someone who says "pokaži mi hrvatski ples" is
    # asking exactly this question, and matching only the English words meant
    # she offered to talk about Filipino dances instead.
    croatia_named = any(w in t for w in ("croatia", "croatian", "hrvatsk"))
    dance_named = any(w in t for w in ("dance", "dances", "folk", "sayaw",
                                       "ples", "plesov", "kolo", "folklor"))
    if croatia_named and dance_named:
        global _croatian_turn
        names = list(CROATIAN_DANCES)
        name = names[_croatian_turn % len(names)]
        _croatian_turn += 1
        return _croatian_dance_reply(name, CROATIAN_DANCES[name],
                                     croatian=asked_in_croatian)

    for name, path in DANCES.items():
        if name in t:
            # English, like every dance description — see
            # _croatian_dance_reply for why.
            return {"mode": "video",
                    "reply": _dance_spoken(name),
                    "video_url": f"/media/{path}",
                    # The same thirty seconds every dance gets. A visitor
                    # watches a clip; the next one should not wait through a
                    # full performance.
                    "video_start": 0, "video_seconds": CLIP_SECONDS}

    # Philippine festival dance -> the Singkil, every time.
    #
    # Placed AFTER the loop above on purpose: naming a dance outright still
    # gets you that dance, so "tinikling festival" is still Tinikling. Only a
    # question with no dance named falls through to here.
    #
    # Note the wording. Singkil is a DANCE, not a festival of its own, and
    # ALZONA is meant to correct cultural mistakes rather than make them — so
    # she recommends it as the dance to watch and never implies otherwise.
    if wants_festival_dance(t):
        return {"mode": "video",
                "reply": _FESTIVAL_SPOKEN,
                "video_url": f"/media/{SINGKIL_VIDEO}",
                "video_start": 0, "video_seconds": CLIP_SECONDS}
    # if ("baybayin" in t or "baybay" in t) and any(w in t for w in ["teach", "learn", "video", "tutorial", "lesson"]):
    #     return {"mode": "video", "reply": "Here is a video teaching the Baybayin script.",
    #             "video_url": f"/media/{TEACHING_VIDEO}"}

    # ---- Lupang Hinirang: imitate / harmonize -------------------------------
    # The actual listening, pitch tracking and synthesis happen in the browser
    # (Web Audio) — it hears the mic directly, so there is no upload round-trip
    # and ALZONA can answer within a beat of the user finishing. This branch
    # only interprets the spoken command and tells the frontend which mode to
    # arm, via the "sing" object.
    sing = detect_sing_command(transcript)
    if sing:
        if sing["mode"] == "harmonize":
            parts = sing["parts"]
            if len(parts) == 1:
                who = parts[0]
            elif len(parts) == len(SATB_PARTS):
                who = "all four parts"
            else:
                who = " and ".join([", ".join(parts[:-1]), parts[-1]])
            reply = (f"Okay — sing Lupang Hinirang and I'll harmonize with you in "
                     f"{who}. Starting on G4, four four time.")
        else:
            reply = "Sing a line of Lupang Hinirang and I'll sing it back to you."
        return {"mode": "sing", "reply": reply, "sing": sing}

    # "what coin is this", "identify this coin", "anong barya ito"
    if any(w in t for w in ("coin", "barya", "salapi", "currency", "money")) and \
            any(w in t for w in ("what", "which", "identify", "read", "see", "this",
                                 "ano", "anong", "kilalanin", "tingnan")):
        if latest_frame is None:
            return {"mode": "coin", "reply": "The camera isn't ready yet."}
        # Acknowledge now, read the coin behind it. The answer reaches the
        # console through /state, which it polls anyway — no new plumbing, and
        # nobody waits in silence for a model round trip.
        global _coin_busy
        if not _coin_busy:
            _coin_busy = True
            threading.Thread(target=_coin_in_background,
                             args=(asked_by,), daemon=True).start()
        return {"mode": "coin", "reply": "Let me look at that coin.",
                "coin_pending": True}

    if wants_baybayin(t):
        target = extract_baybayin_target(transcript)
        if target:
            url = make_baybayin_image(target)
            if url:
                reply = f"Here is '{target}' written in Baybayin."
                if _AUTOPRINT:
                    reply += " I'm printing it for you now."
                return {"mode": "baybayin", "reply": reply,
                        "image_url": url, "word": target, "printing": _AUTOPRINT}
        return {"mode": "chat", "reply": "Which word would you like me to write in Baybayin?"}

    try:
        if not GEMINI_ENABLED:
            reply = answer_from_files(transcript, _wants_long_answer(transcript))
            conversation_history.append((transcript, reply))
            del conversation_history[:-MAX_HISTORY_TURNS]
            return {"mode": "chat", "reply": reply}

        context = retrieve_knowledge(transcript)
        # The language this question is in, falling back to the last one she
        # was speaking when it cannot be told — see current_language.
        lang = language_for(transcript)
        lang_rule = (
            f"Reply ONLY in {lang}. Do not translate the question back, and do "
            f"not explain the user's own words to them."
        )

        parts = []
        if conversation_history:
            convo = "\n".join(
                f"User: {u}\nALZONA: {a}"
                for u, a in conversation_history[-MAX_HISTORY_TURNS:]
            )
            parts.append(
                "Recent conversation so far — use it ONLY for context and to "
                "resolve follow-ups (e.g. 'tell me more', 'and her?'). Do NOT let "
                "its language change the language of your reply:\n" + convo
            )
        if context:
            parts.append(
                "Reference material from your knowledge library — your PRIMARY "
                "source. If it FULLY answers, use ONLY it; if it is incomplete, "
                "supplement with your own accurate knowledge (never contradicting "
                "it):\n\n" + context
            )
        else:
            parts.append(
                "Your knowledge library has no relevant material for this "
                "message. If it is a genuine question, use the Google Search tool "
                "for an accurate answer — do not dismiss it. Romanized foreign "
                "phrases are real language, never gibberish."
            )
        parts.append(f"Current user message: {transcript}")
        want_long = _wants_long_answer(transcript)
        if want_long:
            length_rule = (
                "This question asks for detail or needs several explanations, so "
                "give a fuller answer of AT MOST 50 WORDS — two or three tight, "
                "complete sentences. Do not pad to reach 50; stop when the answer "
                "is complete."
            )
        else:
            length_rule = (
                "Answer in at most 20 WORDS and at most two sentences, stating "
                "the fact asked plus the single most useful detail."
            )
        parts.append(
            f"({lang_rule} {length_rule} Answer ONLY — do NOT add a follow-up "
            "question or any offer of further help. Plain text — no lists, no "
            "headings, no markdown.)"
        )
        prompt = "\n\n".join(parts)

        # KB already has the answer -> skip the web-search grounding round-trip
        # (much faster). Only reach for search when the library came up empty.
        cfg = GEN_CONFIG_FAST if context else GEN_CONFIG
        resp = gen_content(model=CHAT_MODEL, contents=prompt, config=cfg)
        reply = _limit_answer((resp.text or "").strip(),
                              _WORDS_LONG if want_long else _WORDS_SHORT)
        if not reply:
            reply = "Sorry, I didn't catch that. Could you please rephrase?"

        # Remember this exchange for follow-ups; keep only the last few turns.
        conversation_history.append((transcript, reply))
        del conversation_history[:-MAX_HISTORY_TURNS]
    except Exception as e:
        print("gemini error:", e)
        reply = "I'm having trouble answering right now. Please try again."
    return {"mode": "chat", "reply": reply}


@app.post('/upload_knowledge')
async def upload_knowledge(file: UploadFile = File(...)):
    """Add a document to ALZONA's knowledge library."""
    name = os.path.basename(file.filename or "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in KNOWLEDGE_EXTS:
        return JSONResponse(
            {"error": f"Unsupported type '{ext}'. Use: {', '.join(KNOWLEDGE_EXTS)}"},
            status_code=400,
        )
    data = await file.read()
    with open(os.path.join(KNOWLEDGE_DIR, name), "wb") as f:
        f.write(data)
    await run_in_threadpool(reload_knowledge)
    return {"ok": True, "file": name, "chunks": len(_knowledge_chunks)}


@app.get('/knowledge')
def list_knowledge():
    """List the files in the knowledge library."""
    files = [
        {"name": fn, "size": os.path.getsize(os.path.join(KNOWLEDGE_DIR, fn))}
        for fn in sorted(os.listdir(KNOWLEDGE_DIR))
        if fn.lower().endswith(KNOWLEDGE_EXTS)
    ]
    return {"files": files, "chunks": len(_knowledge_chunks)}


@app.delete('/knowledge/{filename}')
def delete_knowledge(filename: str):
    """Remove a file from the knowledge library."""
    name = os.path.basename(filename)
    path = os.path.join(KNOWLEDGE_DIR, name)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    os.remove(path)
    reload_knowledge()
    return {"ok": True}


@app.post('/say')
async def say(text: str = Form(...), voice: str = Form("")):
    """Speak a short phrase in ALZONA's voice (wake/stop acknowledgments).

    voice="gemini" skips ElevenLabs for this call only. The consoles share one
    backend, so a caller that wants the faster voice asks for it rather than
    turning it off for everybody.
    """
    text = (text or "").strip()
    if not text:
        return JSONResponse({"tts_url": None, "error": "empty"})
    fast = voice.strip().lower() in ("gemini", "fast", "default")
    # Threadpool keeps the event loop (and the /video stream) responsive.
    # cache=True: identical phrases (wake/stop acknowledgments, repeated
    # replies) are synthesized once and replayed instantly afterwards.
    fn = await run_in_threadpool(fast_voice, text, True, fast)
    return {"tts_url": f"/tts/{fn}" if fn else None}


@app.get('/tts/{filename}')
def serve_tts(filename: str):
    file_path = os.path.join(BASE, 'static', 'tts', filename)
    if not os.path.exists(file_path):
        return Response(status_code=404)
    media = 'audio/wav' if filename.lower().endswith('.wav') else 'audio/mpeg'
    return FileResponse(file_path, media_type=media, filename=filename)


@app.post('/identify_coin')
async def identify_coin_endpoint():
    """Grab the current camera frame and identify the coin in it."""
    frame = latest_frame
    if frame is None:
        return JSONResponse({"ok": False, "error": "The camera isn't ready yet."})
    result = await run_in_threadpool(identify_coin, frame)
    if result.get("ok") and result.get("spoken"):
        tts = await run_in_threadpool(fast_voice, result["spoken"])
        result["tts_url"] = f"/tts/{tts}" if tts else None
    return result


_anthem_lines_cache = None


def _anthem_lines():
    """The anthem's lyric lines, in order, from the aligned lyrics file."""
    global _anthem_lines_cache
    if _anthem_lines_cache is None:
        try:
            path = os.path.join(BASE, "source", "harmony", "lyrics.json")
            with open(path, encoding="utf-8") as f:
                _anthem_lines_cache = [l["text"] for l in json.load(f)["lines"]]
        except Exception as e:
            print("lyrics unavailable:", str(e)[:80])
            _anthem_lines_cache = []
    return _anthem_lines_cache


@app.post('/listen')
async def listen(file: UploadFile = File(...)):
    """One ear for everything: decide whether a clip is SINGING or SPEECH.

    The browser's SpeechRecognition insists on owning the microphone, so it
    cannot run alongside the continuous pitch tracking the harmony needs — one
    starves the other and BOTH features stop working, which is exactly what was
    measured: recognition restarting every second and hearing nothing, while the
    singing side received no audio either.

    Routing every clip through here instead means a single always-open
    microphone serves both. ALZONA hears singing and harmonises, or hears a
    question and answers it, with nothing to switch between and no command to
    remember.
    """
    try:
        data = await file.read()
        if not data:
            return {"kind": "none", "error": "empty audio"}
        if not GEMINI_ENABLED:
            # Telling singing from speech needs Gemini; the pitch tracking and
            # harmony run in the browser and keep working without it.
            return {"kind": "none"}
        lines = _anthem_lines()
        numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(lines))
        prompt = (
            "Listen to this short clip and decide what it is.\n\n"
            "If the person is SINGING, the song ALZONA knows is the Philippine "
            "national anthem, Lupang Hinirang. Its lines, numbered:\n\n"
            + numbered +
            "\n\nIf the person is SPEAKING — asking a question, giving an "
            "instruction, or talking — transcribe what they said.\n\n"
            "Reply with ONLY a JSON object:\n"
            '  "kind"  - "singing", "speech", or "none" for silence or noise\n'
            '  "line"  - when singing: which numbered line they START on, or '
            "null if the words are unclear\n"
            '  "song"  - when singing: the title you recognise, else null\n'
            '  "text"  - when speaking: what they said, verbatim\n\n'
            "Sung words stretch across held notes; speech does not. Judge the "
            "line from words you can actually hear, never from the tune alone — "
            "several lines share a melody. Do not default to line 0."
        )
        r = await run_in_threadpool(
            gen_content,
            model=CHAT_MODEL,
            contents=[types.Part.from_bytes(data=data, mime_type="audio/webm"), prompt],
            config=GEN_CONFIG_FAST)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (r.text or "").strip())
        try:
            info = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            info = json.loads(m.group(0)) if m else {}

        kind = (info.get("kind") or "none").lower()

        if kind == "singing":
            idx = info.get("line")
            idx = int(idx) if str(idx).isdigit() else None
            ok = idx is not None and 0 <= idx < len(lines)
            return {"kind": "singing", "song": info.get("song"),
                    "index": idx if ok else None,
                    "text": lines[idx] if ok else None}

        if kind == "speech":
            said = (info.get("text") or "").strip()
            if not said:
                return {"kind": "none"}
            result = await run_in_threadpool(route_command, said)
            reply = result.get("reply", "")
            tts = await run_in_threadpool(fast_voice, reply) if reply else None
            return {"kind": "speech", "transcript": said, "reply": reply,
                    "mode": result.get("mode", "chat"),
                    "image_url": result.get("image_url"),
                    "video_url": result.get("video_url"),
                    "coin": result.get("coin"), "sing": result.get("sing"),
                    "coin_pending": result.get("coin_pending"),
                    "video_start": result.get("video_start"),
                    "video_seconds": result.get("video_seconds"),
                    "tts_url": f"/tts/{tts}" if tts else None}

        return {"kind": "none"}
    except Exception as e:
        print("listen error:", str(e)[:120])
        return {"kind": "none", "error": str(e)[:80]}


# =========================================================
# WHO HOLDS THE MICROPHONE
# =========================================================
# Two consoles can be open at once — 5173 where she follows the singer, 5174
# where she leads — and each runs its own speech recognition. The browser gives
# the microphone to ONE of them, so the other is aborted the moment it starts.
# Measured live, they ping-ponged every two seconds and NEITHER ever heard a
# word, with nothing on either screen to say why.
#
# They are different origins, so they cannot see each other through the browser:
# no shared storage, no BroadcastChannel. The backend is the only thing they
# both talk to, so the lease lives here.
#
# A lease with a deadline rather than a flag someone must remember to clear: a
# page that is closed, reloaded or crashes never sends a release, and a flag
# left set would lock every console out of the microphone for good.
_MIC_LEASE = {"holder": None, "at": 0.0}
_MIC_LEASE_TTL = 4.0        # a holder that stops renewing has gone away


@app.post('/mic_lease')
async def mic_lease(client: str = Form(...), release: str = Form("0"),
                    focused: str = Form("0")):
    """Claim, renew or drop the right to listen. Returns who holds it.

    A FOCUSED console takes the microphone from an unfocused one. Without that
    the first page to load kept it for ever, and the answer to "this console
    cannot hear me" was to go and close the other one — which is no answer at
    all when both are wanted open. Whichever window someone is actually looking
    at is the one they are talking to.
    """
    now = time.time()
    lease = _MIC_LEASE
    has_focus = focused in ("1", "true", "yes")

    if release in ("1", "true", "yes"):
        if lease["holder"] == client:
            lease["holder"] = None
        return {"holder": lease["holder"], "yours": False}

    expired = now - lease["at"] > _MIC_LEASE_TTL
    # Only a focused claimant may take it from a live holder. An unfocused page
    # must wait for the lease to lapse, or it would snatch it straight back.
    if lease["holder"] in (None, client) or expired or has_focus:
        lease["holder"] = client
        lease["at"] = now

    return {"holder": lease["holder"], "yours": lease["holder"] == client}


@app.post('/debug_log')
async def debug_log(line: str = Form(...)):
    """Take a diagnostic line from the browser and append it to a file.

    The singing features run entirely in the browser, so without this there is
    no server-side trace of what the microphone actually delivered — and that
    trace is what found every real bug in this feature.
    """
    try:
        path = os.path.join(BASE, "static", "sing_debug.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {line[:400]}\n")
    except Exception as e:
        return {"ok": False, "error": str(e)[:80]}
    return {"ok": True}


@app.get('/state')
def state():
    """Return a minimal state object the frontend expects."""
    return {
        'face_state': face_state,
        'age_result': age_result,
        'chat': True,
        'running': running,
        'print_status': last_print_status,
        # Rises by one each time a coin is read, so the console can tell a new
        # answer from the one it already showed.
        'coin': last_coin_result,
        # "server": this backend has the webcam and streams it on /video.
        # "browser": it has none (CAMERA_INDEX=off), so the console shows its
        # own camera and uploads frames to /frame.
        'camera': 'server' if CAMERA_ENABLED else 'browser',
    }


_MAX_FRAME_BYTES = 2_000_000    # a 640x480 JPEG is ~50 KB; this is generous


@app.post('/frame')
async def upload_frame(file: UploadFile = File(...)):
    """A camera frame from the console's browser, for a backend with no webcam.

    Does what the laptop's camera loop does for each frame it reads: keeps it
    as latest_frame (which coin identification reads) and runs face presence
    and age on it (face_state / age_result, which drive the greeting).
    """
    global latest_frame
    if CAMERA_ENABLED:
        # This machine has its own camera; a second source would fight it.
        return JSONResponse({"ok": False, "error": "this backend uses its own camera"},
                            status_code=409)
    data = await file.read()
    if not data or len(data) > _MAX_FRAME_BYTES:
        return JSONResponse({"ok": False, "error": "empty or oversized frame"},
                            status_code=400)
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"ok": False, "error": "not an image"}, status_code=400)
    latest_frame = data
    # Busy analysing the previous upload: keep this one for coins, skip faces.
    if not _face_lock.acquire(blocking=False):
        return {"ok": True, "analysed": False}
    try:
        await run_in_threadpool(analyse_frame, frame)
    finally:
        _face_lock.release()
    return {"ok": True, "analysed": True}


@app.get('/video')
def video_feed():
    """Return an MJPEG stream of the latest camera frames."""
    if not CAMERA_ENABLED:
        return Response(status_code=204)   # no camera here; see /frame

    def generate():
        global latest_frame

        while True:
            if latest_frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + latest_frame
                    + b"\r\n"
                )
            time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
        },
    )


# =========================================================
# WHEN THE RECOGNISER'S FAVOURITE GUESS IS WRONG
# =========================================================
# Speech recognition returns several readings of the same audio and its top
# choice is not always the right one. Measured live: "translate pilipinas to
# baybayin" arrived as "Can you translate Filipinas to be buying?" — a word the
# recogniser has never seen becomes whatever English it resembles.
#
# Only a COMMAND is worth second-guessing. A question is answered from whatever
# was heard and the model copes with a stray word; a command either matches a
# pattern or silently does nothing, and that is what a demo cannot afford.
#
# Nothing here runs a command — it only asks whether a reading LOOKS like one.
# Executing each candidate to find out would print several Baybayin sheets.


def transcribe_clip(data):
    """What was actually said, according to Gemini. "" when it cannot tell.

    The browser's recogniser has no Filipino vocabulary, even in its en-PH
    model, so it forces Filipino audio onto the nearest English it knows and
    reports itself certain: "baybayin" came back as "by buying", "into buying"
    and "by Bayern", and the word to translate as Leslie, Guadalupe, Kishata.

    It is also the only way a language OTHER than the selected one is heard at
    all. The Web Speech API cannot detect language — it transcribes whatever it
    is given as the language it was told to expect — so a visitor speaking
    Spanish into a recogniser set to en-PH produces English-shaped nonsense.
    This transcribes what was actually said, in whatever language it was said,
    which is what lets her answer in it.
    """
    if not data or not GEMINI_ENABLED:
        return ""         # keep the browser's own reading
    try:
        r = gen_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_bytes(data=data, mime_type="audio/webm"),
                "Transcribe this clip verbatim, in whatever language it is "
                "spoken. It will be English, Filipino, Spanish, Mandarin "
                "Chinese or Croatian, and may mix English with any of them in "
                "one sentence. Write it in that language's own script — do NOT "
                "translate it into English, and do not romanise Chinese. "
                "Expect Filipino and Croatian names of places, dances and "
                "people, including 'Baybayin', the pre-colonial Philippine "
                "script, and 'ALZONA', the robot being addressed. Write exactly "
                "what was said, with no commentary and no quotation marks. "
                "Reply with an empty string if there is no speech.",
            ],
            config=GEN_CONFIG_FAST,
        )
        return " ".join((r.text or "").split()).strip().strip('"')
    except Exception as e:
        print("transcribe_clip failed:", str(e)[:120])
        return ""


def _looks_like_command(text):
    if not text:
        return False
    if detect_sing_command(text):
        return True
    if wants_baybayin(text):
        return True
    if identity_reply(text):
        return True
    t = text.lower()
    if "coin" in t and any(w in t for w in ("identify", "what", "scan", "read", "check")):
        return True
    return False


def _same_utterance(a, b):
    """Are these two readings of the SAME audio?

    Alternatives from the recogniser differ in a word or two, not wholesale. A
    candidate sharing nothing with the top reading is not a rival reading of it,
    and accepting one turns an innocent question into a command: "who is jose
    rizal" became a harmonise instruction in testing because some unrelated
    alternative happened to look like one. Requiring real overlap keeps the
    second-guessing to what it is for.
    """
    wa = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(w) > 2}
    wb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(w) > 2}
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.34


def _needs_second_opinion(text):
    """Is the browser's reading worth checking against Gemini?

    Not a command at all — yes, obviously.

    A BAYBAYIN command — also yes, even though the intent was recognised. The
    word is free-form and cannot be checked against anything, and it is the
    entire point of the request: "Can you translate Leslie to by buying?" is
    recognised as a Baybayin request and would confidently print LESLIE. Getting
    the intent right while getting the word wrong is the failure that wastes
    paper and puts the wrong thing in a visitor's hand.

    A sing or coin command — no. Their content is a fixed vocabulary the browser
    either matched or did not, so a second reading has nothing to add.
    """
    if not _looks_like_command(text):
        return True
    return bool(wants_baybayin(text))


def pick_command_text(primary, alts):
    """The reading to act on: the top guess, unless a runner-up is a command."""
    if _looks_like_command(primary):
        return primary, False
    for a in alts or []:
        a = (a or "").strip()
        if not a or a == primary:
            continue
        if _looks_like_command(a) and _same_utterance(primary, a):
            return a, True
    return primary, False


@app.post('/command')
async def command(text: str = Form(...), skip_tts: str = Form(""),
                  alts: str = Form(""), audio: UploadFile = File(None),
                  client: str = Form("")):
    """Text command (from the browser's speech recognition) -> route -> reply.

    `alts` carries the recogniser's other readings of the same audio. They are
    consulted only when the top reading is not a command — see
    pick_command_text — so a command does not hinge on the recogniser's first
    choice while a question is still answered from what was actually heard.

    With skip_tts=1 the reply text returns immediately and the frontend
    fetches the audio separately via /say (text shows while voice renders)."""
    text = (text or "").strip()
    if not text:
        return JSONResponse({"transcript": "", "reply": "", "error": "empty"})

    others = []
    if alts:
        try:
            parsed = json.loads(alts)
            if isinstance(parsed, list):
                others = [str(a) for a in parsed][:5]
        except Exception:
            others = []

    chosen, from_alt = pick_command_text(text, others)
    if from_alt:
        print(f"command: heard {text!r}, acting on {chosen!r}")
    text = chosen

    # The browser got a command out of it — usually no second opinion needed,
    # and no reason to spend a Gemini call or the second it costs.
    if audio is not None and _needs_second_opinion(text):
        clip = await audio.read()
        better = await run_in_threadpool(transcribe_clip, clip)
        if better and better.lower() != text.lower():
            print(f"command: browser heard {text!r}, Gemini heard {better!r}")
            text = better

    # Gemini calls block for seconds; threadpool keeps /video streaming.
    result = await run_in_threadpool(route_command, text, client)
    reply = result.get("reply", "")
    tts = None
    if reply and skip_tts != "1":
        tts = await run_in_threadpool(fast_voice, reply)
    return {
        "transcript": text, "reply": reply, "mode": result.get("mode", "chat"),
        "image_url": result.get("image_url"), "video_url": result.get("video_url"),
        "word": result.get("word"), "command": result.get("command"),
        "coin": result.get("coin"), "printing": result.get("printing"),
        "sing": result.get("sing"),
        # True while a coin is still being read in the background; the answer
        # itself arrives on /state.
        "coin_pending": result.get("coin_pending"),
        "video_start": result.get("video_start"),
        "video_seconds": result.get("video_seconds"),
        "tts_url": f"/tts/{tts}" if tts else None,
    }


@app.post('/upload_audio')
async def upload_audio(file: UploadFile = File(...)):
    """Voice -> Gemini STT -> route (baybayin / video / arduino / chat) -> spoken reply."""
    global last_user_input, subtitle_history
    try:
        data = await file.read()
        transcript = await run_in_threadpool(
            gemini_transcribe, data, file.content_type or "audio/webm"
        )
        if not transcript:
            return JSONResponse({"transcript": "", "reply": "", "error": "No speech detected"})
        last_user_input = transcript
        result = await run_in_threadpool(route_command, transcript)
        reply = result.get("reply", "")
        tts = await run_in_threadpool(fast_voice, reply) if reply else None
        return {
            "transcript": transcript,
            "reply": reply,
            "mode": result.get("mode", "chat"),
            "image_url": result.get("image_url"),
            "video_url": result.get("video_url"),
            "word": result.get("word"),
            "command": result.get("command"),
            "tts_url": f"/tts/{tts}" if tts else None,
        }
    except Exception as e:
        print("upload_audio error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "status": "running"
    }

# =========================================================
# PLAY AUDIO
# =========================================================
# Note: local playback removed. Frontend will request TTS files at /tts/<file>
def play_audio(file_path):
    """No-op playback function to keep existing code paths stable.

    TTS files are generated to `static/tts/` and served to the frontend.
    """
    print(f"play_audio (noop) -> {file_path}")


def listen():
    """Return the most recent transcript posted by the web UI (once).

    The web UI should POST audio to `/upload_audio`, which updates
    `last_user_input`. `listen()` consumes that value and clears it.
    """
    global last_user_input, subtitle_history

    if not last_user_input:
        return None

    text = last_user_input
    last_user_input = ""

    # keep a minimal subtitle history entry
    try:
        subtitle_history = []
        subtitle_history.append({
            'id': str(uuid.uuid4()),
            'speaker': 'user',
            'text': text,
            'time': time.time()
        })
    except Exception:
        pass

    print(f"USER (web): {text}")
    return text

# def play_audio(file_path):

#     pygame.mixer.music.load(file_path)
#     pygame.mixer.music.play()

#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)

#     pygame.mixer.music.stop()
#     pygame.mixer.music.unload()

#     # keep the single voice file around; it will be removed before next synthesis

def detect_command(text):
    if not text:
        return False

    t = text.lower().strip()

    teach_words = [
        'teach', 
        'turo', 
        'turuan', 
        'turuan mo ako', 
        'turuan mo ako ng baybayin',
        'can you teach me baybayin',
        'can you help me translate to baybayin',
    ]

    return any(word in t for word in teach_words)

# =========================================================
# RUN PROGRAM
# =========================================================
if __name__ == "__main__":
    # Start background threads for face detection and chatbot
    if CAMERA_ENABLED:
        t1 = threading.Thread(target=face_detection, daemon=True)
        t1.start()
        print("✓ Face detection thread started")
    else:
        print("Camera off (CAMERA_INDEX=off) — no face detection.")

    # t2 = threading.Thread(target=chatbot, daemon=True)
    # t2.start()
    # print("✓ Chatbot thread started")

    # Run FastAPI server
    # Hosts such as Render choose the port and pass it in PORT.
    port = int(os.environ.get("PORT", "5002"))
    print(f"✓ Starting FastAPI on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)




