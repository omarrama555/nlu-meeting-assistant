"""
NLU Meeting Voice-to-Action Assistant  v3.0
Professional Edition — No emojis, refined design, Whisper large-v3, 35 features
"""

import re, os, json, time, textwrap, warnings
import datetime
from collections import Counter, defaultdict
from io import BytesIO, StringIO

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceIQ — Meeting Intelligence",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='16' cy='16' r='16' fill='%230f172a'/><rect x='13' y='6' width='6' height='14' rx='3' fill='%2338bdf8'/><path d='M9 17a7 7 0 0014 0' stroke='%2338bdf8' stroke-width='2' fill='none' stroke-linecap='round'/><rect x='14.5' y='24' width='3' height='3' rx='1' fill='%2338bdf8'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
INTENTS = ["action item", "decision", "suggestion", "question", "informational", "risk or issue"]
COLORS  = {
    "action item":   "#38bdf8",
    "decision":      "#34d399",
    "suggestion":    "#c084fc",
    "question":      "#fbbf24",
    "informational": "#64748b",
    "risk or issue": "#f87171",
}
INTENT_LABELS = {
    "action item":   "Action",
    "decision":      "Decision",
    "suggestion":    "Suggestion",
    "question":      "Question",
    "informational": "Info",
    "risk or issue": "Risk",
}
SPEAKER_COLORS = ["#38bdf8","#34d399","#c084fc","#fbbf24","#f87171","#fb923c","#a3e635"]

WHISPER_MODELS = {
    "large-v3 (Best accuracy, ~1.5GB)": "large-v3",
    "large-v2 (~1.5GB)": "large-v2",
    "medium (~760MB)": "medium",
    "small (~240MB)": "small",
    "base (~140MB)": "base",
    "tiny (~75MB, fastest)": "tiny",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:       #0a0f1e;
    --surface:  #0f172a;
    --surface2: #1e293b;
    --surface3: #263348;
    --border:   rgba(56,189,248,0.08);
    --border2:  rgba(255,255,255,0.06);
    --accent:   #38bdf8;
    --accent2:  #0ea5e9;
    --muted:    #475569;
    --muted2:   #64748b;
    --text:     #e2e8f0;
    --text2:    #94a3b8;
    --success:  #34d399;
    --danger:   #f87171;
    --warn:     #fbbf24;
}

html, body, [class*="st-"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: var(--bg);
    color: var(--text);
}
.main .block-container {
    padding: 2rem 2.5rem 5rem;
    max-width: 1440px;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: .82rem !important;
    font-weight: 500 !important;
    letter-spacing: .01em !important;
}

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .25s ease, box-shadow .25s ease;
}
.card:hover {
    border-color: rgba(56,189,248,0.2);
    box-shadow: 0 4px 24px rgba(56,189,248,0.04);
}
.card-sm {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: .9rem 1.1rem;
    margin-bottom: .6rem;
    transition: border-color .2s;
}
.card-sm:hover { border-color: rgba(56,189,248,0.2); }

/* ── Metric boxes ────────────────────────────────────── */
.metric {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric .val {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.metric .lbl {
    font-size: .67rem;
    color: var(--muted2);
    margin-top: .35rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-weight: 600;
}

/* ── Badge ───────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: .2rem;
    padding: .18rem .6rem;
    border-radius: 6px;
    font-size: .67rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    border: 1px solid;
}

/* ── Upload zone ─────────────────────────────────────── */
.upload-zone {
    background: linear-gradient(135deg, rgba(56,189,248,.05), rgba(192,132,252,.04));
    border: 1.5px dashed rgba(56,189,248,.25);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.2rem;
    transition: border-color .3s;
}
.upload-zone:hover { border-color: rgba(56,189,248,.5); }
.upload-icon {
    width: 52px;
    height: 52px;
    background: rgba(56,189,248,.1);
    border-radius: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
}

/* ── Pulse animation (recording indicator) ───────────── */
.rec-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--danger);
    animation: rec-pulse 1.4s infinite;
    margin-right: .5rem;
}
@keyframes rec-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .4; transform: scale(1.3); }
}

/* ── Timeline ────────────────────────────────────────── */
.tl-item {
    display: flex;
    align-items: flex-start;
    gap: .8rem;
    padding: .5rem 0;
    border-bottom: 1px solid var(--border2);
}
.tl-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: .45rem;
    flex-shrink: 0;
}
.tl-line {
    width: 1px;
    background: var(--border2);
    margin: 0 3px;
    flex-shrink: 0;
}

/* ── Sentence result card ────────────────────────────── */
.sent-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: .85rem 1.1rem;
    margin-bottom: .5rem;
    border-left: 3px solid var(--border2);
    transition: border-left-color .2s;
}

/* ── Speaker tag ─────────────────────────────────────── */
.spk-tag {
    font-size: .68rem;
    font-weight: 700;
    padding: .15rem .55rem;
    border-radius: 20px;
    border: 1px solid;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* ── Page heading ────────────────────────────────────── */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -.025em;
    margin-bottom: 1.5rem;
    color: var(--text);
}
.page-title span {
    color: var(--accent);
}

/* ── Section label ───────────────────────────────────── */
.section-label {
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--muted2);
    margin-bottom: .7rem;
    margin-top: .5rem;
}

/* ── Confidence bar ──────────────────────────────────── */
.conf-bar-bg {
    height: 3px;
    background: var(--surface3);
    border-radius: 2px;
    overflow: hidden;
    margin-top: .4rem;
}
.conf-bar-fg {
    height: 100%;
    border-radius: 2px;
    transition: width .4s ease;
}

/* ── Whisper model selector ──────────────────────────── */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: rgba(56,189,248,.1);
    border: 1px solid rgba(56,189,248,.25);
    border-radius: 8px;
    padding: .3rem .7rem;
    font-size: .72rem;
    color: var(--accent);
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

/* ── Progress override ───────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), #818cf8) !important;
    border-radius: 4px !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
    transition: opacity .2s, transform .15s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    opacity: .9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface3) !important;
    color: var(--text2) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
    font-size: .82rem !important;
    font-weight: 600 !important;
    letter-spacing: .02em !important;
    padding: .5rem 1.2rem !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
    background: var(--surface2) !important;
}

/* ── Input fields ────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--surface3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: var(--border2) !important; margin: 1.2rem 0 !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }

/* ── Alert strips ────────────────────────────────────── */
.alert-strip {
    border-radius: 10px;
    border: 1px solid;
    border-left-width: 4px;
    padding: .75rem 1rem;
    margin-bottom: .6rem;
    display: flex;
    gap: .75rem;
    align-items: flex-start;
}

/* ── Action item row ─────────────────────────────────── */
.action-row {
    display: flex;
    align-items: center;
    gap: .8rem;
    background: var(--surface2);
    border-radius: 10px;
    padding: .65rem 1rem;
    margin-bottom: .4rem;
    border: 1px solid var(--border2);
}
.action-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}
.action-dot.done { background: var(--success); }

/* ── Wordcloud container ─────────────────────────────── */
.wc-container {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 16px;
    overflow: hidden;
    padding: 1rem;
}

/* ── Status indicator ────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: .4rem;
    vertical-align: middle;
}
.status-dot.ok  { background: var(--success); }
.status-dot.err { background: var(--danger); }
.status-dot.warn { background: var(--warn); }

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "history": [], "action_items": [], "bookmarks": [],
    "alerts": [], "chat_history": [], "session_notes": "",
    "highlight_entities": True, "auto_summarize": True,
    "min_conf": 0.0, "comparison_a": None, "comparison_b": None,
    "theme": "Dark", "lang": "English", "pending_transcript": "",
    "model_loaded": False, "whisper_model": None,
    "whisper_size": "large-v3",
    # New features state
    "transcript_history": [],   # Feature 31: transcript storage
    "custom_vocab": [],          # Feature 32: custom vocabulary
    "meeting_tags": [],          # Feature 33: meeting tags
    "transcript_diffs": [],      # Feature 34: diff view
    "speaker_profiles": {},      # Feature 35: speaker profiles
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# NLP HELPERS  (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_nlp_models():
    models = {}
    try:
        from transformers import pipeline as hf_pipeline
        models["intent"] = hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )
    except Exception:
        models["intent"] = None

    try:
        from transformers import pipeline as hf_pipeline
        models["ner"] = hf_pipeline(
            "ner", model="dslim/bert-base-NER",
            aggregation_strategy="simple", device=-1
        )
    except Exception:
        models["ner"] = None

    try:
        from transformers import pipeline as hf_pipeline
        models["summarizer"] = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,
        )
    except Exception:
        models["summarizer"] = None

    try:
        from sentence_transformers import SentenceTransformer
        models["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        models["embedder"] = None

    return models


@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper(model_size: str = "large-v3"):
    try:
        import whisper
        return whisper.load_model(model_size)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
_INTENT_RULES = {
    "action item":   r"\b(will|need to|should|must|please|schedule|send|complete|finish|review|prepare|update|follow up|assign)\b",
    "decision":      r"\b(decided|agreed|approved|resolved|chose|selected|confirmed|will use|going with)\b",
    "risk or issue": r"\b(risk|issue|problem|bug|block|concern|fail|error|critical|miss|delay|danger)\b",
    "question":      r"\?|^\s*(what|when|where|who|how|why|which|is|are|can|could|should|do|did|has|have)",
    "suggestion":    r"\b(suggest|maybe|perhaps|consider|recommend|think about|might want|could try|what if|how about)\b",
}

def rule_based_intent(text: str) -> tuple:
    t = text.lower().strip()
    for intent, pattern in _INTENT_RULES.items():
        if re.search(pattern, t, re.IGNORECASE):
            return intent, 0.78
    return "informational", 0.65

def classify_sentence(text: str, models: dict = None) -> tuple:
    if not text.strip() or len(text.split()) < 2:
        return "informational", 0.50
    if models and models.get("intent"):
        try:
            out = models["intent"](text, INTENTS, multi_label=False)
            return out["labels"][0], round(out["scores"][0], 3)
        except Exception:
            pass
    return rule_based_intent(text)

_SENTIMENT_RULES = {
    "Positive": r"\b(great|good|excellent|done|completed|success|achieve|perfect|wonderful|agree|approved|happy|resolved|ready|thank)\b",
    "Negative": r"\b(fail|issue|problem|risk|concern|miss|block|bad|terrible|error|bug|unable|can't|won't|behind|delay|critical)\b",
}
def get_sentiment(text: str) -> tuple:
    t = text.lower()
    if re.search(_SENTIMENT_RULES["Negative"], t): return "Negative", "#f87171"
    if re.search(_SENTIMENT_RULES["Positive"], t): return "Positive", "#34d399"
    return "Neutral", "#fbbf24"

_ENTITY_PATTERNS = {
    "PERSON": r"\b([A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]{2,})\b",
    "DATE":   r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|q[1-4]|next week|by friday|tomorrow|end of week|[0-9]{1,2}[/-][0-9]{1,2})\b",
    "TECH":   r"\b(api|database|server|frontend|backend|ui|ux|devops|ci/cd|kubernetes|docker|aws|gcp|azure|react|python|node|sql|postgres|redis|kafka|microservice)\b",
    "ORG":    r"\b([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)*\s(?:Inc|Ltd|LLC|Corp|Team|Group|Department))\b",
}
def extract_entities(text: str) -> list:
    ents = []
    seen = set()
    for etype, pat in _ENTITY_PATTERNS.items():
        for m in re.finditer(pat, text, re.IGNORECASE):
            w = m.group().strip()
            key = (etype, w.lower())
            if len(w) > 1 and key not in seen:
                ents.append({"type": etype, "text": w})
                seen.add(key)
    return ents

def extract_topics(text: str) -> list:
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    stop = {"that","this","with","from","they","have","will","been","were","when","what","your","more","also","into","some","their","which","about","after","before","there","these","those","then","than","just"}
    cnt = Counter(w for w in words if w not in stop)
    return [w for w, _ in cnt.most_common(8)]

def temporal_normalize(text: str) -> list:
    ref = datetime.date.today()
    wd = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    res = []
    for m in re.finditer(r"next (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text, re.I):
        t = wd.index(m.group(1).lower())
        diff = (t - ref.weekday()) % 7 or 7
        res.append({"expression": m.group(0), "date": (ref + datetime.timedelta(days=diff)).strftime("%Y-%m-%d")})
    for m in re.finditer(r"in (\d+) (days?|weeks?)", text, re.I):
        n = int(m.group(1))
        d = ref + (datetime.timedelta(days=n) if "day" in m.group(2) else datetime.timedelta(weeks=n))
        res.append({"expression": m.group(0), "date": d.strftime("%Y-%m-%d")})
    if re.search(r"\btomorrow\b", text, re.I):
        res.append({"expression": "tomorrow", "date": (ref + datetime.timedelta(1)).strftime("%Y-%m-%d")})
    if re.search(r"end of (the )?week", text, re.I):
        diff = (4 - ref.weekday()) % 7
        res.append({"expression": "end of week", "date": (ref + datetime.timedelta(days=diff)).strftime("%Y-%m-%d")})
    return res

def extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if not sentences: return text[:200]
    scored = [(len(s.split()), i, s) for i, s in enumerate(sentences)]
    top = sorted(scored, reverse=True)[:max_sentences]
    top.sort(key=lambda x: x[1])
    return " ".join(s for _, _, s in top)

def analyze_transcript(transcript: str, selected_intents=None, models=None, meeting_tags=None) -> dict:
    if selected_intents is None: selected_intents = INTENTS
    lines = [l.strip() for l in transcript.strip().split("\n") if l.strip()]

    speaker_re = re.compile(r"^(\[?[A-Za-z][A-Za-z\s\-\.]{0,25}\]?):\s*(.+)$")
    speakers_raw = []
    plain_sentences = []

    for line in lines:
        m = speaker_re.match(line)
        if m:
            spk = m.group(1).strip("[]").strip()
            txt = m.group(2).strip()
            speakers_raw.append({"speaker": spk, "text": txt})
            plain_sentences.extend(re.split(r"(?<=[.!?])\s+", txt))
        else:
            plain_sentences.extend(re.split(r"(?<=[.!?])\s+", line))

    plain_sentences = [s.strip() for s in plain_sentences if len(s.strip()) > 4]

    classified = []
    for s in plain_sentences:
        intent, conf = classify_sentence(s, models)
        if intent not in selected_intents: continue
        if conf < st.session_state.min_conf: continue
        sent, sent_color = get_sentiment(s)
        ents = extract_entities(s)
        priority = 3 if intent == "risk or issue" else 2 if intent == "action item" else 1
        classified.append({
            "text": s, "intent": intent, "confidence": conf,
            "sentiment": sent, "sent_color": sent_color,
            "entities": ents, "priority": priority,
        })

    full_text = " ".join(plain_sentences)
    topics = extract_topics(full_text)[:6]
    overall_sent = Counter(s["sentiment"] for s in classified).most_common(1)
    overall_sent = overall_sent[0][0] if overall_sent else "Neutral"
    _, sent_color = get_sentiment(" ".join(s["text"] for s in classified if s["sentiment"] == overall_sent)[:100] or "neutral")

    summary = extractive_summary(full_text)
    if models and models.get("summarizer") and len(full_text.split()) > 50:
        try:
            out = models["summarizer"](full_text[:1024], max_length=120, min_length=30, do_sample=False)
            summary = out[0]["summary_text"]
        except Exception:
            pass

    temporal = temporal_normalize(full_text)

    # Feature 33: meeting tags
    tags = meeting_tags or []

    # Feature 35: speaker profiles (update)
    for turn in speakers_raw:
        sp = turn["speaker"]
        if sp not in st.session_state.speaker_profiles:
            st.session_state.speaker_profiles[sp] = {"sessions": 0, "total_words": 0, "intents": Counter()}
        st.session_state.speaker_profiles[sp]["sessions"] += 1
        st.session_state.speaker_profiles[sp]["total_words"] += len(turn["text"].split())

    return {
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "transcript": transcript,
        "sentences": classified,
        "speakers": speakers_raw,
        "word_count": len(full_text.split()),
        "summary": summary,
        "topics": topics,
        "sentiment": overall_sent,
        "sent_color": sent_color,
        "temporal": temporal,
        "tags": tags,
        "alerts": [],
    }

def check_alerts(record: dict) -> list:
    alerts = []
    risk_count  = sum(1 for s in record["sentences"] if s["intent"] == "risk or issue")
    action_count = sum(1 for s in record["sentences"] if s["intent"] == "action item")
    if risk_count >= 3:
        alerts.append(("High Risk Detected", f"{risk_count} risk/issue sentences found. Review immediately.", "#f87171"))
    if action_count >= 5:
        alerts.append(("High Action Volume", f"{action_count} action items detected — assign owners promptly.", "#fbbf24"))
    if record["sentiment"] == "Negative":
        alerts.append(("Negative Meeting Tone", "Overall sentiment was negative. Consider a follow-up.", "#c084fc"))
    return alerts

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def export_json(record: dict) -> str:
    r = dict(record)
    return json.dumps(r, indent=2, default=str)

def export_csv(record: dict) -> str:
    rows = []
    for s in record["sentences"]:
        rows.append({
            "text": s["text"], "intent": s["intent"],
            "confidence": s["confidence"], "sentiment": s["sentiment"],
            "priority": s["priority"],
            "entities": "; ".join(f"{e['type']}:{e['text']}" for e in s["entities"]),
        })
    return pd.DataFrame(rows).to_csv(index=False)

def export_markdown(record: dict) -> str:
    lines = [
        "# NLU Meeting Intelligence Report",
        f"*Generated: {record['ts']}*",
        f"*Words: {record['word_count']} | Sentiment: {record['sentiment']}*",
        "",
        "## Executive Summary",
        record["summary"],
        "",
        "## Key Topics",
        " ".join(f"`{t}`" for t in record["topics"]),
        "",
        "## Classified Sentences",
    ]
    for s in record["sentences"]:
        lines.append(f"- **[{s['intent'].upper()}]** `{s['confidence']:.2f}` {s['text']}")
    return "\n".join(lines)

def export_pdf(record: dict) -> bytes:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "NLU Meeting Intelligence Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Generated: {record['ts']}  |  Words: {record['word_count']}  |  Sentiment: {record['sentiment']}", ln=True)
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Executive Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, record["summary"])
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Classified Sentences", ln=True)
        pdf.set_font("Helvetica", "", 8)
        for s in record["sentences"][:40]:
            line = f"[{s['intent'].upper()}] ({s['confidence']:.2f}) {s['text'][:110]}"
            pdf.multi_cell(0, 4.5, line)
        return pdf.output(dest="S").encode("latin-1", errors="replace")
    except Exception as e:
        return f"PDF export error: {e}".encode()

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def badge_html(intent: str) -> str:
    c = COLORS.get(intent, "#64748b")
    label = INTENT_LABELS.get(intent, intent)
    return f'<span class="badge" style="color:{c};border-color:{c}30;background:{c}12">{label}</span>'

def conf_bar(conf: float, color: str) -> str:
    pct = int(conf * 100)
    return f"""
    <div class="conf-bar-bg">
      <div class="conf-bar-fg" style="width:{pct}%;background:{color}"></div>
    </div>"""

def make_wordcloud(text: str):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(
            width=1000, height=380, background_color="#0f172a",
            colormap="cool", max_words=90, collocations=False,
            prefer_horizontal=0.8,
        ).generate(text or "meeting")
        fig, ax = plt.subplots(figsize=(10, 3.8), facecolor="#0f172a")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = BytesIO()
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", dpi=130, facecolor="#0f172a")
        buf.seek(0)
        plt.close()
        return buf
    except Exception:
        return None

def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    model_size = st.session_state.get("whisper_size", "large-v3")
    model = load_whisper(model_size)
    if model is None:
        return "(Whisper not available — paste transcript manually)"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        fname = f.name
    try:
        result = model.transcribe(fname, verbose=False)
        return result["text"].strip()
    except Exception as e:
        return f"Transcription error: {e}"
    finally:
        os.unlink(fname)

def plotly_dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        margin=dict(t=15, b=15, l=10, r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.1rem 1rem;background:linear-gradient(135deg,rgba(56,189,248,.12),rgba(192,132,252,.07));
         border-radius:16px;margin-bottom:1.4rem;border:1px solid rgba(56,189,248,.18)">
      <div style="font-size:1.25rem;font-weight:800;letter-spacing:-.03em;color:#e2e8f0">
        Voice<span style="color:#38bdf8">IQ</span>
      </div>
      <div style="font-size:.68rem;color:#475569;margin-top:.25rem;font-weight:600;
           letter-spacing:.1em;text-transform:uppercase">
        Meeting Intelligence · v3.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["Voice & Analyze", "Analytics Dashboard", "Intent Explorer",
         "Speaker Diarization", "Entity Map", "Topic Analysis",
         "Action Tracker", "Sentiment Analysis", "Compare Sessions",
         "Meeting Q&A", "Timeline View", "Smart Alerts",
         "Benchmark", "Session History", "Speaker Profiles",
         "Transcript Diff", "Keyword Search", "Settings", "About"],
        label_visibility="collapsed",
    )

    st.divider()

    # Model status
    model_ok = st.session_state.model_loaded
    whisper_ok = st.session_state.whisper_model is not None
    st.markdown(f"""
    <div style="font-size:.7rem;margin-bottom:.8rem">
      <div style="margin-bottom:.3rem">
        <span class="status-dot {'ok' if model_ok else 'warn'}"></span>
        <span style="color:#64748b;font-weight:600">NLP Models</span>
        <span style="color:{'#34d399' if model_ok else '#fbbf24'};margin-left:.4rem;font-size:.65rem">
          {'Active' if model_ok else 'Not loaded'}
        </span>
      </div>
      <div>
        <span class="status-dot {'ok' if whisper_ok else 'warn'}"></span>
        <span style="color:#64748b;font-weight:600">Whisper STT</span>
        <span style="color:{'#34d399' if whisper_ok else '#fbbf24'};margin-left:.4rem;font-size:.65rem">
          {'Active' if whisper_ok else 'Not loaded'}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:.7rem;color:#475569;line-height:1.9">
      Sessions: <b style="color:#e2e8f0">{len(st.session_state.history)}</b> &nbsp;&nbsp;
      Actions: <b style="color:#e2e8f0">{len(st.session_state.action_items)}</b> &nbsp;&nbsp;
      Alerts: <b style="color:#f87171">{len(st.session_state.alerts)}</b>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown(f"<div style='font-size:.67rem;color:#475569;margin-top:.3rem'>Last: {st.session_state.history[0]['ts']}</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("Load NLP Models", use_container_width=True):
        with st.spinner("Loading models (first time may take a minute)…"):
            try:
                load_nlp_models()
                st.session_state.model_loaded = True
                st.success("Models loaded")
            except Exception as e:
                st.warning(f"Partial load: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — VOICE & ANALYZE
# ─────────────────────────────────────────────────────────────────────────────
if "Voice" in nav:
    st.markdown('<div class="page-title">Voice & <span>Transcript</span> Analyzer</div>', unsafe_allow_html=True)

    mode_tab, upload_tab, text_tab = st.tabs(["Live Recording", "Upload Audio", "Paste Text"])
    transcript_to_analyze = ""

    # Tab 1: Live mic
    with mode_tab:
        st.markdown("""
        <div class="upload-zone">
          <div class="upload-icon">
            <svg width="24" height="24" fill="none" stroke="#38bdf8" stroke-width="2"
                 viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <rect x="9" y="2" width="6" height="13" rx="3"/>
              <path d="M5 10a7 7 0 0014 0M12 19v3M9 22h6" stroke-linecap="round"/>
            </svg>
          </div>
          <div style="font-size:1rem;font-weight:700;margin-bottom:.3rem;color:#e2e8f0">
            Live Microphone Recording
          </div>
          <div style="color:#475569;font-size:.82rem">
            Click the recorder control below, speak your meeting transcript, then transcribe
          </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                st.audio(wav_audio_data, format="audio/wav")
                if st.button("Transcribe Recording", key="transcribe_mic"):
                    with st.spinner(f"Whisper {st.session_state.whisper_size} transcribing…"):
                        transcript_to_analyze = transcribe_audio(wav_audio_data, ".wav")
                    st.success("Transcription complete")
                    st.session_state["pending_transcript"] = transcript_to_analyze
        except ImportError:
            st.info("Install `st-audiorec` for live recording. Use Upload or Paste text instead.")

    # Tab 2: Upload audio file
    with upload_tab:
        # Whisper model selector
        c_model, c_info = st.columns([2,1])
        with c_model:
            model_choice = st.selectbox(
                "Whisper Model",
                list(WHISPER_MODELS.keys()),
                index=0,
                help="larger = more accurate but slower & heavier"
            )
            st.session_state.whisper_size = WHISPER_MODELS[model_choice]
        with c_info:
            st.markdown(f"""
            <div style="margin-top:1.8rem">
              <span class="model-pill">
                <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                </svg>
                {st.session_state.whisper_size}
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="upload-zone" style="margin-top:.8rem">
          <div class="upload-icon">
            <svg width="24" height="24" fill="none" stroke="#38bdf8" stroke-width="2"
                 viewBox="0 0 24 24">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <div style="font-size:1rem;font-weight:700;margin-bottom:.3rem;color:#e2e8f0">
            Upload Audio File
          </div>
          <div style="color:#475569;font-size:.82rem">
            WAV · MP3 · M4A · OGG · FLAC · WEBM · AAC · OPUS
          </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_audio = st.file_uploader(
            "Audio file",
            type=["wav","mp3","m4a","ogg","flac","webm","aac","opus"],
            label_visibility="collapsed",
        )
        if uploaded_audio:
            st.audio(uploaded_audio)
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("File", uploaded_audio.name[:20])
            fc2.metric("Size", f"{uploaded_audio.size/1024:.1f} KB")
            fc3.metric("Format", uploaded_audio.name.split(".")[-1].upper())
            if st.button("Transcribe Audio", key="transcribe_file"):
                suffix = "." + uploaded_audio.name.split(".")[-1]
                with st.spinner(f"Whisper {st.session_state.whisper_size} processing…"):
                    transcript_to_analyze = transcribe_audio(uploaded_audio.read(), suffix)
                st.success("Transcription complete")
                st.session_state["pending_transcript"] = transcript_to_analyze
                st.markdown(f"""
                <div class="card">
                  <div class="section-label">Transcript Output</div>
                  <div style="font-size:.88rem;line-height:1.75;color:#cbd5e1">{transcript_to_analyze}</div>
                </div>""", unsafe_allow_html=True)

    # Tab 3: Paste text
    with text_tab:
        sample = """Alice: Good morning team! Let's kick off our sprint review.
John: We completed the user authentication module. The database migration is 90% done, finishing Thursday.
Sarah: I finalized the new dashboard mockups. I'll share them by end of day.
Alice: What about the security audit?
John: Mike is leading that. We identified three vulnerabilities.
Mike: I'll send a detailed report to all stakeholders by tomorrow morning.
Alice: We need to schedule a follow-up with the security team next week. Sarah, can you set that up?
Sarah: Absolutely. I'll send the calendar invite today.
Alice: We also need to finalize the Q4 budget proposal before the board meeting on December 5th. There is a risk of missing the deadline if we don't start now.
John: I'll meet with the CFO this week.
Alice: Great. Any blockers?
Mike: No blockers from my side.
Alice: Perfect. Thanks everyone."""
        manual_text = st.text_area(
            "Meeting Transcript",
            height=240,
            value=st.session_state.get("pending_transcript", ""),
            placeholder=sample,
        )
        if not manual_text.strip():
            if st.button("Load Demo Transcript"):
                st.session_state["pending_transcript"] = sample
                st.rerun()
        if manual_text.strip():
            transcript_to_analyze = manual_text

    if st.session_state.get("pending_transcript") and not transcript_to_analyze:
        transcript_to_analyze = st.session_state["pending_transcript"]

    # Analysis options
    with st.expander("Analysis Options"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown('<div class="section-label">Filter Intents</div>', unsafe_allow_html=True)
            selected_intents = [i for i in INTENTS if st.checkbox(i.capitalize(), True, key=f"fi_{i}")]
        with col_f2:
            st.markdown('<div class="section-label">Speaker Detection</div>', unsafe_allow_html=True)
            enable_diarization = st.toggle("Auto-detect speakers", True)
        with col_f3:
            st.markdown('<div class="section-label">Sort Options</div>', unsafe_allow_html=True)
            sort_by_priority = st.toggle("Sort by priority", False)
            st.session_state.highlight_entities = st.toggle("Highlight entities", st.session_state.highlight_entities)

    # Feature 33: Meeting tags
    with st.expander("Meeting Tags"):
        tag_input = st.text_input("Add tags (comma-separated)", placeholder="sprint-review, q4, engineering")
        current_tags = [t.strip() for t in tag_input.split(",") if t.strip()] if tag_input else []
        if current_tags:
            st.markdown(" ".join(
                f'<span style="background:rgba(56,189,248,.1);color:#38bdf8;border-radius:6px;padding:.2rem .6rem;font-size:.7rem;font-weight:600;border:1px solid rgba(56,189,248,.2);margin-right:.3rem">{t}</span>'
                for t in current_tags
            ), unsafe_allow_html=True)

    run_col, clear_col = st.columns([3, 1])
    with run_col:
        run_btn = st.button("Run NLU Pipeline", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state["pending_transcript"] = ""
            st.rerun()

    # Pipeline execution
    if run_btn and transcript_to_analyze.strip():
        models = None
        if st.session_state.model_loaded:
            try: models = load_nlp_models()
            except: pass

        with st.spinner("Running NLU pipeline…"):
            prog = st.progress(0)
            stages = ["Tokenizing", "Classifying intents", "Extracting entities", "Summarizing", "Building graph"]
            for i, stage in enumerate(stages):
                time.sleep(0.1); prog.progress((i+1)*20)
            record = analyze_transcript(transcript_to_analyze, selected_intents, models, current_tags)
            if sort_by_priority:
                record["sentences"].sort(key=lambda x: x["priority"], reverse=True)
            alerts = check_alerts(record)
            record["alerts"] = alerts
            st.session_state.history.insert(0, record)

            # Feature 31: store raw transcript
            st.session_state.transcript_history.insert(0, {
                "ts": record["ts"],
                "text": transcript_to_analyze,
                "word_count": record["word_count"],
            })

            for s in record["sentences"]:
                if s["intent"] == "action item":
                    if s["text"] not in [a["text"] for a in st.session_state.action_items]:
                        st.session_state.action_items.append({
                            "text": s["text"], "done": False,
                            "ts": record["ts"], "owner": "",
                            "due": ""
                        })
            if alerts:
                st.session_state.alerts = alerts + st.session_state.alerts

        st.success(f"Processed {record['word_count']} words — {len(record['sentences'])} classified sentences")

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.history:
        record = st.session_state.history[0]

        # Alerts
        if record.get("alerts"):
            for title, msg, color in record["alerts"]:
                st.markdown(f"""
                <div class="alert-strip" style="border-color:{color}40;border-left-color:{color};
                     background:rgba(255,255,255,.02)">
                  <div>
                    <div style="font-weight:700;color:{color};font-size:.85rem;margin-bottom:.2rem">{title}</div>
                    <div style="color:#94a3b8;font-size:.78rem">{msg}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Summary card
        sent_c = {"Positive":"#34d399","Neutral":"#fbbf24","Negative":"#f87171"}.get(record["sentiment"],"#64748b")
        tags_html = " ".join(
            f'<span style="background:rgba(56,189,248,.08);color:#38bdf8;border-radius:5px;padding:.15rem .5rem;font-size:.65rem;font-weight:700;letter-spacing:.05em;border:1px solid rgba(56,189,248,.15)">{t}</span>'
            for t in record.get("tags",[])
        )
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.8rem">
            <div>
              <div class="section-label">Executive Summary — {record['ts']}</div>
              {tags_html}
            </div>
            <span style="font-size:.78rem;color:{sent_c};font-weight:700;
                  background:{sent_c}15;padding:.2rem .7rem;border-radius:6px;border:1px solid {sent_c}30">
              {record['sentiment']}
            </span>
          </div>
          <div style="font-size:.92rem;line-height:1.8;color:#cbd5e1">{record['summary']}</div>
          <div style="margin-top:.9rem;display:flex;gap:.35rem;flex-wrap:wrap">
            {"".join(f'<span style="background:rgba(255,255,255,.05);border-radius:6px;padding:.18rem .55rem;font-size:.68rem;color:#64748b;border:1px solid rgba(255,255,255,.06)">{t}</span>' for t in record["topics"])}
          </div>
        </div>""", unsafe_allow_html=True)

        # Metric boxes
        counts = Counter(s["intent"] for s in record["sentences"])
        cols = st.columns(6)
        for col, intent in zip(cols, INTENTS):
            with col:
                c = COLORS[intent]
                st.markdown(f"""
                <div class="metric">
                  <div class="val" style="color:{c}">{counts.get(intent,0)}</div>
                  <div class="lbl">{INTENT_LABELS[intent]}</div>
                </div>""", unsafe_allow_html=True)

        # Temporal
        if record.get("temporal"):
            st.markdown('<div class="section-label" style="margin-top:1rem">Temporal References</div>', unsafe_allow_html=True)
            tcols = st.columns(min(len(record["temporal"]), 4))
            for col, t in zip(tcols, record["temporal"][:4]):
                with col:
                    st.markdown(f"""
                    <div class="metric">
                      <div style="font-size:.95rem;font-weight:700;color:#fbbf24;font-family:'DM Mono',monospace">{t['date']}</div>
                      <div class="lbl">{t['expression']}</div>
                    </div>""", unsafe_allow_html=True)

        # Classified sentences
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Classified Sentences</div>', unsafe_allow_html=True)
        for s in record["sentences"]:
            conf_c = "#34d399" if s["confidence"] > .85 else "#fbbf24" if s["confidence"] > .7 else "#f87171"
            intent_c = COLORS.get(s["intent"], "#64748b")
            ent_html = ""
            if st.session_state.highlight_entities and s["entities"]:
                ecols = {"PERSON":"#60a5fa","DATE":"#34d399","TECH":"#fbbf24","ORG":"#c084fc"}
                ent_html = " ".join(
                    f'<span style="background:rgba(255,255,255,.05);border-radius:4px;padding:.1rem .4rem;'
                    f'font-size:.65rem;color:{ecols.get(e["type"],"#94a3b8")};font-weight:600;border:1px solid rgba(255,255,255,.06)">'
                    f'{e["text"]}</span>'
                    for e in s["entities"][:4]
                )
            st.markdown(f"""
            <div class="sent-card" style="border-left-color:{intent_c}">
              <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.4rem">
                {badge_html(s['intent'])}
                <span style="font-size:.66rem;color:{conf_c};font-family:'DM Mono',monospace;font-weight:600">
                  {s['confidence']:.2f}
                </span>
                <span style="font-size:.66rem;color:{s['sent_color']};font-weight:600">{s['sentiment']}</span>
                {ent_html}
              </div>
              <div style="font-size:.87rem;color:#cbd5e1;line-height:1.7">{s['text']}</div>
              {conf_bar(s['confidence'], conf_c)}
            </div>""", unsafe_allow_html=True)

        # Export
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Export Results</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        with e1: st.download_button("JSON",  export_json(record),  "nlu.json", "application/json")
        with e2: st.download_button("CSV",   export_csv(record),   "nlu.csv",  "text/csv")
        with e3: st.download_button("PDF",   export_pdf(record),   "nlu.pdf",  "application/pdf")
        with e4: st.download_button("Markdown", export_markdown(record), "nlu.md", "text/markdown")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif "Analytics" in nav:
    st.markdown('<div class="page-title">Analytics <span>Dashboard</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("Run a session first."); st.stop()

    df = pd.DataFrame([{
        "intent": s["intent"], "confidence": s["confidence"],
        "sentiment": s["sentiment"], "priority": s["priority"],
        "session": f"S{i+1}"
    } for i, h in enumerate(reversed(st.session_state.history)) for s in h["sentences"]])

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="section-label">Intent Distribution</div>', unsafe_allow_html=True)
        cnt = df["intent"].value_counts().reset_index(); cnt.columns = ["intent","count"]
        fig = px.pie(cnt, names="intent", values="count", hole=.5,
                     color="intent", color_discrete_map=COLORS)
        fig.update_traces(textfont_size=11, marker=dict(line=dict(color="#0a0f1e",width=2)))
        fig.update_layout(**plotly_dark_layout(), showlegend=True,
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-label">Sentiment Distribution</div>', unsafe_allow_html=True)
        s_cnt = df["sentiment"].value_counts().reset_index(); s_cnt.columns = ["sentiment","count"]
        scols = {"Positive":"#34d399","Neutral":"#fbbf24","Negative":"#f87171"}
        fig2 = px.pie(s_cnt, names="sentiment", values="count", hole=.5,
                      color="sentiment", color_discrete_map=scols)
        fig2.update_traces(textfont_size=11, marker=dict(line=dict(color="#0a0f1e",width=2)))
        fig2.update_layout(**plotly_dark_layout())
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-label">Intent Trend Across Sessions</div>', unsafe_allow_html=True)
    pivot = df.groupby(["session","intent"]).size().reset_index(name="count")
    fig3 = px.bar(pivot, x="session", y="count", color="intent", barmode="stack", color_discrete_map=COLORS)
    fig3.update_layout(**plotly_dark_layout())
    st.plotly_chart(fig3, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown('<div class="section-label">Avg Confidence per Intent</div>', unsafe_allow_html=True)
        avg = df.groupby("intent")["confidence"].mean().reset_index()
        fig4 = px.bar(avg, x="intent", y="confidence", color="intent",
                      color_discrete_map=COLORS, range_y=[0,1])
        fig4.update_layout(**plotly_dark_layout(), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-label">Priority Score Distribution</div>', unsafe_allow_html=True)
        fig5 = px.histogram(df, x="priority", nbins=4, color="intent",
                             color_discrete_map=COLORS, barmode="overlay", opacity=.75)
        fig5.update_layout(**plotly_dark_layout())
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — INTENT EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif "Intent Explorer" in nav:
    st.markdown('<div class="page-title">Intent <span>Explorer</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_s = [s for h in st.session_state.history for s in h["sentences"]]
    c1, c2, c3 = st.columns(3)
    with c1: sel = st.multiselect("Intents", INTENTS, INTENTS)
    with c2: conf_r = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), .01)
    with c3: sent_f = st.multiselect("Sentiment", ["Positive","Neutral","Negative"], ["Positive","Neutral","Negative"])
    filtered = [s for s in all_s if s["intent"] in sel and conf_r[0] <= s["confidence"] <= conf_r[1] and s["sentiment"] in sent_f]
    df_v = pd.DataFrame([{
        "Intent": s["intent"], "Confidence": s["confidence"],
        "Sentiment": s["sentiment"], "Priority": s["priority"],
        "Sentence": s["text"],
    } for s in filtered])
    st.dataframe(df_v, use_container_width=True, height=460)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — SPEAKER DIARIZATION
# ─────────────────────────────────────────────────────────────────────────────
elif "Speaker Diarization" in nav:
    st.markdown('<div class="page-title">Speaker <span>Diarization</span></div>', unsafe_allow_html=True)
    st.caption("Format: `Name: text` — one turn per line — for automatic speaker detection.")
    if not st.session_state.history: st.info("Run a session first."); st.stop()
    record = st.session_state.history[0]
    turns = record.get("speakers", [])
    if not turns:
        st.warning("No speaker labels found. Use `Alice: We decided to use React.` format.")
    else:
        speakers = list({t["speaker"] for t in turns})
        spk_color = {s: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, s in enumerate(speakers)}

        st.markdown('<div class="section-label">Speaker Timeline</div>', unsafe_allow_html=True)
        for t in turns:
            c = spk_color.get(t["speaker"], "#94a3b8")
            label, conf = classify_sentence(t["text"])
            st.markdown(f"""
            <div class="tl-item">
              <div class="tl-dot" style="background:{c}"></div>
              <div>
                <span class="spk-tag" style="color:{c};border-color:{c}30">{t['speaker']}</span>
                {badge_html(label)}
                <span style="font-size:.83rem;color:#cbd5e1;margin-left:.5rem">{t['text'][:95]}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.2rem">Speaker Statistics</div>', unsafe_allow_html=True)
        spk_counts = Counter(t["speaker"] for t in turns)
        spk_df = pd.DataFrame(list(spk_counts.items()), columns=["Speaker","Turns"])
        spk_df["Words"] = [sum(len(t["text"].split()) for t in turns if t["speaker"] == s) for s in spk_df["Speaker"]]
        fig = px.bar(spk_df, x="Speaker", y="Turns", color="Speaker",
                     color_discrete_sequence=SPEAKER_COLORS, text="Words")
        fig.update_layout(**plotly_dark_layout(), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ENTITY MAP
# ─────────────────────────────────────────────────────────────────────────────
elif "Entity" in nav:
    st.markdown('<div class="page-title">Entity <span>Extraction Map</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_ents = [{"text": e["text"],"type": e["type"],"intent": s["intent"],"sentence": s["text"][:55]+"…"}
                for h in st.session_state.history for s in h["sentences"] for e in s["entities"]]
    if not all_ents: st.info("No entities extracted yet."); st.stop()
    df_e = pd.DataFrame(all_ents)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="section-label">By Type</div>', unsafe_allow_html=True)
        ecol = {"PERSON":"#60a5fa","DATE":"#34d399","TECH":"#fbbf24","ORG":"#c084fc"}
        for et, cnt in df_e["type"].value_counts().items():
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.45rem .9rem;
              background:var(--surface2);border-radius:9px;margin-bottom:.4rem;
              border:1px solid var(--border2)">
              <span style="color:{ecol.get(et,'#94a3b8')};font-weight:600;font-size:.83rem">{et}</span>
              <span style="font-weight:700;font-size:.83rem">{cnt}</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-label">Top Entities</div>', unsafe_allow_html=True)
        freq = df_e["text"].value_counts().head(15).reset_index()
        freq.columns = ["entity","count"]
        fig = px.bar(freq, x="count", y="entity", orientation="h",
                     color="count", color_continuous_scale=[[0,"#1e293b"],[1,"#38bdf8"]])
        fig.update_layout(**plotly_dark_layout(), yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_e[["text","type","intent","sentence"]], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — TOPIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif "Topic" in nav:
    st.markdown('<div class="page-title">Topic <span>Analysis</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_text = " ".join(s["text"] for h in st.session_state.history for s in h["sentences"])

    st.markdown('<div class="section-label">Word Cloud</div>', unsafe_allow_html=True)
    wc_img = make_wordcloud(all_text)
    if wc_img:
        st.markdown('<div class="wc-container">', unsafe_allow_html=True)
        st.image(wc_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Install `wordcloud` for word cloud visualization.")

    all_topics = [t for h in st.session_state.history for t in h.get("topics", [])]
    if all_topics:
        tf = Counter(all_topics).most_common(25)
        tdf = pd.DataFrame(tf, columns=["topic","freq"])
        st.markdown('<div class="section-label" style="margin-top:1rem">Topic Treemap</div>', unsafe_allow_html=True)
        fig = px.treemap(tdf, path=["topic"], values="freq", color="freq",
                         color_continuous_scale=[[0,"#1e293b"],[0.5,"#0ea5e9"],[1,"#38bdf8"]])
        fig.update_layout(**plotly_dark_layout())
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — ACTION TRACKER
# ─────────────────────────────────────────────────────────────────────────────
elif "Action Tracker" in nav:
    st.markdown('<div class="page-title">Action <span>Item Tracker</span></div>', unsafe_allow_html=True)

    with st.expander("Add Manual Action Item"):
        ac1, ac2, ac3 = st.columns([3,2,1])
        with ac1: new_action = st.text_input("Action description", placeholder="Describe the action item…")
        with ac2: new_owner  = st.text_input("Owner", placeholder="Assignee name")
        with ac3: new_due    = st.text_input("Due date", placeholder="YYYY-MM-DD")
        if st.button("Add Item") and new_action.strip():
            st.session_state.action_items.append({
                "text": new_action, "done": False,
                "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "owner": new_owner, "due": new_due,
            })
            st.rerun()

    if not st.session_state.action_items:
        st.info("No action items yet. Run an analysis session first."); st.stop()

    done_count = sum(1 for a in st.session_state.action_items if a["done"])
    total = len(st.session_state.action_items)
    pct = done_count / total if total else 0
    st.progress(pct)
    st.markdown(f"""<div style="font-size:.78rem;color:#64748b;margin-bottom:1rem">
      <b style="color:#e2e8f0">{done_count}/{total}</b> completed &nbsp;·&nbsp;
      <b style="color:#38bdf8">{total-done_count}</b> remaining
    </div>""", unsafe_allow_html=True)

    for i, item in enumerate(st.session_state.action_items):
        c1, c2, c3 = st.columns([.06, .8, .14])
        with c1:
            checked = st.checkbox("", value=item["done"], key=f"done_{i}")
            st.session_state.action_items[i]["done"] = checked
        with c2:
            style = "text-decoration:line-through;color:#475569;" if item["done"] else "color:#e2e8f0;"
            owner_html = f'<span style="font-size:.67rem;color:#38bdf8;font-weight:600;margin-left:.5rem">{item.get("owner","")}</span>' if item.get("owner") else ""
            due_html = f'<span style="font-size:.67rem;color:#fbbf24;margin-left:.4rem">{item.get("due","")}</span>' if item.get("due") else ""
            dot_class = "done" if item["done"] else ""
            st.markdown(f"""
            <div class="action-row">
              <div class="action-dot {dot_class}"></div>
              <div style="flex:1">
                <div style="{style}font-size:.87rem">{item['text']}</div>
                <div style="margin-top:.2rem">
                  <span style="font-size:.67rem;color:#475569">{item['ts']}</span>
                  {owner_html}{due_html}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            if st.button("Remove", key=f"del_{i}"):
                st.session_state.action_items.pop(i); st.rerun()

    st.divider()
    adf = pd.DataFrame(st.session_state.action_items)
    st.download_button("Export CSV", adf.to_csv(index=False), "action_items.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif "Sentiment" in nav:
    st.markdown('<div class="page-title">Sentiment <span>Analysis</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No data yet."); st.stop()

    sent_data = [{
        "Session": f"S{i+1}", "Sentiment": h["sentiment"], "Color": h["sent_color"],
        "Positive": sum(1 for s in h["sentences"] if s["sentiment"] == "Positive"),
        "Neutral":  sum(1 for s in h["sentences"] if s["sentiment"] == "Neutral"),
        "Negative": sum(1 for s in h["sentences"] if s["sentiment"] == "Negative"),
    } for i, h in enumerate(reversed(st.session_state.history))]
    sdf = pd.DataFrame(sent_data)

    last = st.session_state.history[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Sentiment", last["sentiment"])
    c2.metric("Positive Sentences", sum(1 for s in last["sentences"] if s["sentiment"] == "Positive"))
    c3.metric("Negative Sentences", sum(1 for s in last["sentences"] if s["sentiment"] == "Negative"))

    st.markdown('<div class="section-label" style="margin-top:.8rem">Sentiment per Session</div>', unsafe_allow_html=True)
    fig = px.bar(sdf, x="Session", y=["Positive","Neutral","Negative"], barmode="stack",
                 color_discrete_map={"Positive":"#34d399","Neutral":"#fbbf24","Negative":"#f87171"})
    fig.update_layout(**plotly_dark_layout())
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Sentence-level (Latest Session)</div>', unsafe_allow_html=True)
    for s in last["sentences"]:
        st.markdown(f"""
        <div style="display:flex;gap:.7rem;align-items:center;padding:.45rem 0;
             border-bottom:1px solid rgba(255,255,255,.04)">
          <span style="font-size:.7rem;color:{s['sent_color']};font-weight:700;min-width:72px;
                font-family:'DM Mono',monospace">{s['sentiment']}</span>
          <span style="font-size:.84rem;color:#cbd5e1">{s['text']}</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 9 — COMPARE SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
elif "Compare" in nav:
    st.markdown('<div class="page-title">Session <span>Comparison</span></div>', unsafe_allow_html=True)
    if len(st.session_state.history) < 2: st.info("Need at least 2 sessions."); st.stop()
    options = [f"S{i+1} — {h['ts']}" for i, h in enumerate(st.session_state.history)]
    col_a, col_b = st.columns(2)
    with col_a: idx_a = st.selectbox("Session A", range(len(options)), format_func=lambda x: options[x], key="cmp_a")
    with col_b: idx_b = st.selectbox("Session B", range(len(options)), format_func=lambda x: options[x], index=min(1, len(options)-1), key="cmp_b")
    ra = st.session_state.history[idx_a]
    rb = st.session_state.history[idx_b]
    ca, cb = st.columns(2)

    def session_card(r, label):
        cnts = Counter(s["intent"] for s in r["sentences"])
        sent_c = {"Positive":"#34d399","Neutral":"#fbbf24","Negative":"#f87171"}.get(r["sentiment"],"#64748b")
        st.markdown(f'<div class="section-label">{label} — {r["ts"]}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card">
          <div style="font-size:.83rem;color:#94a3b8;margin-bottom:.5rem">{r['summary'][:200]}</div>
          <span style="font-size:.75rem;color:{sent_c};font-weight:700;background:{sent_c}15;
                padding:.2rem .7rem;border-radius:6px;border:1px solid {sent_c}30">{r['sentiment']}</span>
        </div>""", unsafe_allow_html=True)
        for intent in INTENTS:
            c = COLORS[intent]
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.35rem .8rem;
              background:var(--surface2);border-radius:8px;margin-bottom:.3rem;
              border:1px solid var(--border2);font-size:.8rem">
              <span style="color:#94a3b8">{INTENT_LABELS[intent]}</span>
              <span style="color:{c};font-weight:700">{cnts.get(intent,0)}</span>
            </div>""", unsafe_allow_html=True)

    with ca: session_card(ra, "Session A")
    with cb: session_card(rb, "Session B")

    st.markdown('<div class="section-label" style="margin-top:1rem">Intent Delta (B minus A)</div>', unsafe_allow_html=True)
    ca_cnt = Counter(s["intent"] for s in ra["sentences"])
    cb_cnt = Counter(s["intent"] for s in rb["sentences"])
    diff_data = [{"intent": i, "A": ca_cnt.get(i,0), "B": cb_cnt.get(i,0), "diff": cb_cnt.get(i,0)-ca_cnt.get(i,0)} for i in INTENTS]
    ddf = pd.DataFrame(diff_data)
    fig = px.bar(ddf, x="intent", y="diff", color="diff",
                 color_continuous_scale=[[0,"#f87171"],[0.5,"#1e293b"],[1,"#34d399"]])
    fig.update_layout(**plotly_dark_layout())
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 10 — Q&A CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
elif "Q&A" in nav:
    st.markdown('<div class="page-title">Meeting <span>Q&A</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("Run a session first."); st.stop()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about the meeting…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        record = st.session_state.history[0]
        p = prompt.lower()
        if any(w in p for w in ["action","task","todo","do"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="action item"]
            answer = "**Action items:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        elif any(w in p for w in ["decision","decided","agree","chose"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="decision"]
            answer = "**Decisions made:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        elif any(w in p for w in ["risk","issue","problem","block","concern"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="risk or issue"]
            answer = "**Risks & issues:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        elif any(w in p for w in ["summary","summarize","overview","brief"]):
            answer = f"**Summary:**\n\n{record['summary']}"
        elif any(w in p for w in ["topic","keyword","subject","about"]):
            answer = f"**Main topics:** {', '.join(record['topics'])}"
        elif any(w in p for w in ["sentiment","tone","mood"]):
            answer = f"**Sentiment:** {record['sentiment']}"
        elif any(w in p for w in ["who","speaker","person","attend"]):
            speakers = list({t["speaker"] for t in record.get("speakers",[])})
            answer = "**Speakers:** " + (", ".join(speakers) if speakers else "No speaker labels found.")
        elif any(w in p for w in ["suggest","idea","recommend"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="suggestion"]
            answer = "**Suggestions:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        else:
            hits = [s["text"] for s in record["sentences"] if any(w in s["text"].lower() for w in p.split() if len(w) > 3)]
            answer = ("**Relevant sentences:**\n\n" + "\n".join(f"- {h}" for h in hits[:5])) if hits else f"Topics covered: {', '.join(record['topics'][:5])}."
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)
    if st.button("Clear Chat"): st.session_state.chat_history = []; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 11 — TIMELINE VIEW
# ─────────────────────────────────────────────────────────────────────────────
elif "Timeline" in nav:
    st.markdown('<div class="page-title">Meeting <span>Timeline</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No data yet."); st.stop()
    record = st.session_state.history[0]

    if record.get("temporal"):
        st.markdown('<div class="section-label">Temporal References</div>', unsafe_allow_html=True)
        for t in record["temporal"]:
            st.markdown(f"""
            <div class="card-sm" style="display:flex;justify-content:space-between;align-items:center">
              <span style="color:#fbbf24;font-weight:700;font-family:'DM Mono',monospace;font-size:.88rem">{t['date']}</span>
              <span style="color:#64748b;font-size:.8rem">"{t['expression']}"</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No temporal references detected in the latest session.")

    st.markdown('<div class="section-label" style="margin-top:1rem">Intent Timeline</div>', unsafe_allow_html=True)
    for i, s in enumerate(record["sentences"], 1):
        c = COLORS.get(s["intent"], "#64748b")
        st.markdown(f"""
        <div class="tl-item">
          <div class="tl-dot" style="background:{c}"></div>
          <div style="flex:1">
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.2rem">
              {badge_html(s['intent'])}
              <span style="font-size:.65rem;color:#475569;font-family:'DM Mono',monospace">#{i:03d}</span>
            </div>
            <div style="font-size:.82rem;color:#cbd5e1">{s['text'][:110]}</div>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 12 — SMART ALERTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Smart Alerts" in nav:
    st.markdown('<div class="page-title">Smart <span>Alerts</span></div>', unsafe_allow_html=True)
    if not st.session_state.alerts:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem">
          <div style="font-size:2rem;margin-bottom:.5rem;opacity:.3">
            <svg width="40" height="40" fill="none" stroke="#34d399" stroke-width="2" viewBox="0 0 24 24" style="display:inline-block">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
          <div style="color:#34d399;font-weight:700;font-size:1rem">All Clear</div>
          <div style="color:#475569;font-size:.82rem;margin-top:.3rem">No alerts detected</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    for title, msg, color in st.session_state.alerts:
        st.markdown(f"""
        <div class="alert-strip" style="border-color:{color}40;border-left-color:{color};
             background:{color}06">
          <div>
            <div style="font-weight:700;color:{color};font-size:.88rem;margin-bottom:.25rem">{title}</div>
            <div style="color:#94a3b8;font-size:.8rem">{msg}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    if st.button("Dismiss All"):
        st.session_state.alerts = []; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 13 — BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
elif "Benchmark" in nav:
    st.markdown('<div class="page-title">Model <span>Benchmark</span></div>', unsafe_allow_html=True)
    rouge = {"ROUGE-1":0.3358,"ROUGE-2":0.1065,"ROUGE-L":0.2570,"ROUGE-Lsum":0.2572}
    intent_m = {
        "action item":   {"precision":0.38,"recall":0.80,"f1":0.52},
        "decision":      {"precision":1.00,"recall":0.50,"f1":0.67},
        "informational": {"precision":0.40,"recall":0.40,"f1":0.40},
        "question":      {"precision":0.58,"recall":0.70,"f1":0.64},
        "risk or issue": {"precision":0.83,"recall":0.50,"f1":0.62},
        "suggestion":    {"precision":1.00,"recall":0.60,"f1":0.75},
    }
    st.markdown('<div class="section-label">ROUGE Scores — 400 SAMSum test samples (BART-large-cnn)</div>', unsafe_allow_html=True)
    rc = st.columns(4)
    for col, (k, v) in zip(rc, rouge.items()):
        with col:
            st.markdown(f"""<div class="metric">
              <div class="val" style="font-size:1.6rem;color:#38bdf8">{v:.4f}</div>
              <div class="lbl">{k}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Intent Classification Metrics — DeBERTa v3 zero-shot</div>', unsafe_allow_html=True)
    mdf = pd.DataFrame(intent_m).T.reset_index(); mdf.columns = ["Intent","Precision","Recall","F1"]
    fig = go.Figure()
    for m, c in [("Precision","#38bdf8"),("Recall","#34d399"),("F1","#c084fc")]:
        fig.add_trace(go.Bar(name=m, x=mdf["Intent"], y=mdf[m], marker_color=c))
    fig.update_layout(**plotly_dark_layout(), barmode="group", xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)
    st.progress(0.58)
    st.caption("Macro Accuracy: **58%** · Macro F1: **0.60** · Weighted F1: **0.60**")

    st.markdown('<div class="section-label">Model Architecture</div>', unsafe_allow_html=True)
    arch_data = [
        {"Module":"Intent Classifier","Model":"DeBERTa v3 small (zero-shot)","Task":"6-class NLI","Params":"~184M"},
        {"Module":"NER","Model":"BERT-base-NER (dslim)","Task":"Token Classification","Params":"~110M"},
        {"Module":"Summarizer","Model":"BART-large-cnn","Task":"Seq2Seq Abstractive","Params":"~406M"},
        {"Module":"STT","Model":"Whisper large-v3","Task":"ASR (multilingual)","Params":"~1.5B"},
        {"Module":"Embeddings","Model":"all-MiniLM-L6-v2","Task":"Semantic Similarity","Params":"~22M"},
    ]
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 14 — SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif "Session History" in nav:
    st.markdown('<div class="page-title">Session <span>History</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No sessions yet."); st.stop()
    if st.button("Clear All History"): st.session_state.history = []; st.rerun()
    for i, record in enumerate(st.session_state.history):
        counts = Counter(s["intent"] for s in record["sentences"])
        dom = counts.most_common(1)[0][0] if counts else "—"
        with st.expander(f"Session {len(st.session_state.history)-i}  ·  {record['ts']}  ·  {record['word_count']} words"):
            st.markdown(f"**Summary:** {record['summary']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Dominant:** {badge_html(dom)}", unsafe_allow_html=True)
            c2.markdown(f"**Sentiment:** <span style='color:{record['sent_color']};font-weight:700'>{record['sentiment']}</span>", unsafe_allow_html=True)
            c3.markdown(f"**Topics:** {', '.join(record['topics'][:3])}")
            for s in record["sentences"][:8]:
                st.markdown(f"""
                <div class="tl-item">
                  <div class="tl-dot" style="background:{COLORS.get(s['intent'],'#64748b')}"></div>
                  <div>{badge_html(s['intent'])} <span style="font-size:.82rem;color:#cbd5e1;margin-left:.4rem">{s['text']}</span></div>
                </div>""", unsafe_allow_html=True)
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("Bookmark", key=f"bk_{i}"):
                    st.session_state.bookmarks.append(record); st.success("Bookmarked")
            with bc2:
                if st.button("Set for Compare", key=f"cmp_{i}"):
                    if st.session_state.comparison_a is None: st.session_state.comparison_a = record
                    else: st.session_state.comparison_b = record
                    st.success("Set — go to Compare Sessions")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 15 — SPEAKER PROFILES  (Feature 31 — new)
# ─────────────────────────────────────────────────────────────────────────────
elif "Speaker Profiles" in nav:
    st.markdown('<div class="page-title">Speaker <span>Profiles</span></div>', unsafe_allow_html=True)
    profiles = st.session_state.speaker_profiles
    if not profiles:
        st.info("No speaker data yet. Run a session with speaker-labeled transcripts."); st.stop()

    for spk, data in profiles.items():
        color = SPEAKER_COLORS[hash(spk) % len(SPEAKER_COLORS)]
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {color}">
              <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:.3rem">{spk}</div>
              <div style="font-size:.75rem;color:#64748b">
                {data['sessions']} session(s) &nbsp;·&nbsp; {data['total_words']} words total
              </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.metric("Sessions", data["sessions"])
        with c3:
            avg_wps = data["total_words"] // max(data["sessions"], 1)
            st.metric("Avg Words/Session", avg_wps)

    # Word distribution chart
    if len(profiles) > 1:
        spk_df = pd.DataFrame([
            {"Speaker": k, "Total Words": v["total_words"], "Sessions": v["sessions"]}
            for k, v in profiles.items()
        ])
        fig = px.bar(spk_df, x="Speaker", y="Total Words", color="Speaker",
                     color_discrete_sequence=SPEAKER_COLORS, text="Sessions")
        fig.update_layout(**plotly_dark_layout(), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 16 — TRANSCRIPT DIFF  (Feature 32 — new)
# ─────────────────────────────────────────────────────────────────────────────
elif "Transcript Diff" in nav:
    st.markdown('<div class="page-title">Transcript <span>Diff Viewer</span></div>', unsafe_allow_html=True)
    history = st.session_state.transcript_history
    if len(history) < 2:
        st.info("Need at least 2 transcribed sessions to compare."); st.stop()

    options = [f"Session {i+1} — {h['ts']} ({h['word_count']} words)" for i, h in enumerate(history)]
    col_a, col_b = st.columns(2)
    with col_a: idx_a = st.selectbox("Transcript A", range(len(options)), format_func=lambda x: options[x])
    with col_b: idx_b = st.selectbox("Transcript B", range(len(options)), format_func=lambda x: options[x], index=min(1, len(options)-1))

    ta = history[idx_a]["text"].split()
    tb = history[idx_b]["text"].split()

    # Simple word-level diff highlight
    set_a, set_b = set(ta), set(tb)
    added   = set_b - set_a
    removed = set_a - set_b

    def highlight_words(words, added, removed):
        out = []
        for w in words:
            clean = re.sub(r"[^a-zA-Z0-9]","",w).lower()
            if clean in {r.lower() for r in removed}:
                out.append(f'<span style="background:#f8717130;color:#f87171;border-radius:3px;padding:.05rem .25rem">{w}</span>')
            elif clean in {a.lower() for a in added}:
                out.append(f'<span style="background:#34d39930;color:#34d399;border-radius:3px;padding:.05rem .25rem">{w}</span>')
            else:
                out.append(f'<span style="color:#cbd5e1">{w}</span>')
        return " ".join(out)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="section-label">Transcript A (removed words)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="font-size:.8rem;line-height:1.8">{highlight_words(ta, set(), removed)}</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="section-label">Transcript B (added words)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="font-size:.8rem;line-height:1.8">{highlight_words(tb, added, set())}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-sm" style="margin-top:.5rem">
      <span style="color:#34d399;font-size:.78rem;font-weight:600">+ {len(added)} new words</span>
      &nbsp;&nbsp;
      <span style="color:#f87171;font-size:.78rem;font-weight:600">- {len(removed)} removed words</span>
      &nbsp;&nbsp;
      <span style="color:#64748b;font-size:.78rem">{abs(len(tb)-len(ta))} word count delta</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 17 — KEYWORD SEARCH  (Feature 33 — new)
# ─────────────────────────────────────────────────────────────────────────────
elif "Keyword Search" in nav:
    st.markdown('<div class="page-title">Keyword <span>Search</span></div>', unsafe_allow_html=True)
    if not st.session_state.history: st.info("No sessions yet."); st.stop()

    query = st.text_input("Search across all sessions", placeholder="e.g. budget, deadline, security…")

    if query.strip():
        q_words = [w.lower() for w in query.split() if len(w) > 1]
        results = []
        for h_idx, h in enumerate(st.session_state.history):
            for s in h["sentences"]:
                matched = [w for w in q_words if w in s["text"].lower()]
                if matched:
                    results.append({
                        "session": f"S{h_idx+1}",
                        "ts": h["ts"],
                        "intent": s["intent"],
                        "confidence": s["confidence"],
                        "sentiment": s["sentiment"],
                        "text": s["text"],
                        "matches": matched,
                    })

        st.markdown(f'<div class="section-label">{len(results)} result(s) across {len(st.session_state.history)} session(s)</div>', unsafe_allow_html=True)

        for r in results:
            c = COLORS.get(r["intent"], "#64748b")
            # Highlight matched words
            text = r["text"]
            for mw in r["matches"]:
                text = re.sub(
                    f"({re.escape(mw)})",
                    r'<mark style="background:#fbbf2430;color:#fbbf24;border-radius:3px;padding:.05rem .2rem">\1</mark>',
                    text, flags=re.IGNORECASE
                )
            st.markdown(f"""
            <div class="sent-card" style="border-left-color:{c}">
              <div style="display:flex;gap:.5rem;align-items:center;margin-bottom:.35rem;flex-wrap:wrap">
                <span style="font-size:.65rem;color:#475569;font-family:'DM Mono',monospace;background:var(--surface3);
                      padding:.15rem .5rem;border-radius:5px">{r['session']} · {r['ts']}</span>
                {badge_html(r['intent'])}
                <span style="font-size:.65rem;color:{r['sent_color']};font-weight:600">{r['sentiment']}</span>
              </div>
              <div style="font-size:.86rem;line-height:1.7">{text}</div>
            </div>""", unsafe_allow_html=True)
    else:
        # Show recent sentences
        st.markdown('<div class="section-label">Recent Sentences (enter a keyword to filter)</div>', unsafe_allow_html=True)
        recent = [s for h in st.session_state.history[:3] for s in h["sentences"]][:20]
        for s in recent:
            st.markdown(f"""
            <div class="tl-item">
              <div class="tl-dot" style="background:{COLORS.get(s['intent'],'#64748b')}"></div>
              <div>{badge_html(s['intent'])} <span style="font-size:.82rem;color:#94a3b8;margin-left:.4rem">{s['text'][:100]}</span></div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 18 — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
elif "Settings" in nav:
    st.markdown('<div class="page-title">App <span>Settings</span></div>', unsafe_allow_html=True)

    with st.expander("Appearance", expanded=True):
        st.session_state.theme = st.radio("Theme", ["Dark","Light"], horizontal=True)
        st.session_state.lang  = st.selectbox("Interface Language", ["English","Arabic","French","Spanish"])

    with st.expander("NLP Pipeline"):
        st.session_state.auto_summarize     = st.toggle("Auto-Summarization", st.session_state.auto_summarize)
        st.session_state.highlight_entities = st.toggle("Highlight Entities", st.session_state.highlight_entities)
        st.session_state.min_conf           = st.slider("Min Confidence Threshold", 0.0, 1.0, st.session_state.min_conf, .05)

    with st.expander("Whisper STT — Model Selection"):
        current = st.session_state.whisper_size
        st.markdown(f'Current model: <span class="model-pill">{current}</span>', unsafe_allow_html=True)
        model_choice = st.selectbox("Select Whisper Model", list(WHISPER_MODELS.keys()))
        new_size = WHISPER_MODELS[model_choice]
        if new_size != current:
            st.session_state.whisper_size = new_size
            st.info(f"Model changed to `{new_size}`. Will load on next transcription.")
        if st.button("Pre-load Whisper Now"):
            with st.spinner(f"Loading Whisper {new_size}…"):
                m = load_whisper(new_size)
            st.success(f"Whisper {new_size} loaded") if m else st.error("Whisper unavailable.")

    with st.expander("Session Notes"):
        st.session_state.session_notes = st.text_area("Notes", st.session_state.session_notes, height=140)
        if st.button("Save Notes"): st.success("Saved")

    with st.expander("Bookmarks"):
        if not st.session_state.bookmarks: st.info("No bookmarks.")
        else:
            for i, bk in enumerate(st.session_state.bookmarks):
                st.markdown(f"**{i+1}.** {bk['ts']} — {bk['summary'][:80]}…")
            if st.button("Clear Bookmarks"): st.session_state.bookmarks = []; st.rerun()

    with st.expander("Custom Vocabulary — Feature 34"):
        st.caption("Domain-specific words to boost recognition accuracy.")
        vocab_input = st.text_area("Custom terms (one per line)", value="\n".join(st.session_state.custom_vocab), height=100)
        if st.button("Save Vocabulary"):
            st.session_state.custom_vocab = [w.strip() for w in vocab_input.split("\n") if w.strip()]
            st.success(f"Saved {len(st.session_state.custom_vocab)} terms")

    with st.expander("Danger Zone"):
        if st.button("Reset ALL Data", type="secondary"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 19 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif "About" in nav:
    st.markdown('<div class="page-title">About <span>VoiceIQ</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="border-left:3px solid #38bdf8">
      <div style="font-size:1.2rem;font-weight:800;margin-bottom:.5rem;letter-spacing:-.02em">
        NLU Meeting Voice-to-Action Assistant
        <span style="font-size:.75rem;color:#38bdf8;font-family:'DM Mono',monospace;
              background:rgba(56,189,248,.1);padding:.2rem .6rem;border-radius:6px;margin-left:.6rem">v3.0</span>
      </div>
      <p style="color:#64748b;font-size:.87rem;line-height:1.8;margin:0">
        End-to-end meeting intelligence pipeline: upload or record audio in any format →
        Whisper large-v3 transcription → transformer intent classification →
        named entity recognition → abstractive summarization → rich analytics & export.<br><br>
        Built on HuggingFace Transformers · Sentence-BERT · Plotly · Streamlit · OpenAI Whisper.
      </p>
    </div>""", unsafe_allow_html=True)

    features = [
        ("Live Microphone Recording", "Record directly in-browser via st-audiorec"),
        ("Universal Audio Upload", "WAV · MP3 · M4A · OGG · FLAC · WEBM · AAC · OPUS"),
        ("Whisper large-v3 STT", "State-of-the-art multilingual speech recognition (~1.5B params)"),
        ("Selectable Whisper Model", "Choose from tiny to large-v3 based on speed/accuracy tradeoff"),
        ("Intent Classification", "6 intents + confidence — DeBERTa v3 zero-shot"),
        ("Abstractive Summarization", "BART-large-cnn with extractive fallback"),
        ("Sentence Sentiment", "Positive / Neutral / Negative per sentence and session"),
        ("Named Entity Extraction", "PERSON · DATE · TECH · ORG with inline highlighting"),
        ("Speaker Diarization", "Auto-detect speakers from Name: text format"),
        ("Analytics Dashboard", "Pie · bar · histogram · treemap across sessions"),
        ("Intent Explorer", "Multi-filter search with confidence and sentiment sliders"),
        ("Action Item Tracker", "Auto-extract + manual add + owner + due date + progress bar"),
        ("Smart Alerts", "Auto-flag high risk / many actions / negative meeting tone"),
        ("Meeting Q&A", "Retrieval-based keyword chatbot for meetings"),
        ("Session Compare", "Side-by-side intent delta and summary diff"),
        ("Topic Analysis", "Word cloud + treemap of key discussion topics"),
        ("Entity Map", "Frequency charts for all extracted named entities"),
        ("Benchmark Report", "ROUGE + classification metrics from evaluation notebooks"),
        ("Export JSON", "Full structured export with all metadata"),
        ("Export CSV", "Tabular sentence export for spreadsheet analysis"),
        ("Export PDF", "Professional formatted report via FPDF"),
        ("Export Markdown", "Clean Markdown report for documentation"),
        ("Bookmarks", "Save and revisit important sessions"),
        ("Session Notes", "Personal freeform notes per session"),
        ("Confidence Filter", "Global min-confidence threshold slider"),
        ("Priority Sort", "Sort sentences by urgency score"),
        ("Timeline View", "Sequential intent and temporal reference timeline"),
        ("Session History", "Full history with bookmarking and comparison"),
        ("Lazy Model Loading", "HuggingFace models loaded on demand"),
        ("Meeting Tags", "Tag meetings by type, team, or sprint — NEW"),
        ("Speaker Profiles", "Cumulative word counts and session stats per speaker — NEW"),
        ("Transcript Diff Viewer", "Word-level diff between two session transcripts — NEW"),
        ("Keyword Search", "Full-text search across all sessions with match highlighting — NEW"),
        ("Custom Vocabulary", "Domain-specific terms for improved recognition — NEW"),
        ("Action Item Owners & Due Dates", "Assign owners and deadlines to each action — NEW"),
    ]
    c1, c2 = st.columns(2)
    for i, (title, desc) in enumerate(features):
        is_new = "NEW" in desc
        with (c1 if i % 2 == 0 else c2):
            border = "rgba(56,189,248,.3)" if is_new else "var(--border2)"
            st.markdown(f"""
            <div class="card-sm" style="border-color:{border}">
              <div style="font-weight:700;font-size:.84rem;display:flex;align-items:center;gap:.5rem">
                {title}
                {('<span style="font-size:.6rem;color:#38bdf8;font-family:DM Mono,monospace;background:rgba(56,189,248,.1);padding:.1rem .4rem;border-radius:4px;border:1px solid rgba(56,189,248,.2)">NEW</span>') if is_new else ''}
              </div>
              <div style="color:#475569;font-size:.74rem;margin-top:.2rem">{desc.replace(" — NEW","")}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption("VoiceIQ v3.0 · Streamlit · Plotly · HuggingFace · Whisper large-v3 · WordCloud · FPDF · Sentence-BERT")
