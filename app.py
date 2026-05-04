"""
NLU Meeting Voice-to-Action Assistant  v2.0
30 Features | Professional Dark Glassmorphism UI
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
    page_title="NLU Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
INTENTS = ["action item", "decision", "suggestion", "question", "informational", "risk or issue"]
COLORS  = {
    "action item":   "#3d7fff",
    "decision":      "#06d6a0",
    "suggestion":    "#f72585",
    "question":      "#f59e0b",
    "informational": "#5d6b8a",
    "risk or issue": "#ef4444",
}
ICONS = {
    "action item":   "✅",
    "decision":      "🟢",
    "suggestion":    "💡",
    "question":      "❓",
    "informational": "ℹ️",
    "risk or issue": "⚠️",
}
SPEAKER_COLORS = ["#3d7fff","#06d6a0","#f72585","#f59e0b","#a78bfa","#34d399","#fb923c"]

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS (glassmorphism dark theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0d1117;
    --surface: #161b22;
    --surface2:#21262d;
    --border:  rgba(255,255,255,0.08);
    --accent:  #3d7fff;
    --muted:   #5d6b8a;
    --text:    #e8eaf6;
}

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.main .block-container { padding: 1.5rem 2rem 4rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Cards */
.nlu-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(8px);
    transition: border-color .2s;
}
.nlu-card:hover { border-color: rgba(61,127,255,0.3); }

/* Metric boxes */
.metric-box {
    background: var(--surface2);
    border-radius: 12px;
    padding: .8rem;
    text-align: center;
    border: 1px solid var(--border);
}
.metric-box .val { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-box .lbl { font-size: .68rem; color: var(--muted); margin-top: .3rem; text-transform: uppercase; letter-spacing: .08em; }

/* Badge */
.badge {
    display: inline-flex; align-items: center; gap: .25rem;
    padding: .2rem .55rem; border-radius: 20px; font-size: .7rem;
    font-weight: 600; border: 1px solid; letter-spacing: .04em;
}

/* Voice zone */
.voice-zone {
    background: linear-gradient(135deg,rgba(61,127,255,.08),rgba(247,37,133,.05));
    border: 1px dashed rgba(61,127,255,.3);
    border-radius: 16px; padding: 1.6rem; text-align: center; margin-bottom: 1rem;
}
.rec-pulse {
    font-size: 2.5rem; animation: pulse 1.5s infinite;
    display: inline-block; margin-bottom: .5rem;
}
@keyframes pulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.15);opacity:.8} }

/* Timeline */
.timeline-item {
    display: flex; align-items: flex-start; gap: .7rem;
    padding: .45rem 0; border-bottom: 1px solid var(--border);
}
.timeline-dot {
    width: 10px; height: 10px; border-radius: 50%;
    margin-top: .35rem; flex-shrink: 0;
}

/* Speaker tag */
.speaker-tag {
    font-size: .7rem; font-weight: 700; padding: .15rem .5rem;
    border-radius: 20px; border: 1px solid; letter-spacing: .05em;
}

/* Action card */
.action-card {
    display: flex; align-items: center; gap: .7rem;
    background: var(--surface2); border-radius: 10px;
    padding: .6rem .9rem; margin-bottom: .3rem;
}
.action-check { font-size: 1.1rem; flex-shrink: 0; }

/* Scrollable table */
.df-scroll { max-height: 420px; overflow-y: auto; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#3d7fff,#0051e6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# NLP HELPERS  (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_nlp_models():
    """Load NLP pipeline models once."""
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


@st.cache_resource(show_spinner="Loading Whisper STT…")
def load_whisper():
    try:
        import whisper
        return whisper.load_model("base")
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Intent keyword rules (fast fallback if model unavailable)
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
    if re.search(_SENTIMENT_RULES["Negative"], t): return "Negative","#ef4444"
    if re.search(_SENTIMENT_RULES["Positive"], t): return "Positive","#06d6a0"
    return "Neutral","#f59e0b"

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
    return [w for w in Counter(w for w in words if w not in stop).most_common(8) if w[1]>0] and [w for w,_ in Counter(w for w in words if w not in stop).most_common(8)]

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

def analyze_transcript(transcript: str, selected_intents=None, models=None) -> dict:
    if selected_intents is None: selected_intents = INTENTS
    lines = [l.strip() for l in transcript.strip().split("\n") if l.strip()]
    
    # Parse speaker format "Name: text"
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
    
    # Summary
    summary = extractive_summary(full_text)
    if models and models.get("summarizer") and len(full_text.split()) > 50:
        try:
            out = models["summarizer"](full_text[:1024], max_length=120, min_length=30, do_sample=False)
            summary = out[0]["summary_text"]
        except Exception:
            pass
    
    # Temporal
    temporal = temporal_normalize(full_text)
    
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
        "alerts": [],
    }

def check_alerts(record: dict) -> list:
    alerts = []
    risk_count = sum(1 for s in record["sentences"] if s["intent"] == "risk or issue")
    action_count = sum(1 for s in record["sentences"] if s["intent"] == "action item")
    if risk_count >= 3:
        alerts.append(("⚠️","High Risk Detected",f"{risk_count} risk/issue sentences found.","#ef4444"))
    if action_count >= 5:
        alerts.append(("📋","Many Action Items",f"{action_count} action items detected — assign owners.","#f59e0b"))
    if record["sentiment"] == "Negative":
        alerts.append(("😟","Negative Tone",f"Overall meeting tone is negative.","#f472b6"))
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
        f"# NLU Meeting Report",
        f"*Generated: {record['ts']}*",
        f"*Words: {record['word_count']} | Sentiment: {record['sentiment']}*",
        "",
        "## Summary",
        record["summary"],
        "",
        "## Topics",
        " ".join(f"#{t}" for t in record["topics"]),
        "",
        "## Classified Sentences",
    ]
    for s in record["sentences"]:
        lines.append(f"- **[{s['intent']}]** `{s['confidence']:.2f}` {s['text']}")
    return "\n".join(lines)

def export_pdf(record: dict) -> bytes:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "NLU Meeting Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Generated: {record['ts']} | Words: {record['word_count']}", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, record["summary"])
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Classified Sentences", ln=True)
        pdf.set_font("Helvetica", "", 8)
        for s in record["sentences"][:40]:
            line = f"[{s['intent'].upper()}] {s['text'][:100]}"
            pdf.multi_cell(0, 4.5, line)
        return pdf.output(dest="S").encode("latin-1", errors="replace")
    except Exception as e:
        return f"PDF export error: {e}".encode()

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def badge_html(intent: str) -> str:
    c = COLORS.get(intent, "#5d6b8a")
    icon = ICONS.get(intent, "")
    return f'<span class="badge" style="color:{c};border-color:{c}40;background:{c}15">{icon} {intent}</span>'

def make_wordcloud(text: str):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(
            width=900, height=350, background_color="black",
            colormap="Blues", max_words=80,
            collocations=False,
        ).generate(text or "meeting")
        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="black")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = BytesIO()
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", dpi=120, facecolor="black")
        buf.seek(0)
        plt.close()
        return buf
    except Exception:
        return None

def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    model = load_whisper()
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

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.8rem;background:linear-gradient(135deg,rgba(61,127,255,.15),rgba(247,37,133,.08));
         border-radius:14px;margin-bottom:1.2rem;border:1px solid rgba(61,127,255,.2)">
      <div style="font-size:1.3rem;font-weight:800;letter-spacing:-.02em">🎙️ NLU Meeting</div>
      <div style="font-size:.72rem;color:#5d6b8a;margin-top:.15rem">Voice-to-Action Assistant v2</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["🎙️ Voice & Analyze", "📊 Dashboard", "🏷️ Intent Explorer",
         "👤 Speaker Diarization", "🗂️ Entity Map", "#️⃣ Topics",
         "✅ Action Tracker", "😊 Sentiment", "🔄 Compare Sessions",
         "🤖 Q&A Chatbot", "⏰ Timeline", "⚠️ Smart Alerts",
         "📈 Benchmark", "🕒 History", "⚙️ Settings", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"""
    <div style="font-size:.72rem;color:#5d6b8a">
      Sessions: <b style="color:#e8eaf6">{len(st.session_state.history)}</b> &nbsp;|&nbsp;
      Actions: <b style="color:#e8eaf6">{len(st.session_state.action_items)}</b> &nbsp;|&nbsp;
      Alerts: <b style="color:#ef4444">{len(st.session_state.alerts)}</b>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.caption(f"Last: {st.session_state.history[0]['ts']}")

    # Feature 30: Model load toggle
    st.divider()
    if st.button("⚡ Load NLP Models", use_container_width=True):
        with st.spinner("Loading models (first time may take a minute)…"):
            try:
                models = load_nlp_models()
                st.session_state.model_loaded = True
                st.success("✅ Models loaded!")
            except Exception as e:
                st.warning(f"Partial load: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — VOICE & ANALYZE  (Features 1-10)
# ─────────────────────────────────────────────────────────────────────────────
if "Voice" in nav:
    st.markdown("## 🎙️ Voice & Transcript Analyzer")

    mode_tab, upload_tab, text_tab = st.tabs(
        ["🎤 Record Voice", "📁 Upload Audio File", "✍️ Paste Text"]
    )
    transcript_to_analyze = ""

    # Feature 1: Live mic recording
    with mode_tab:
        st.markdown("""
        <div class="voice-zone">
          <div class="rec-pulse">🎤</div>
          <div style="font-size:1rem;font-weight:700;margin-bottom:.3rem">Live Microphone Recording</div>
          <div style="color:#5d6b8a;font-size:.82rem">Click the recorder → speak → stop → transcribe</div>
        </div>
        """, unsafe_allow_html=True)
        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                st.audio(wav_audio_data, format="audio/wav")
                if st.button("🧠 Transcribe Recording", key="transcribe_mic"):
                    with st.spinner("Whisper is transcribing…"):
                        transcript_to_analyze = transcribe_audio(wav_audio_data, ".wav")
                    st.success("✅ Done!")
                    st.session_state["pending_transcript"] = transcript_to_analyze
        except ImportError:
            st.info("Install `st-audiorec` for live recording. Use Upload or Paste text instead.")

    # Feature 2: Audio file upload
    with upload_tab:
        st.markdown("""
        <div class="voice-zone">
          <div style="font-size:2rem;margin-bottom:.5rem">📁</div>
          <div style="font-weight:700;margin-bottom:.3rem">Upload Audio File</div>
          <div style="color:#5d6b8a;font-size:.82rem">WAV · MP3 · M4A · OGG · FLAC</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_audio = st.file_uploader(
            "Choose audio file",
            type=["wav","mp3","m4a","ogg","flac","webm"],
            label_visibility="collapsed",
        )
        if uploaded_audio:
            st.audio(uploaded_audio)
            c1, c2 = st.columns(2)
            c1.metric("File", uploaded_audio.name)
            c2.metric("Size", f"{uploaded_audio.size/1024:.1f} KB")
            if st.button("🧠 Transcribe Audio", key="transcribe_file"):
                suffix = "." + uploaded_audio.name.split(".")[-1]
                with st.spinner(f"Whisper processing '{uploaded_audio.name}'…"):
                    transcript_to_analyze = transcribe_audio(uploaded_audio.read(), suffix)
                st.success("✅ Done!")
                st.session_state["pending_transcript"] = transcript_to_analyze
                st.markdown(f"""
                <div class="nlu-card">
                  <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;margin-bottom:.5rem">📝 Transcript</div>
                  <div style="font-size:.9rem;line-height:1.7">{transcript_to_analyze}</div>
                </div>""", unsafe_allow_html=True)

    # Feature 3: Paste text
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
            height=220,
            value=st.session_state.get("pending_transcript",""),
            placeholder=sample,
        )
        if not manual_text.strip():
            if st.button("📋 Load Demo Transcript"):
                st.session_state["pending_transcript"] = sample
                st.rerun()
        if manual_text.strip():
            transcript_to_analyze = manual_text

    if st.session_state.get("pending_transcript") and not transcript_to_analyze:
        transcript_to_analyze = st.session_state["pending_transcript"]

    # Feature 4: Analysis options
    with st.expander("🔧 Analysis Options"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("**Filter Intents**")
            selected_intents = [i for i in INTENTS if st.checkbox(i.capitalize(), True, key=f"fi_{i}")]
        with col_f2:
            st.markdown("**Speaker Detection**")
            enable_diarization = st.toggle("Auto-detect speakers", True)
        with col_f3:
            st.markdown("**Sort Options**")
            sort_by_priority = st.toggle("Sort by priority", False)
            st.session_state.highlight_entities = st.toggle("Highlight entities", st.session_state.highlight_entities)

    run_col, clear_col = st.columns([3,1])
    with run_col:
        run_btn = st.button("🚀 Run Full NLU Pipeline", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["pending_transcript"] = ""
            st.rerun()

    # Feature 5: NLU pipeline execution
    if run_btn and transcript_to_analyze.strip():
        models = None
        if st.session_state.model_loaded:
            try: models = load_nlp_models()
            except: pass

        with st.spinner("Running NLU pipeline…"):
            prog = st.progress(0)
            stages = ["Tokenizing…","Classifying intents…","Extracting entities…","Summarizing…","Building graph…"]
            for i, stage in enumerate(stages):
                time.sleep(0.12); prog.progress((i+1)*20)
            record = analyze_transcript(transcript_to_analyze, selected_intents, models)
            if sort_by_priority:
                record["sentences"].sort(key=lambda x: x["priority"], reverse=True)
            alerts = check_alerts(record)
            record["alerts"] = alerts
            st.session_state.history.insert(0, record)
            for s in record["sentences"]:
                if s["intent"] == "action item":
                    if s["text"] not in [a["text"] for a in st.session_state.action_items]:
                        st.session_state.action_items.append({"text":s["text"],"done":False,"ts":record["ts"]})
            if alerts:
                st.session_state.alerts = alerts + st.session_state.alerts

        st.success(f"✅ Processed {record['word_count']} words → {len(record['sentences'])} classified sentences")

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.history:
        record = st.session_state.history[0]

        # Feature 6: Smart alerts banner
        if record.get("alerts"):
            for icon, title, msg, color in record["alerts"]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);border:1px solid {color}40;border-left:4px solid {color};
                     border-radius:10px;padding:.7rem 1rem;margin-bottom:.5rem;display:flex;gap:.7rem;align-items:center">
                  <span style="font-size:1.2rem">{icon}</span>
                  <div><div style="font-weight:700;color:{color};font-size:.85rem">{title}</div>
                  <div style="color:#9ca3af;font-size:.78rem">{msg}</div></div>
                </div>""", unsafe_allow_html=True)

        # Feature 7: Summary card with sentiment & topics
        s_icon = "🟢" if record["sentiment"]=="Positive" else "🔴" if record["sentiment"]=="Negative" else "🟡"
        st.markdown(f"""
        <div class="nlu-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem">
            <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em">
              📝 AI Summary — {record['ts']}
            </div>
            <div style="font-size:.78rem;color:{record['sent_color']}">{s_icon} {record['sentiment']}</div>
          </div>
          <div style="font-size:.92rem;line-height:1.75;color:#d1d5db">{record['summary']}</div>
          <div style="margin-top:.8rem;display:flex;gap:.4rem;flex-wrap:wrap">
            {"".join(f'<span style="background:rgba(255,255,255,.06);border-radius:6px;padding:.18rem .55rem;font-size:.68rem;color:#5d6b8a">#{t}</span>' for t in record["topics"])}
          </div>
        </div>""", unsafe_allow_html=True)

        # Feature 8: Intent metric boxes
        counts = Counter(s["intent"] for s in record["sentences"])
        cols = st.columns(6)
        for col, intent in zip(cols, INTENTS):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="val" style="color:{COLORS[intent]}">{counts.get(intent,0)}</div>
                  <div class="lbl">{ICONS[intent]}<br>{intent.split()[0]}</div>
                </div>""", unsafe_allow_html=True)

        # Feature 8b: Temporal references
        if record.get("temporal"):
            st.markdown("#### ⏰ Temporal References")
            tcols = st.columns(len(record["temporal"][:4]))
            for col, t in zip(tcols, record["temporal"][:4]):
                with col:
                    st.markdown(f"""<div class="metric-box">
                      <div style="font-size:.8rem;font-weight:700;color:#f59e0b">{t['date']}</div>
                      <div class="lbl">{t['expression']}</div>
                    </div>""", unsafe_allow_html=True)

        # Feature 9: Classified sentences
        st.markdown("### 🔍 Classified Sentences")
        for s in record["sentences"]:
            conf_color = "#06d6a0" if s["confidence"]>.85 else "#f59e0b" if s["confidence"]>.7 else "#ef4444"
            ent_html = ""
            if st.session_state.highlight_entities and s["entities"]:
                ecols = {"PERSON":"#60a5fa","DATE":"#34d399","TECH":"#fbbf24","ORG":"#a78bfa"}
                ent_html = " ".join(
                    f'<span style="background:rgba(255,255,255,.06);border-radius:4px;padding:.1rem .4rem;'
                    f'font-size:.68rem;color:{ecols.get(e["type"],"#9ca3af")};margin-right:.2rem">'
                    f'{e["text"]}</span>'
                    for e in s["entities"][:4]
                )
            st.markdown(f"""
            <div class="nlu-card" style="padding:.9rem 1.1rem">
              <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.35rem">
                {badge_html(s['intent'])}
                <span style="font-size:.68rem;color:{conf_color};font-family:'JetBrains Mono',monospace">▲{s['confidence']:.2f}</span>
                <span style="font-size:.68rem;color:{s['sent_color']};margin-left:.2rem">{s['sentiment']}</span>
                {ent_html}
              </div>
              <div style="font-size:.87rem;color:#c8cfe8;line-height:1.65">{s['text']}</div>
            </div>""", unsafe_allow_html=True)

        # Feature 10: Export (4 formats)
        st.markdown("### 📤 Export Results")
        e1, e2, e3, e4 = st.columns(4)
        with e1: st.download_button("📄 JSON",  export_json(record),     "nlu.json","application/json")
        with e2: st.download_button("📊 CSV",   export_csv(record),      "nlu.csv", "text/csv")
        with e3: st.download_button("📑 PDF",   export_pdf(record),      "nlu.pdf", "application/pdf")
        with e4: st.download_button("📝 Markdown", export_markdown(record),"nlu.md","text/markdown")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DASHBOARD  (Features 11-15)
# ─────────────────────────────────────────────────────────────────────────────
elif "Dashboard" in nav:
    st.markdown("## 📊 Analytics Dashboard")
    if not st.session_state.history: st.info("Run a session first."); st.stop()

    df = pd.DataFrame([{
        "intent":s["intent"],"confidence":s["confidence"],
        "sentiment":s["sentiment"],"priority":s["priority"],
        "session":f"S{i+1}"
    } for i,h in enumerate(reversed(st.session_state.history)) for s in h["sentences"]])

    # Feature 11: Intent distribution pie
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("#### Intent Distribution")
        cnt = df["intent"].value_counts().reset_index(); cnt.columns=["intent","count"]
        fig = px.pie(cnt,names="intent",values="count",hole=.45,color="intent",color_discrete_map=COLORS)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)

    # Feature 12: Sentiment distribution
    with r1c2:
        st.markdown("#### Sentiment Distribution")
        s_cnt = df["sentiment"].value_counts().reset_index(); s_cnt.columns=["sentiment","count"]
        scols = {"Positive":"#06d6a0","Neutral":"#f59e0b","Negative":"#ef4444"}
        fig2 = px.pie(s_cnt,names="sentiment",values="count",hole=.45,color="sentiment",color_discrete_map=scols)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2,use_container_width=True)

    # Feature 13: Intent trend across sessions
    st.markdown("#### Intent Trend Across Sessions")
    pivot = df.groupby(["session","intent"]).size().reset_index(name="count")
    fig3 = px.bar(pivot,x="session",y="count",color="intent",barmode="stack",color_discrete_map=COLORS)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig3,use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    # Feature 14: Avg confidence per intent
    with r2c1:
        st.markdown("#### Avg Confidence per Intent")
        avg = df.groupby("intent")["confidence"].mean().reset_index()
        fig4 = px.bar(avg,x="intent",y="confidence",color="intent",color_discrete_map=COLORS,range_y=[0,1])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",showlegend=False,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig4,use_container_width=True)

    # Feature 15: Priority distribution
    with r2c2:
        st.markdown("#### Priority Score Distribution")
        fig5 = px.histogram(df,x="priority",nbins=4,color="intent",color_discrete_map=COLORS,barmode="overlay",opacity=.75)
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig5,use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — INTENT EXPLORER  (Feature 16)
# ─────────────────────────────────────────────────────────────────────────────
elif "Intent Explorer" in nav:
    st.markdown("## 🏷️ Intent Explorer")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_s = [s for h in st.session_state.history for s in h["sentences"]]
    c1, c2, c3 = st.columns(3)
    with c1: sel = st.multiselect("Intents",INTENTS,INTENTS)
    with c2: conf_r = st.slider("Confidence",0.0,1.0,(0.0,1.0),.01)
    with c3: sent_f = st.multiselect("Sentiment",["Positive","Neutral","Negative"],["Positive","Neutral","Negative"])
    filtered = [s for s in all_s if s["intent"] in sel
                and conf_r[0]<=s["confidence"]<=conf_r[1]
                and s["sentiment"] in sent_f]
    search = st.text_input("🔎 Search sentences","")
    if search: filtered=[s for s in filtered if search.lower() in s["text"].lower()]
    st.caption(f"Showing **{len(filtered)}** sentences")
    df_v = pd.DataFrame([{
        "Intent":f"{ICONS.get(s['intent'],'')} {s['intent']}",
        "Confidence":round(s["confidence"],3),
        "Sentiment":s["sentiment"],
        "Priority":s["priority"],
        "Sentence":s["text"],
    } for s in filtered])
    st.dataframe(df_v,use_container_width=True,height=440)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — SPEAKER DIARIZATION  (Feature 17)
# ─────────────────────────────────────────────────────────────────────────────
elif "Speaker" in nav:
    st.markdown("## 👤 Speaker Diarization")
    st.caption("Format: `Name: text` (one turn per line) for automatic speaker detection.")
    if not st.session_state.history: st.info("Run a session first."); st.stop()
    record = st.session_state.history[0]
    turns = record.get("speakers",[])
    if not turns:
        st.warning("No speaker labels found. Use `Alice: We decided to use React.` format.")
    else:
        speakers = list({t["speaker"] for t in turns})
        spk_color = {s:SPEAKER_COLORS[i%len(SPEAKER_COLORS)] for i,s in enumerate(speakers)}
        st.markdown("#### Speaker Timeline")
        for t in turns:
            c = spk_color.get(t["speaker"],"#9ca3af")
            label, conf = classify_sentence(t["text"])
            st.markdown(f"""
            <div class="timeline-item">
              <div class="timeline-dot" style="background:{c}"></div>
              <div>
                <span class="speaker-tag" style="color:{c};border-color:{c}40">{t['speaker']}</span>
                {badge_html(label)}
                <span style="font-size:.84rem;color:#c8cfe8;margin-left:.4rem">{t['text'][:90]}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Speaker Stats")
        spk_counts = Counter(t["speaker"] for t in turns)
        spk_df = pd.DataFrame(list(spk_counts.items()),columns=["Speaker","Turns"])
        spk_df["Words"] = [sum(len(t["text"].split()) for t in turns if t["speaker"]==s) for s in spk_df["Speaker"]]
        fig = px.bar(spk_df,x="Speaker",y="Turns",color="Speaker",color_discrete_sequence=SPEAKER_COLORS,text="Words")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",showlegend=False,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ENTITY MAP  (Feature 18)
# ─────────────────────────────────────────────────────────────────────────────
elif "Entity" in nav:
    st.markdown("## 🗂️ Entity Extraction Map")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_ents = [{"text":e["text"],"type":e["type"],"intent":s["intent"],"sentence":s["text"][:55]+"…"}
                for h in st.session_state.history for s in h["sentences"] for e in s["entities"]]
    if not all_ents: st.info("No entities extracted yet."); st.stop()
    df_e = pd.DataFrame(all_ents)
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown("#### By Type")
        ecol = {"PERSON":"#60a5fa","DATE":"#34d399","TECH":"#fbbf24","ORG":"#a78bfa"}
        for et, cnt in df_e["type"].value_counts().items():
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.4rem .8rem;
              background:var(--surface2);border-radius:8px;margin-bottom:.4rem">
              <span style="color:{ecol.get(et,'#9ca3af')}">{et}</span>
              <span style="font-weight:700">{cnt}</span></div>""",unsafe_allow_html=True)
    with c2:
        st.markdown("#### Top Entities")
        freq = df_e["text"].value_counts().head(15).reset_index()
        freq.columns=["entity","count"]
        fig = px.bar(freq,x="count",y="entity",orientation="h",color="count",color_continuous_scale="blues")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",yaxis_title="",showlegend=False,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df_e[["text","type","intent","sentence"]],use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — TOPICS  (Feature 19)
# ─────────────────────────────────────────────────────────────────────────────
elif "Topics" in nav:
    st.markdown("## #️⃣ Topic Analysis")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_text = " ".join(s["text"] for h in st.session_state.history for s in h["sentences"])
    
    st.markdown("#### Word Cloud")
    wc_img = make_wordcloud(all_text)
    if wc_img:
        st.image(wc_img, use_container_width=True)
    else:
        st.info("Install `wordcloud` for word cloud visualization.")
    
    all_topics = [t for h in st.session_state.history for t in h.get("topics",[])]
    if all_topics:
        tf = Counter(all_topics).most_common(25)
        tdf = pd.DataFrame(tf,columns=["topic","freq"])
        st.markdown("#### Topic Treemap")
        fig = px.treemap(tdf,path=["topic"],values="freq",color="freq",color_continuous_scale="blues")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — ACTION TRACKER  (Feature 20)
# ─────────────────────────────────────────────────────────────────────────────
elif "Action Tracker" in nav:
    st.markdown("## ✅ Action Item Tracker")
    st.caption("Auto-extracted from every session. Mark them done or add manually.")

    with st.expander("➕ Add Manual Action Item"):
        new_action = st.text_input("Action Item Text")
        if st.button("Add") and new_action.strip():
            st.session_state.action_items.append({"text":new_action,"done":False,"ts":datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.rerun()

    if not st.session_state.action_items:
        st.info("No action items yet. Run an analysis session first."); st.stop()

    done_count = sum(1 for a in st.session_state.action_items if a["done"])
    total = len(st.session_state.action_items)
    st.progress(done_count/total if total else 0)
    st.caption(f"**{done_count}/{total}** completed")

    for i, item in enumerate(st.session_state.action_items):
        c1, c2, c3 = st.columns([.05,.8,.15])
        with c1:
            checked = st.checkbox("",value=item["done"],key=f"done_{i}")
            st.session_state.action_items[i]["done"] = checked
        with c2:
            style = "text-decoration:line-through;color:#5d6b8a;" if item["done"] else "color:#e8eaf6;"
            st.markdown(f"""
            <div class="action-card">
              <div class="action-check">{'✅' if item['done'] else '⬜'}</div>
              <div>
                <div style="{style}font-size:.88rem;">{item['text']}</div>
                <div style="font-size:.68rem;color:#5d6b8a;margin-top:.2rem">{item['ts']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            if st.button("🗑️",key=f"del_{i}"):
                st.session_state.action_items.pop(i); st.rerun()

    st.divider()
    adf = pd.DataFrame(st.session_state.action_items)
    st.download_button("📥 Export CSV",adf.to_csv(index=False),"action_items.csv","text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — SENTIMENT  (Feature 21)
# ─────────────────────────────────────────────────────────────────────────────
elif "Sentiment" in nav:
    st.markdown("## 😊 Sentiment Analysis")
    if not st.session_state.history: st.info("No data yet."); st.stop()

    sent_data = [{
        "Session":f"S{i+1}","Sentiment":h["sentiment"],"Color":h["sent_color"],
        "Positive":sum(1 for s in h["sentences"] if s["sentiment"]=="Positive"),
        "Neutral":sum(1 for s in h["sentences"] if s["sentiment"]=="Neutral"),
        "Negative":sum(1 for s in h["sentences"] if s["sentiment"]=="Negative"),
    } for i,h in enumerate(reversed(st.session_state.history))]
    sdf = pd.DataFrame(sent_data)

    last = st.session_state.history[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Sentiment", last["sentiment"])
    c2.metric("Positive Sentences", sum(1 for s in last["sentences"] if s["sentiment"]=="Positive"))
    c3.metric("Negative Sentences", sum(1 for s in last["sentences"] if s["sentiment"]=="Negative"))

    st.markdown("#### Sentiment per Session")
    fig = px.bar(sdf,x="Session",y=["Positive","Neutral","Negative"],barmode="stack",
                 color_discrete_map={"Positive":"#06d6a0","Neutral":"#f59e0b","Negative":"#ef4444"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Sentence-level (Latest Session)")
    for s in last["sentences"]:
        st.markdown(f"""
        <div style="display:flex;gap:.6rem;align-items:center;padding:.4rem 0;
             border-bottom:1px solid rgba(255,255,255,.04)">
          <span style="font-size:.72rem;color:{s['sent_color']};font-weight:700;min-width:70px">{s['sentiment']}</span>
          <span style="font-size:.84rem;color:#c8cfe8">{s['text']}</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 9 — COMPARE SESSIONS  (Feature 22)
# ─────────────────────────────────────────────────────────────────────────────
elif "Compare" in nav:
    st.markdown("## 🔄 Session Comparison")
    if len(st.session_state.history) < 2: st.info("Need at least 2 sessions."); st.stop()
    options = [f"S{i+1} — {h['ts']}" for i,h in enumerate(st.session_state.history)]
    col_a, col_b = st.columns(2)
    with col_a: idx_a = st.selectbox("Session A",range(len(options)),format_func=lambda x:options[x],key="cmp_a")
    with col_b: idx_b = st.selectbox("Session B",range(len(options)),format_func=lambda x:options[x],index=min(1,len(options)-1),key="cmp_b")
    ra = st.session_state.history[idx_a]
    rb = st.session_state.history[idx_b]
    ca, cb = st.columns(2)
    def session_card(r, label):
        cnts = Counter(s["intent"] for s in r["sentences"])
        st.markdown(f"#### {label} — {r['ts']}")
        st.markdown(f"""<div class="nlu-card">
          <div style="font-size:.8rem;color:var(--muted);margin-bottom:.4rem">Summary</div>
          <div style="font-size:.85rem">{r['summary'][:200]}</div>
          <div style="margin-top:.7rem;font-size:.78rem;color:{r['sent_color']}">{r['sentiment']}</div>
        </div>""",unsafe_allow_html=True)
        for intent in INTENTS:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.3rem .6rem;
              background:var(--surface2);border-radius:7px;margin-bottom:.3rem;font-size:.82rem">
              <span>{ICONS[intent]} {intent}</span>
              <span style="color:{COLORS[intent]};font-weight:700">{cnts.get(intent,0)}</span>
            </div>""",unsafe_allow_html=True)
    with ca: session_card(ra,"Session A")
    with cb: session_card(rb,"Session B")
    st.markdown("#### Intent Diff (B − A)")
    ca_cnt = Counter(s["intent"] for s in ra["sentences"])
    cb_cnt = Counter(s["intent"] for s in rb["sentences"])
    diff_data = [{"intent":i,"A":ca_cnt.get(i,0),"B":cb_cnt.get(i,0),"diff":cb_cnt.get(i,0)-ca_cnt.get(i,0)} for i in INTENTS]
    ddf = pd.DataFrame(diff_data)
    fig = px.bar(ddf,x="intent",y="diff",color="diff",color_continuous_scale="RdBu",title="B − A  (positive = more in B)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 10 — Q&A CHATBOT  (Feature 23)
# ─────────────────────────────────────────────────────────────────────────────
elif "Q&A" in nav:
    st.markdown("## 🤖 Meeting Q&A Chatbot")
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
            answer = f"**Main topics:** {', '.join('#'+t for t in record['topics'])}"
        elif any(w in p for w in ["sentiment","tone","mood"]):
            answer = f"**Sentiment:** {record['sentiment']}"
        elif any(w in p for w in ["who","speaker","person","attend"]):
            speakers = list({t["speaker"] for t in record.get("speakers",[])})
            answer = "**Speakers:** " + (", ".join(speakers) if speakers else "No speaker labels found.")
        elif any(w in p for w in ["suggest","idea","recommend"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="suggestion"]
            answer = "**Suggestions:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        else:
            hits = [s["text"] for s in record["sentences"] if any(w in s["text"].lower() for w in p.split() if len(w)>3)]
            answer = ("**Relevant sentences:**\n\n" + "\n".join(f"- {h}" for h in hits[:5])) if hits else f"Topics covered: {', '.join(record['topics'][:5])}."
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)
    if st.button("🗑️ Clear Chat"): st.session_state.chat_history=[]; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 11 — TIMELINE  (Feature 24)
# ─────────────────────────────────────────────────────────────────────────────
elif "Timeline" in nav:
    st.markdown("## ⏰ Meeting Timeline")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    record = st.session_state.history[0]
    if record.get("temporal"):
        st.markdown("#### Temporal References Detected")
        for t in record["temporal"]:
            st.markdown(f"""<div class="nlu-card" style="padding:.6rem 1rem">
              <div style="display:flex;justify-content:space-between">
                <span style="color:#f59e0b;font-weight:600">📅 {t['date']}</span>
                <span style="color:#5d6b8a;font-size:.82rem">"{t['expression']}"</span>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No temporal references detected in the latest session.")

    st.markdown("#### Intent Timeline")
    for i, s in enumerate(record["sentences"], 1):
        st.markdown(f"""
        <div class="timeline-item">
          <div class="timeline-dot" style="background:{COLORS.get(s['intent'],'#5d6b8a')}"></div>
          <div style="flex:1">
            <div style="display:flex;align-items:center;gap:.5rem">
              {badge_html(s['intent'])}
              <span style="font-size:.68rem;color:#5d6b8a">#{i}</span>
            </div>
            <div style="font-size:.82rem;color:#c8cfe8;margin-top:.2rem">{s['text'][:100]}</div>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 12 — SMART ALERTS  (Feature 25)
# ─────────────────────────────────────────────────────────────────────────────
elif "Alerts" in nav:
    st.markdown("## ⚠️ Smart Alerts")
    if not st.session_state.alerts:
        st.success("✅ No alerts — everything looks clean!"); st.stop()
    for icon, title, msg, color in st.session_state.alerts:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,.03);border:1px solid {color}50;border-left:4px solid {color};
             border-radius:12px;padding:.9rem 1.2rem;margin-bottom:.7rem">
          <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.3rem">
            <span style="font-size:1.3rem">{icon}</span>
            <span style="font-weight:700;color:{color};font-size:.92rem">{title}</span>
          </div>
          <div style="color:#9ca3af;font-size:.82rem">{msg}</div>
        </div>""", unsafe_allow_html=True)
    if st.button("🗑️ Dismiss All"):
        st.session_state.alerts=[]; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 13 — BENCHMARK  (Feature 26)
# ─────────────────────────────────────────────────────────────────────────────
elif "Benchmark" in nav:
    st.markdown("## 📈 Model Benchmark")
    rouge = {"ROUGE-1":0.3358,"ROUGE-2":0.1065,"ROUGE-L":0.2570,"ROUGE-Lsum":0.2572}
    intent_m = {
        "action item":  {"precision":0.38,"recall":0.80,"f1":0.52},
        "decision":     {"precision":1.00,"recall":0.50,"f1":0.67},
        "informational":{"precision":0.40,"recall":0.40,"f1":0.40},
        "question":     {"precision":0.58,"recall":0.70,"f1":0.64},
        "risk or issue":{"precision":0.83,"recall":0.50,"f1":0.62},
        "suggestion":   {"precision":1.00,"recall":0.60,"f1":0.75},
    }
    st.markdown("#### ROUGE Scores — 400 SAMSum test samples (BART-large-cnn)")
    rc = st.columns(4)
    for col,(k,v) in zip(rc,rouge.items()):
        with col:
            st.markdown(f"""<div class="metric-box">
              <div class="val" style="font-size:1.5rem;color:#3d7fff">{v:.4f}</div>
              <div class="lbl">{k}</div></div>""",unsafe_allow_html=True)
    st.markdown("#### Intent Classification Metrics — DeBERTa v3 zero-shot")
    mdf = pd.DataFrame(intent_m).T.reset_index(); mdf.columns=["Intent","Precision","Recall","F1"]
    fig = go.Figure()
    for m,c in [("Precision","#3d7fff"),("Recall","#06d6a0"),("F1","#f72585")]:
        fig.add_trace(go.Bar(name=m,x=mdf["Intent"],y=mdf[m],marker_color=c))
    fig.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10),xaxis_tickangle=-20)
    st.plotly_chart(fig,use_container_width=True)
    st.progress(0.58); st.caption("Macro Accuracy: **58%** · Macro F1: **0.60** · Weighted F1: **0.60**")
    
    st.markdown("#### Model Architecture Summary")
    arch_data = [
        {"Module":"Intent Classifier","Model":"DeBERTa v3 small (zero-shot)","Task":"6-class NLI","Params":"~184M"},
        {"Module":"NER","Model":"BERT-base-NER (dslim)","Task":"Token Classification","Params":"~110M"},
        {"Module":"Summarizer","Model":"BART-large-cnn","Task":"Seq2Seq Abstractive","Params":"~406M"},
        {"Module":"STT","Model":"Whisper base","Task":"ASR","Params":"~74M"},
        {"Module":"Embeddings","Model":"all-MiniLM-L6-v2","Task":"Semantic Similarity","Params":"~22M"},
    ]
    st.dataframe(pd.DataFrame(arch_data),use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 14 — HISTORY  (Feature 27)
# ─────────────────────────────────────────────────────────────────────────────
elif "History" in nav:
    st.markdown("## 🕒 Session History")
    if not st.session_state.history: st.info("No sessions yet."); st.stop()
    if st.button("🗑️ Clear All History"): st.session_state.history=[]; st.rerun()
    for i, record in enumerate(st.session_state.history):
        counts = Counter(s["intent"] for s in record["sentences"])
        dom = counts.most_common(1)[0][0] if counts else "—"
        with st.expander(f"📋 S{len(st.session_state.history)-i}  ·  {record['ts']}  ·  {record['word_count']} words  ·  {len(record['sentences'])} items"):
            st.markdown(f"**Summary:** {record['summary']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Dominant:** {badge_html(dom)}", unsafe_allow_html=True)
            c2.markdown(f"**Sentiment:** <span style='color:{record['sent_color']}'>{record['sentiment']}</span>", unsafe_allow_html=True)
            c3.markdown(f"**Topics:** {', '.join('#'+t for t in record['topics'][:3])}")
            for s in record["sentences"][:8]:
                st.markdown(f"""
                <div class="timeline-item">
                  <div class="timeline-dot" style="background:{COLORS.get(s['intent'],'#5d6b8a')}"></div>
                  <div>{badge_html(s['intent'])} <span style="font-size:.83rem;color:#c8cfe8;margin-left:.4rem">{s['text']}</span></div>
                </div>""", unsafe_allow_html=True)
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("⭐ Bookmark",key=f"bk_{i}"):
                    st.session_state.bookmarks.append(record); st.success("Bookmarked!")
            with bc2:
                if st.button("🔄 Set for Compare",key=f"cmp_{i}"):
                    if st.session_state.comparison_a is None: st.session_state.comparison_a = record
                    else: st.session_state.comparison_b = record
                    st.success("Set — go to Compare Sessions")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 15 — SETTINGS  (Features 28-29)
# ─────────────────────────────────────────────────────────────────────────────
elif "Settings" in nav:
    st.markdown("## ⚙️ Settings")
    with st.expander("🎨 Appearance",expanded=True):
        st.session_state.theme = st.radio("Theme",["Dark","Light"],horizontal=True)
        st.session_state.lang  = st.selectbox("Language",["English","Arabic","French","Spanish"])
    with st.expander("🧠 Pipeline"):
        st.session_state.auto_summarize       = st.toggle("Auto-Summarization",st.session_state.auto_summarize)
        st.session_state.highlight_entities   = st.toggle("Highlight Entities",st.session_state.highlight_entities)
        st.session_state.min_conf             = st.slider("Min Confidence",0.0,1.0,st.session_state.min_conf,.05)
    with st.expander("🎤 Whisper STT"):
        st.info("Whisper 'base' loads on first transcription. Pre-load here.")
        if st.button("🔄 Pre-load Whisper Now"):
            with st.spinner("Loading…"):
                m = load_whisper()
            st.success("✅ Whisper loaded!" if m else "❌ Whisper unavailable.")
    with st.expander("📋 Session Notes"):
        st.session_state.session_notes = st.text_area("Notes",st.session_state.session_notes,height=130)
        if st.button("💾 Save"): st.success("Saved!")
    with st.expander("⭐ Bookmarks"):
        if not st.session_state.bookmarks: st.info("No bookmarks.")
        else:
            for i,bk in enumerate(st.session_state.bookmarks):
                st.markdown(f"**{i+1}.** {bk['ts']} — {bk['summary'][:70]}…")
            if st.button("🗑️ Clear Bookmarks"): st.session_state.bookmarks=[]; st.rerun()
    with st.expander("⚠️ Danger Zone"):
        if st.button("Reset ALL Data",type="secondary"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 16 — ABOUT  (Feature 30)
# ─────────────────────────────────────────────────────────────────────────────
elif "About" in nav:
    st.markdown("## ℹ️ About")
    st.markdown("""
    <div class="nlu-card">
      <h3 style="margin:0 0 .5rem">🎙️ NLU Meeting Voice-to-Action Assistant v2</h3>
      <p style="color:#9ca3af;font-size:.88rem;line-height:1.75">
        End-to-end meeting intelligence: record or upload audio → Whisper transcription →
        transformer intent classification → NER → summarization → export.<br>
        Built on HuggingFace Transformers · Sentence-BERT · Plotly · Streamlit.
      </p>
    </div>""", unsafe_allow_html=True)

    features = [
        ("🎤 Live Mic Recording","Record directly in-browser via st-audiorec"),
        ("📁 Audio File Upload","WAV·MP3·M4A·OGG·FLAC → Whisper STT"),
        ("🧠 Whisper Transcription","OpenAI Whisper base — same as training notebook"),
        ("🏷️ Intent Classification","6 intents + confidence — DeBERTa zero-shot"),
        ("📝 Auto Summarization","Abstractive (BART) or extractive fallback"),
        ("😊 Sentence Sentiment","Positive/Neutral/Negative per sentence + session"),
        ("🏛️ Named Entity Extraction","PERSON · DATE · TECH · ORG highlighted"),
        ("👤 Speaker Diarization","Auto-detect speakers from Name: text format"),
        ("📊 Analytics Dashboard","Pie · bar · histogram · treemap across sessions"),
        ("🏷️ Intent Explorer","Multi-filter search across all sessions"),
        ("✅ Action Item Tracker","Auto-extract + manual add + progress bar"),
        ("⚠️ Smart Alerts","Auto-flag high risk / many actions / negative tone"),
        ("🤖 Q&A Chatbot","Retrieval-based meeting QA chatbot"),
        ("🔄 Session Compare","Side-by-side diff of two sessions"),
        ("#️⃣ Topic Analysis","Word cloud + treemap of key topics"),
        ("🗂️ Entity Map","Frequency charts for all extracted entities"),
        ("📈 Benchmark","ROUGE + classification metrics from notebooks"),
        ("📄 Export JSON","Full structured export"),
        ("📊 Export CSV","Tabular sentence export"),
        ("📑 Export PDF","Professional report via FPDF"),
        ("📝 Export Markdown","Clean Markdown report"),
        ("⭐ Bookmarks","Save important sessions for later"),
        ("📋 Session Notes","Personal freeform notes per session"),
        ("🔒 Confidence Filter","Global min-confidence threshold slider"),
        ("⚡ Priority Sort","Sort sentences by urgency score"),
        ("⏰ Timeline View","Sequential intent + temporal timeline"),
        ("🕒 Session History","Full history with bookmarking"),
        ("⚙️ Settings","Theme · language · pipeline toggles"),
        ("⚡ Lazy Model Loading","Load HuggingFace models on demand"),
        ("🌐 GitHub Ready","requirements.txt + .streamlit/config.toml included"),
    ]
    c1, c2 = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with (c1 if i%2==0 else c2):
            st.markdown(f"""<div class="nlu-card" style="padding:.8rem 1rem">
              <div style="font-weight:700;font-size:.87rem">{title}</div>
              <div style="color:#5d6b8a;font-size:.76rem;margin-top:.15rem">{desc}</div>
            </div>""",unsafe_allow_html=True)
    st.divider()
    st.caption("v2.0 · Streamlit · Plotly · HuggingFace · Whisper · WordCloud · FPDF · Sentence-BERT")
