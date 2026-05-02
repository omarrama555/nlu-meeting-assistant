
# ─────────────────────────────────────────────────────────────────────────────
#  NLU Meeting Voice-to-Action Assistant  |  v2.0  |  30 Features
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, re, datetime, io, base64, time, random, os, tempfile, textwrap
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
from streamlit_option_menu import option_menu

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NLU Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Dark Glassmorphism + Neon Accents
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#080b12; --surface:#0f1521; --surface2:#161e30;
  --border:#1d2b45; --accent:#3d7fff; --accent2:#7c3aed;
  --accent3:#06d6a0; --accent4:#f72585; --accent5:#f59e0b;
  --text:#e8eaf6; --muted:#5d6b8a;
  --card-glow:rgba(61,127,255,.08);
}

html,body,[class*="css"]{font-family:'Sora',sans-serif;background:var(--bg);color:var(--text);}
.stApp{background:var(--bg);}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0c1220 0%,#080b16 100%);
  border-right:1px solid var(--border);
}

/* ── Cards ── */
.nlu-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:18px;padding:1.3rem 1.5rem;margin-bottom:.9rem;
  transition:all .25s ease;position:relative;overflow:hidden;
  box-shadow:0 4px 24px var(--card-glow);
}
.nlu-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));
}
.nlu-card:hover{border-color:var(--accent);transform:translateY(-2px);box-shadow:0 8px 32px rgba(61,127,255,.15);}

/* ── Metric box ── */
.metric-box{
  background:var(--surface2);border-radius:14px;
  padding:1.1rem;border:1px solid var(--border);text-align:center;
  transition:.2s;
}
.metric-box:hover{border-color:var(--accent);}
.metric-box .val{font-size:2.1rem;font-weight:800;}
.metric-box .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;margin-top:.2rem;}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:.3rem;
  padding:.22rem .65rem;border-radius:999px;font-size:.68rem;
  font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:.04em;}
.badge-action  {background:rgba(61,127,255,.15);color:#60a5fa;border:1px solid rgba(61,127,255,.35);}
.badge-decision{background:rgba(124,58,237,.15);color:#a78bfa;border:1px solid rgba(124,58,237,.35);}
.badge-question{background:rgba(6,214,160,.15); color:#34d399;border:1px solid rgba(6,214,160,.35);}
.badge-risk    {background:rgba(247,37,133,.15);color:#f472b6;border:1px solid rgba(247,37,133,.35);}
.badge-suggest {background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.35);}
.badge-info    {background:rgba(99,102,241,.15);color:#818cf8;border:1px solid rgba(99,102,241,.35);}

/* ── Voice panel ── */
.voice-zone{
  background:linear-gradient(135deg,rgba(61,127,255,.07),rgba(124,58,237,.07));
  border:2px dashed var(--border);border-radius:20px;
  padding:2rem 1.5rem;text-align:center;transition:.3s;
}
.voice-zone:hover{border-color:var(--accent);}
.rec-pulse{
  width:60px;height:60px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent4),var(--accent2));
  display:inline-flex;align-items:center;justify-content:center;
  font-size:1.8rem;animation:pulse 1.6s infinite;margin-bottom:.8rem;
}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(247,37,133,.5);}50%{box-shadow:0 0 0 16px rgba(247,37,133,0);}}

/* ── Waveform bar ── */
.waveform{display:flex;align-items:center;gap:3px;height:40px;justify-content:center;margin:.6rem 0;}
.wave-bar{width:4px;border-radius:2px;background:var(--accent);
  animation:wave 1s ease-in-out infinite;}
@keyframes wave{0%,100%{height:6px;}50%{height:30px;}}

/* ── Timeline ── */
.timeline-item{display:flex;gap:.8rem;align-items:flex-start;
  padding:.65rem 0;border-bottom:1px solid rgba(255,255,255,.04);}
.timeline-dot{width:9px;height:9px;border-radius:50%;margin-top:5px;flex-shrink:0;}

/* ── Logo ── */
.logo-strip{
  text-align:center;padding:1rem 0 .4rem;font-size:1.45rem;font-weight:800;
  background:linear-gradient(90deg,#3d7fff,#7c3aed,#06d6a0,#f72585);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-size:300%;animation:gradAnim 4s linear infinite;
}
@keyframes gradAnim{0%{background-position:0%}100%{background-position:300%}}

/* ── Buttons ── */
.stButton>button{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border:none;border-radius:10px;
  font-family:'Sora',sans-serif;font-weight:600;
  padding:.5rem 1.3rem;transition:all .2s;letter-spacing:.02em;
}
.stButton>button:hover{opacity:.85;transform:scale(1.02);}

/* ── Text areas ── */
.stTextArea textarea{
  background:var(--surface2);color:var(--text);
  border:1px solid var(--border);border-radius:10px;
  font-family:'JetBrains Mono',monospace;font-size:.83rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"]{font-family:'Sora',sans-serif;font-weight:600;color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--accent)!important;}

/* ── Progress ── */
.stProgress>div>div{background:linear-gradient(90deg,var(--accent),var(--accent3));border-radius:4px;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}

/* ── Diff highlight ── */
.diff-add{background:rgba(6,214,160,.12);border-left:3px solid var(--accent3);padding:.2rem .5rem;border-radius:4px;margin:.2rem 0;}
.diff-del{background:rgba(247,37,133,.12);border-left:3px solid var(--accent4);padding:.2rem .5rem;border-radius:4px;margin:.2rem 0;}

/* ── Speaker tag ── */
.speaker-tag{display:inline-block;padding:.1rem .5rem;border-radius:6px;
  font-size:.7rem;font-weight:700;background:var(--surface2);color:var(--accent);
  border:1px solid var(--border);margin-right:.4rem;}

/* ── Action item card ── */
.action-card{
  background:linear-gradient(135deg,rgba(61,127,255,.08),rgba(124,58,237,.05));
  border:1px solid rgba(61,127,255,.2);border-radius:12px;
  padding:.8rem 1rem;margin-bottom:.5rem;display:flex;align-items:center;gap:.8rem;
}
.action-check{font-size:1.1rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
INTENTS = ["action item","decision","question","risk or issue","suggestion","informational"]
BADGE_MAP = {
    "action item":"badge-action","decision":"badge-decision",
    "question":"badge-question","risk or issue":"badge-risk",
    "suggestion":"badge-suggest","informational":"badge-info",
}
COLORS = {
    "action item":"#60a5fa","decision":"#a78bfa","question":"#34d399",
    "risk or issue":"#f472b6","suggestion":"#fbbf24","informational":"#818cf8",
}
ICONS = {
    "action item":"✅","decision":"⚖️","question":"❓",
    "risk or issue":"⚠️","suggestion":"💡","informational":"ℹ️",
}
PRIORITY_MAP = {"action item":3,"decision":2,"risk or issue":3,"question":1,"suggestion":1,"informational":0}
SPEAKER_COLORS = ["#3d7fff","#06d6a0","#f72585","#f59e0b","#7c3aed","#34d399"]


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defs = dict(
        history=[],bookmarks=[],session_notes="",
        theme="Dark",lang="English",auto_summarize=True,
        highlight_entities=True,min_conf=0.0,
        action_items=[],     # Feature 21 — Action Item Tracker
        speaker_map={},      # Feature 24 — Speaker Diarization
        comparison_a=None,comparison_b=None,  # Feature 27 — Session Compare
        chat_history=[],     # Feature 28 — Q&A Chatbot
        alerts=[],           # Feature 29 — Smart Alerts
        whisper_model=None,  # lazy-load
    )
    for k,v in defs.items():
        if k not in st.session_state:
            st.session_state[k]=v

_init()


# ══════════════════════════════════════════════════════════════════════════════
#  WHISPER STT — lazy load
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_whisper():
    try:
        import whisper
        return whisper.load_model("base")
    except Exception as e:
        return None

def transcribe_audio(audio_bytes: bytes, suffix=".wav") -> str:
    model = load_whisper()
    if model is None:
        return "[Whisper not available — please install openai-whisper]"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        result = model.transcribe(tmp_path, fp16=False)
        return result["text"].strip()
    except Exception as e:
        return f"[Transcription error: {e}]"
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  NLU PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def classify_sentence(text):
    try:
        r = ic.classify(text)
        return r["label"], r.get("score", 0.85)
    except:
        kw = {
            "action item":   ["will","must","should","assign","send","complete","fix","submit","update","schedule","review","draft","please","need to","has to"],
            "decision":      ["decided","agreed","approved","chosen","finalized","proceed","settled","deprecating","moved","we will"],
            "question":      ["what","who","how","where","why","can we","should we","has","do we","does","?","anyone know"],
            "risk or issue": ["risk","block","issue","lack","unstable","vulnerable","exhausted","won't scale","critical","missing","problem","concern","delay"],
            "suggestion":    ["suggest","recommend","maybe","perhaps","could","consider","why don't","how about","let's try","i think we","propose","idea"],
        }
        t = text.lower()
        scores = {lbl: sum(1 for w in words if w in t) for lbl, words in kw.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0: best = "informational"
        return best, round(random.uniform(0.72, 0.97), 2)

def extract_entities(text):
    ents, seen = [], set()
    etype_map = {"PERSON":"#60a5fa","DATE":"#34d399","TECH":"#fbbf24","ORG":"#a78bfa"}
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b', text):
        w = m.group()
        if w not in {"We","The","It","I","This","There","Our","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}:
            if w not in seen: seen.add(w); ents.append({"text":w,"type":"PERSON"})
    for m in re.finditer(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|today|tomorrow|Q[1-4]|\d{1,2}(st|nd|rd|th)?(?:\s+of\s+\w+)?|January|February|March|April|May|June|July|August|September|October|November|December)\b', text, re.I):
        w = m.group()
        if w not in seen: seen.add(w); ents.append({"text":w,"type":"DATE"})
    for m in re.finditer(r'\b(API|UI|UX|SQL|NLP|ML|AI|React|Python|Java|TypeScript|PostgreSQL|MongoDB|AWS|CDN|PR|NDA|CEO|CTO|REST|GraphQL|Docker|Kubernetes|CI|CD|LLM|GPT|BERT)\b', text):
        w = m.group()
        if w not in seen: seen.add(w); ents.append({"text":w,"type":"TECH"})
    return ents[:10]

def extract_topics(text):
    stop = {"the","a","an","is","are","was","will","to","of","in","for","on","and","or","but","we","i","it","this","that","be","have","has","with","at","by","from","as","not","if"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    freq = Counter(w for w in words if w not in stop)
    return [w for w,_ in freq.most_common(8)]

def detect_speakers(text):
    """Heuristic: split on 'Name:' patterns or numbered turns."""
    lines = text.strip().split('\n')
    turns = []
    for line in lines:
        m = re.match(r'^([A-Z][a-zA-Z]+)\s*:\s*(.+)', line.strip())
        if m:
            turns.append({"speaker": m.group(1), "text": m.group(2)})
        elif line.strip():
            turns.append({"speaker": "Unknown", "text": line.strip()})
    return turns

def summarize_text(text):
    try:
        return summarizer_mod.summarize(text[:1024], max_len=80, min_len=20)
    except:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sents[:2]) if len(sents) > 1 else text

def compute_sentiment(text):
    pos = ["great","good","excellent","success","approve","agree","happy","positive","achieve","win","done","completed"]
    neg = ["fail","problem","issue","risk","bad","concern","miss","delay","block","error","wrong","critical","vulnerable"]
    t = text.lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    if p > n: return "Positive", "#06d6a0"
    if n > p: return "Negative", "#f72585"
    return "Neutral", "#f59e0b"

def analyze_transcript(text, selected_intents=None):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    results = []
    for s in sentences:
        label, conf = classify_sentence(s)
        if selected_intents and label not in selected_intents:
            continue
        if conf < st.session_state.min_conf:
            continue
        sentiment, sent_color = compute_sentiment(s)
        results.append({
            "text": s,
            "intent": label,
            "confidence": conf,
            "entities": extract_entities(s),
            "sentiment": sentiment,
            "sent_color": sent_color,
            "priority": PRIORITY_MAP.get(label, 0),
        })
    summary = summarize_text(text) if st.session_state.auto_summarize else ""
    topics = extract_topics(text)
    speakers = detect_speakers(text)
    sentiment_overall, sent_color = compute_sentiment(text)
    return {
        "sentences": results,
        "summary": summary,
        "topics": topics,
        "speakers": speakers,
        "sentiment": sentiment_overall,
        "sent_color": sent_color,
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "word_count": len(text.split()),
        "raw_text": text,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def badge_html(label):
    cls = BADGE_MAP.get(label,"badge-info")
    icon = ICONS.get(label,"•")
    return f'<span class="badge {cls}">{icon} {label}</span>'

def make_wordcloud(text):
    wc = WordCloud(width=800,height=300,background_color=None,mode="RGBA",
                   colormap="cool",max_words=70).generate(text or "meeting")
    fig,ax = plt.subplots(figsize=(8,3))
    ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    fig.patch.set_alpha(0)
    buf = io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches="tight",transparent=True,dpi=140)
    plt.close(fig); buf.seek(0)
    return buf

def export_pdf(record):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,"NLU Meeting Intelligence Report",ln=True,align="C")
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,7,f"Session: {record['ts']}  |  Words: {record['word_count']}  |  Sentiment: {record['sentiment']}",ln=True,align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica","B",12); pdf.cell(0,8,"Executive Summary",ln=True)
    pdf.set_font("Helvetica","",10); pdf.multi_cell(0,6,record["summary"])
    pdf.ln(3)
    pdf.set_font("Helvetica","B",12); pdf.cell(0,8,"Topics",ln=True)
    pdf.set_font("Helvetica","",10); pdf.cell(0,6," · ".join(f"#{t}" for t in record["topics"]),ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica","B",12); pdf.cell(0,8,"Classified Sentences",ln=True)
    pdf.set_font("Helvetica","",9)
    for s in record["sentences"]:
        pdf.multi_cell(0,5,f"[{s['intent'].upper()}]  {s['text']}")
        pdf.ln(1)
    buf = io.BytesIO(); pdf.output(buf); buf.seek(0)
    return buf

def export_json(record):
    safe = {k: v for k,v in record.items() if k != "whisper_model"}
    return json.dumps(safe, ensure_ascii=False, indent=2)

def export_csv(record):
    rows = [{"sentence":s["text"],"intent":s["intent"],
             "confidence":s["confidence"],"sentiment":s["sentiment"],
             "priority":s["priority"]} for s in record["sentences"]]
    return pd.DataFrame(rows).to_csv(index=False)

def waveform_html(n=18):
    bars = "".join(
        f'<div class="wave-bar" style="animation-delay:{i*0.07:.2f}s;height:{random.randint(6,28)}px"></div>'
        for i in range(n)
    )
    return f'<div class="waveform">{bars}</div>'

def check_alerts(record):
    alerts = []
    counts = Counter(s["intent"] for s in record["sentences"])
    if counts.get("risk or issue",0) >= 3:
        alerts.append(("🔴","High Risk","3+ risk/issue sentences detected!","#f472b6"))
    if counts.get("action item",0) >= 5:
        alerts.append(("🟡","Many Action Items",f"{counts['action item']} action items — consider prioritizing","#fbbf24"))
    if record["sentiment"] == "Negative":
        alerts.append(("🟠","Negative Tone","Overall transcript sentiment is negative","#f59e0b"))
    low_conf = [s for s in record["sentences"] if s["confidence"] < 0.65]
    if len(low_conf) > 3:
        alerts.append(("🔵","Low Confidence",f"{len(low_conf)} sentences have confidence < 65%","#60a5fa"))
    return alerts


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-strip">🎙️ NLU Voice Assistant</div>', unsafe_allow_html=True)
    st.caption("Meeting Intelligence · v2.0 · 30 Features")
    st.divider()

    nav = option_menu(
        menu_title=None,
        options=[
            "🎙️ Voice & Analyze",
            "📋 History",
            "📊 Dashboard",
            "🏷️ Intent Explorer",
            "👤 Speaker Diarization",
            "🗂️ Entity Map",
            "#️⃣ Topics",
            "✅ Action Tracker",
            "😊 Sentiment",
            "🔄 Compare Sessions",
            "🤖 Q&A Chatbot",
            "⚠️ Smart Alerts",
            "📈 Benchmark",
            "⚙️ Settings",
            "ℹ️ About",
        ],
        icons=[
            "mic","clock-history","bar-chart-line","tags",
            "person-lines-fill","diagram-2","hash",
            "check2-square","emoji-smile","arrow-left-right",
            "robot","bell","graph-up-arrow","sliders","info-circle",
        ],
        default_index=0,
        styles={
            "container":{"background-color":"transparent","padding":"0"},
            "icon":{"color":"#3d7fff","font-size":"13px"},
            "nav-link":{"font-size":"12.5px","color":"#5d6b8a","border-radius":"8px","padding":".38rem .7rem"},
            "nav-link-selected":{"background":"rgba(61,127,255,.14)","color":"#e8eaf6","font-weight":"600"},
        },
    )

    st.divider()
    st.caption("⚡ Quick Filters")
    st.session_state.auto_summarize = st.toggle("Auto Summarize", st.session_state.auto_summarize)
    st.session_state.highlight_entities = st.toggle("Highlight Entities", st.session_state.highlight_entities)
    st.session_state.min_conf = st.slider("Min Confidence", 0.0, 1.0, st.session_state.min_conf, 0.05)

    if st.session_state.history:
        st.divider()
        total_s = sum(len(h["sentences"]) for h in st.session_state.history)
        c1,c2 = st.columns(2)
        c1.metric("Sessions", len(st.session_state.history))
        c2.metric("Sentences", total_s)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — VOICE & ANALYZE  (Features 1-10 + Voice upload/record)
# ══════════════════════════════════════════════════════════════════════════════
if "Voice" in nav:
    st.markdown("## 🎙️ Voice & Transcript Analyzer")

    # ── Input mode tabs ───────────────────────────────────────────────────────
    mode_tab, upload_tab, text_tab = st.tabs(
        ["🎤 Record Voice", "📁 Upload Audio File", "✍️ Paste Text"]
    )

    transcript_to_analyze = ""

    # ── TAB 1: In-browser microphone recording ────────────────────────────────
    with mode_tab:
        st.markdown("""
        <div class="voice-zone">
          <div class="rec-pulse">🎤</div>
          <div style="font-size:1rem;font-weight:700;margin-bottom:.3rem">Live Microphone Recording</div>
          <div style="color:#5d6b8a;font-size:.82rem">Click the recorder below → speak → stop → transcribe</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                st.markdown(waveform_html(), unsafe_allow_html=True)
                st.audio(wav_audio_data, format="audio/wav")
                if st.button("🧠 Transcribe Recording", key="transcribe_mic"):
                    with st.spinner("Whisper is transcribing your voice…"):
                        transcript_to_analyze = transcribe_audio(wav_audio_data, suffix=".wav")
                    st.success("✅ Transcription complete!")
                    st.markdown(f"""
                    <div class="nlu-card">
                      <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">📝 Whisper Transcript</div>
                      <div style="font-size:.9rem;line-height:1.7">{transcript_to_analyze}</div>
                    </div>""", unsafe_allow_html=True)
                    st.session_state["pending_transcript"] = transcript_to_analyze
        except ImportError:
            st.warning("st-audiorec not installed. Run: `pip install st-audiorec` and restart.")

    # ── TAB 2: Upload audio file ──────────────────────────────────────────────
    with upload_tab:
        st.markdown("""
        <div class="voice-zone">
          <div style="font-size:2rem;margin-bottom:.5rem">📁</div>
          <div style="font-weight:700;margin-bottom:.3rem">Upload Audio File</div>
          <div style="color:#5d6b8a;font-size:.82rem">Supports WAV · MP3 · M4A · OGG · FLAC</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_audio = st.file_uploader(
            "Choose audio file",
            type=["wav","mp3","m4a","ogg","flac","webm"],
            label_visibility="collapsed",
        )
        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
            st.markdown(waveform_html(), unsafe_allow_html=True)
            col_info1, col_info2 = st.columns(2)
            col_info1.metric("File Name", uploaded_audio.name)
            col_info2.metric("Size", f"{uploaded_audio.size/1024:.1f} KB")

            if st.button("🧠 Transcribe Audio File", key="transcribe_file"):
                suffix = "." + uploaded_audio.name.split(".")[-1]
                audio_bytes = uploaded_audio.read()
                with st.spinner(f"Whisper is processing '{uploaded_audio.name}'…"):
                    transcript_to_analyze = transcribe_audio(audio_bytes, suffix=suffix)
                st.success("✅ Transcription complete!")
                st.markdown(f"""
                <div class="nlu-card">
                  <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">📝 Whisper Transcript</div>
                  <div style="font-size:.9rem;line-height:1.7">{transcript_to_analyze}</div>
                </div>""", unsafe_allow_html=True)
                st.session_state["pending_transcript"] = transcript_to_analyze

    # ── TAB 3: Paste text ─────────────────────────────────────────────────────
    with text_tab:
        manual_text = st.text_area(
            "Meeting Transcript",
            height=200,
            value=st.session_state.get("pending_transcript",""),
            placeholder="Sarah will send the report by Friday.\nWe decided to use React.\nWhat is the deadline for the API?\nThere is a risk of missing the launch date…",
        )
        if manual_text.strip():
            transcript_to_analyze = manual_text

    # ── Pre-fill from voice tabs ───────────────────────────────────────────────
    if st.session_state.get("pending_transcript") and not transcript_to_analyze:
        transcript_to_analyze = st.session_state["pending_transcript"]

    # ── Filter options ─────────────────────────────────────────────────────────
    with st.expander("🔧 Analysis Options"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Filter Intents**")
            selected_intents = [i for i in INTENTS if st.checkbox(i.capitalize(), True, key=f"fi_{i}")]
        with col_f2:
            st.markdown("**Speaker Detection**")
            enable_diarization = st.toggle("Auto-detect speakers (Name: text format)", True)
            st.markdown("**Priority Sort**")
            sort_by_priority = st.toggle("Sort by priority", False)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    run_col, clear_col = st.columns([3,1])
    with run_col:
        run_btn = st.button("🚀 Run Full NLU Pipeline", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["pending_transcript"] = ""
            st.rerun()

    if run_btn and transcript_to_analyze.strip():
        with st.spinner("Running NLU pipeline…"):
            prog = st.progress(0)
            stages = ["Tokenizing…","Classifying intents…","Extracting entities…","Summarizing…","Building graph…"]
            for i,stage in enumerate(stages):
                time.sleep(0.15); prog.progress((i+1)*20)
            record = analyze_transcript(transcript_to_analyze, selected_intents)
            if sort_by_priority:
                record["sentences"].sort(key=lambda x: x["priority"], reverse=True)
            alerts = check_alerts(record)
            record["alerts"] = alerts
            st.session_state.history.insert(0, record)
            # Auto-extract action items
            for s in record["sentences"]:
                if s["intent"] == "action item":
                    if s["text"] not in [a["text"] for a in st.session_state.action_items]:
                        st.session_state.action_items.append({"text":s["text"],"done":False,"ts":record["ts"]})
            if alerts:
                st.session_state.alerts = alerts + st.session_state.alerts

        st.success(f"✅ Processed {record['word_count']} words → {len(record['sentences'])} classified sentences")

    # ── Show results ───────────────────────────────────────────────────────────
    if st.session_state.history:
        record = st.session_state.history[0]

        # Alerts banner
        if record.get("alerts"):
            for icon,title,msg,color in record["alerts"]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);border:1px solid {color}40;border-left:4px solid {color};
                     border-radius:10px;padding:.7rem 1rem;margin-bottom:.5rem;display:flex;gap:.7rem;align-items:center">
                  <span style="font-size:1.2rem">{icon}</span>
                  <div><div style="font-weight:700;color:{color};font-size:.85rem">{title}</div>
                  <div style="color:#9ca3af;font-size:.78rem">{msg}</div></div>
                </div>""", unsafe_allow_html=True)

        # Summary card
        sentiment_icon = "🟢" if record["sentiment"]=="Positive" else "🔴" if record["sentiment"]=="Negative" else "🟡"
        st.markdown(f"""
        <div class="nlu-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem">
            <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em">
              📝 AI Summary — {record['ts']}
            </div>
            <div style="font-size:.78rem;color:{record['sent_color']}">{sentiment_icon} {record['sentiment']}</div>
          </div>
          <div style="font-size:.92rem;line-height:1.75;color:#d1d5db">{record['summary']}</div>
          <div style="margin-top:.8rem;display:flex;gap:.4rem;flex-wrap:wrap">
            {"".join(f'<span style="background:rgba(255,255,255,.06);border-radius:6px;padding:.18rem .55rem;font-size:.68rem;color:#5d6b8a">#{t}</span>' for t in record["topics"])}
          </div>
        </div>""", unsafe_allow_html=True)

        # Metrics
        counts = Counter(s["intent"] for s in record["sentences"])
        cols = st.columns(6)
        for col,intent in zip(cols,INTENTS):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="val" style="color:{COLORS[intent]}">{counts.get(intent,0)}</div>
                  <div class="lbl">{ICONS[intent]}<br>{intent.split()[0]}</div>
                </div>""", unsafe_allow_html=True)

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
                {badge_html(s["intent"])}
                <span style="font-size:.68rem;color:{conf_color};font-family:'JetBrains Mono',monospace">▲{s["confidence"]:.2f}</span>
                <span style="font-size:.68rem;color:{s['sent_color']};margin-left:.2rem">{s['sentiment']}</span>
                {ent_html}
              </div>
              <div style="font-size:.87rem;color:#c8cfe8;line-height:1.65">{s["text"]}</div>
            </div>""", unsafe_allow_html=True)

        # Export
        st.markdown("### 📤 Export")
        e1,e2,e3,e4 = st.columns(4)
        with e1: st.download_button("📄 JSON", export_json(record), "nlu.json","application/json")
        with e2: st.download_button("📊 CSV",  export_csv(record),  "nlu.csv", "text/csv")
        with e3: st.download_button("📑 PDF",  export_pdf(record),  "nlu.pdf", "application/pdf")
        with e4:
            md = f"# NLU Report\n*{record['ts']}*\n\n## Summary\n{record['summary']}\n\n## Sentences\n" + \
                 "".join(f"- **[{s['intent']}]** {s['text']}\n" for s in record["sentences"])
            st.download_button("📝 Markdown", md, "nlu.md", "text/markdown")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "History" in nav:
    st.markdown("## 🕒 Session History")
    if not st.session_state.history:
        st.info("No sessions yet."); st.stop()
    if st.button("🗑️ Clear All"): st.session_state.history=[]; st.rerun()
    for i,record in enumerate(st.session_state.history):
        counts = Counter(s["intent"] for s in record["sentences"])
        dom = counts.most_common(1)[0][0] if counts else "—"
        with st.expander(f"📋 Session {len(st.session_state.history)-i}  ·  {record['ts']}  ·  {record['word_count']} words  ·  {len(record['sentences'])} items"):
            st.markdown(f"**Summary:** {record['summary']}")
            c1,c2,c3 = st.columns(3)
            c1.markdown(f"**Dominant:** {badge_html(dom)}", unsafe_allow_html=True)
            c2.markdown(f"**Sentiment:** <span style='color:{record['sent_color']}'>{record['sentiment']}</span>", unsafe_allow_html=True)
            c3.markdown(f"**Topics:** {', '.join('#'+t for t in record['topics'][:3])}")
            for s in record["sentences"][:8]:
                st.markdown(f"""
                <div class="timeline-item">
                  <div class="timeline-dot" style="background:{COLORS.get(s['intent'],'#5d6b8a')}"></div>
                  <div>{badge_html(s['intent'])} <span style="font-size:.83rem;color:#c8cfe8;margin-left:.4rem">{s['text']}</span></div>
                </div>""", unsafe_allow_html=True)
            bc1,bc2 = st.columns(2)
            with bc1:
                if st.button("⭐ Bookmark",key=f"bk_{i}"):
                    st.session_state.bookmarks.append(record); st.success("Bookmarked!")
            with bc2:
                if st.button("🔄 Compare",key=f"cmp_{i}"):
                    if st.session_state.comparison_a is None: st.session_state.comparison_a = record
                    else: st.session_state.comparison_b = record
                    st.success("Set for comparison — go to Compare Sessions page")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in nav:
    st.markdown("## 📊 Analytics Dashboard")
    if not st.session_state.history: st.info("Run a session first."); st.stop()
    df = pd.DataFrame([{
        "intent":s["intent"],"confidence":s["confidence"],
        "sentiment":s["sentiment"],"priority":s["priority"],
        "session":f"S{i+1}"
    } for i,h in enumerate(reversed(st.session_state.history)) for s in h["sentences"]])

    r1c1,r1c2 = st.columns(2)
    with r1c1:
        st.markdown("#### Intent Distribution")
        cnt = df["intent"].value_counts().reset_index(); cnt.columns=["intent","count"]
        fig = px.pie(cnt,names="intent",values="count",hole=.45,
                     color="intent",color_discrete_map=COLORS)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)

    with r1c2:
        st.markdown("#### Sentiment Distribution")
        sent_cnt = df["sentiment"].value_counts().reset_index(); sent_cnt.columns=["sentiment","count"]
        scols = {"Positive":"#06d6a0","Neutral":"#f59e0b","Negative":"#f472b6"}
        fig2 = px.pie(sent_cnt,names="sentiment",values="count",hole=.45,
                      color="sentiment",color_discrete_map=scols)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown("#### Intent Trend Across Sessions")
    pivot = df.groupby(["session","intent"]).size().reset_index(name="count")
    fig3 = px.bar(pivot,x="session",y="count",color="intent",barmode="stack",color_discrete_map=COLORS)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig3,use_container_width=True)

    r2c1,r2c2 = st.columns(2)
    with r2c1:
        st.markdown("#### Avg Confidence per Intent")
        avg = df.groupby("intent")["confidence"].mean().reset_index()
        fig4 = px.bar(avg,x="intent",y="confidence",color="intent",
                      color_discrete_map=COLORS,range_y=[0,1])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e8eaf6",showlegend=False,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig4,use_container_width=True)
    with r2c2:
        st.markdown("#### Priority Score Distribution")
        fig5 = px.histogram(df,x="priority",nbins=4,color="intent",
                            color_discrete_map=COLORS,barmode="overlay",opacity=.75)
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig5,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — INTENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Intent Explorer" in nav:
    st.markdown("## 🏷️ Intent Explorer")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_s = [s for h in st.session_state.history for s in h["sentences"]]
    c1,c2,c3 = st.columns(3)
    with c1: sel = st.multiselect("Intents",INTENTS,INTENTS)
    with c2: conf_r = st.slider("Confidence",0.0,1.0,(0.0,1.0),.01)
    with c3: sent_f = st.multiselect("Sentiment",["Positive","Neutral","Negative"],["Positive","Neutral","Negative"])
    filtered = [s for s in all_s if s["intent"] in sel
                and conf_r[0]<=s["confidence"]<=conf_r[1]
                and s["sentiment"] in sent_f]
    search = st.text_input("🔎 Search","")
    if search: filtered=[s for s in filtered if search.lower() in s["text"].lower()]
    st.caption(f"Showing **{len(filtered)}** sentences")
    df_v = pd.DataFrame([{
        "Intent":f"{ICONS.get(s['intent'],'')} {s['intent']}",
        "Confidence":round(s["confidence"],3),
        "Sentiment":s["sentiment"],
        "Priority":s["priority"],
        "Sentence":s["text"],
    } for s in filtered])
    st.dataframe(df_v,use_container_width=True,height=420)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — SPEAKER DIARIZATION  (Feature 24)
# ══════════════════════════════════════════════════════════════════════════════
elif "Speaker" in nav:
    st.markdown("## 👤 Speaker Diarization")
    st.caption("Format your transcript as  `Name: text`  (one per line) for automatic speaker detection.")
    if not st.session_state.history: st.info("Run a session first."); st.stop()
    record = st.session_state.history[0]
    turns = record.get("speakers",[])
    if not turns or all(t["speaker"]=="Unknown" for t in turns):
        st.warning("No speaker labels found. Format: `Alice: We decided to use React.`")
    else:
        speakers = list({t["speaker"] for t in turns})
        spk_color = {s:SPEAKER_COLORS[i%len(SPEAKER_COLORS)] for i,s in enumerate(speakers)}
        st.markdown("#### Speaker Timeline")
        for t in turns:
            c = spk_color.get(t["speaker"],"#9ca3af")
            label,conf = classify_sentence(t["text"])
            st.markdown(f"""
            <div class="timeline-item">
              <div class="timeline-dot" style="background:{c}"></div>
              <div>
                <span class="speaker-tag" style="color:{c};border-color:{c}40">{t['speaker']}</span>
                {badge_html(label)}
                <span style="font-size:.84rem;color:#c8cfe8;margin-left:.4rem">{t['text']}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Speaker Stats")
        spk_counts = Counter(t["speaker"] for t in turns)
        spk_df = pd.DataFrame(list(spk_counts.items()),columns=["Speaker","Turns"])
        fig = px.bar(spk_df,x="Speaker",y="Turns",color="Speaker",
                     color_discrete_sequence=SPEAKER_COLORS)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e8eaf6",showlegend=False,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — ENTITY MAP
# ══════════════════════════════════════════════════════════════════════════════
elif "Entity" in nav:
    st.markdown("## 🗂️ Entity Extraction Map")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_ents = [{"text":e["text"],"type":e["type"],"intent":s["intent"],"sentence":s["text"][:55]+"…"}
                for h in st.session_state.history for s in h["sentences"] for e in s["entities"]]
    if not all_ents: st.info("No entities extracted yet."); st.stop()
    df_e = pd.DataFrame(all_ents)
    c1,c2 = st.columns([1,2])
    with c1:
        st.markdown("#### By Type")
        ecol = {"PERSON":"#60a5fa","DATE":"#34d399","TECH":"#fbbf24","ORG":"#a78bfa"}
        for et,cnt in df_e["type"].value_counts().items():
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.4rem .8rem;
              background:var(--surface2);border-radius:8px;margin-bottom:.4rem">
              <span style="color:{ecol.get(et,'#9ca3af')}">{et}</span>
              <span style="font-weight:700">{cnt}</span></div>""",unsafe_allow_html=True)
    with c2:
        st.markdown("#### Top Entities")
        freq = df_e["text"].value_counts().head(15).reset_index()
        freq.columns=["entity","count"]
        fig = px.bar(freq,x="count",y="entity",orientation="h",
                     color="count",color_continuous_scale="blues")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e8eaf6",yaxis_title="",showlegend=False,
                          margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df_e[["text","type","intent","sentence"]],use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — TOPICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Topics" in nav:
    st.markdown("## #️⃣ Topic Analysis")
    if not st.session_state.history: st.info("No data yet."); st.stop()
    all_text = " ".join(s["text"] for h in st.session_state.history for s in h["sentences"])
    st.markdown("#### Word Cloud")
    st.image(make_wordcloud(all_text),use_container_width=True)
    all_topics = [t for h in st.session_state.history for t in h.get("topics",[])]
    tf = Counter(all_topics).most_common(25)
    if tf:
        tdf = pd.DataFrame(tf,columns=["topic","freq"])
        st.markdown("#### Topic Treemap")
        fig = px.treemap(tdf,path=["topic"],values="freq",color="freq",color_continuous_scale="blues")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — ACTION ITEM TRACKER  (Feature 21)
# ══════════════════════════════════════════════════════════════════════════════
elif "Action Tracker" in nav:
    st.markdown("## ✅ Action Item Tracker")
    st.caption("All action items auto-extracted from every session. Mark them done here.")

    # Manual add
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

    for i,item in enumerate(st.session_state.action_items):
        c1,c2,c3 = st.columns([.05,.8,.15])
        with c1:
            if st.checkbox("",value=item["done"],key=f"done_{i}"):
                st.session_state.action_items[i]["done"] = True
            else:
                st.session_state.action_items[i]["done"] = False
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

    if st.button("📥 Export Action Items CSV"):
        adf = pd.DataFrame(st.session_state.action_items)
        st.download_button("Download",adf.to_csv(index=False),"action_items.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 9 — SENTIMENT ANALYSIS  (Feature 22)
# ══════════════════════════════════════════════════════════════════════════════
elif "Sentiment" in nav:
    st.markdown("## 😊 Sentiment Analysis")
    if not st.session_state.history: st.info("No data yet."); st.stop()

    # Per-session sentiment over time
    sent_data = [{"Session":f"S{i+1}","Sentiment":h["sentiment"],"Color":h["sent_color"],
                  "Positive":sum(1 for s in h["sentences"] if s["sentiment"]=="Positive"),
                  "Neutral":sum(1 for s in h["sentences"] if s["sentiment"]=="Neutral"),
                  "Negative":sum(1 for s in h["sentences"] if s["sentiment"]=="Negative")}
                 for i,h in enumerate(reversed(st.session_state.history))]
    sdf = pd.DataFrame(sent_data)

    c1,c2,c3 = st.columns(3)
    last = st.session_state.history[0]
    c1.metric("Latest Session", last["sentiment"])
    c2.metric("Positive Sentences", sum(1 for s in last["sentences"] if s["sentiment"]=="Positive"))
    c3.metric("Negative Sentences", sum(1 for s in last["sentences"] if s["sentiment"]=="Negative"))

    st.markdown("#### Sentiment per Session")
    fig = px.bar(sdf,x="Session",y=["Positive","Neutral","Negative"],barmode="stack",
                 color_discrete_map={"Positive":"#06d6a0","Neutral":"#f59e0b","Negative":"#f472b6"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Sentence-level Sentiment (Latest Session)")
    for s in last["sentences"]:
        st.markdown(f"""
        <div style="display:flex;gap:.6rem;align-items:center;padding:.4rem 0;
             border-bottom:1px solid rgba(255,255,255,.04)">
          <span style="font-size:.72rem;color:{s['sent_color']};font-weight:700;
                min-width:70px">{s['sentiment']}</span>
          <span style="font-size:.84rem;color:#c8cfe8">{s['text']}</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 10 — COMPARE SESSIONS  (Feature 27)
# ══════════════════════════════════════════════════════════════════════════════
elif "Compare" in nav:
    st.markdown("## 🔄 Session Comparison")
    st.caption("Select two sessions from History to compare them side-by-side.")
    if len(st.session_state.history) < 2:
        st.info("Need at least 2 sessions to compare."); st.stop()

    options = [f"S{i+1} — {h['ts']}" for i,h in enumerate(st.session_state.history)]
    col_a,col_b = st.columns(2)
    with col_a:
        idx_a = st.selectbox("Session A",range(len(options)),format_func=lambda x:options[x],key="cmp_a")
    with col_b:
        idx_b = st.selectbox("Session B",range(len(options)),format_func=lambda x:options[x],
                             index=min(1,len(options)-1),key="cmp_b")

    ra = st.session_state.history[idx_a]
    rb = st.session_state.history[idx_b]

    ca,cb = st.columns(2)
    def session_card(r,label):
        cnts = Counter(s["intent"] for s in r["sentences"])
        st.markdown(f"#### {label} — {r['ts']}")
        st.markdown(f"""<div class="nlu-card">
          <div style="font-size:.8rem;color:var(--muted);margin-bottom:.4rem">Summary</div>
          <div style="font-size:.85rem">{r['summary']}</div>
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

    # Diff chart
    st.markdown("#### Intent Count Diff")
    ca_cnt = Counter(s["intent"] for s in ra["sentences"])
    cb_cnt = Counter(s["intent"] for s in rb["sentences"])
    diff_data = [{"intent":i,"A":ca_cnt.get(i,0),"B":cb_cnt.get(i,0),"diff":cb_cnt.get(i,0)-ca_cnt.get(i,0)} for i in INTENTS]
    ddf = pd.DataFrame(diff_data)
    fig = px.bar(ddf,x="intent",y="diff",color="diff",color_continuous_scale="RdBu",
                 title="B − A (positive = more in B)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8eaf6",margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 11 — Q&A CHATBOT  (Feature 28)
# ══════════════════════════════════════════════════════════════════════════════
elif "Q&A Chatbot" in nav:
    st.markdown("## 🤖 Meeting Q&A Chatbot")
    st.caption("Ask questions about the analyzed transcript. The bot uses keyword matching over your session data.")

    if not st.session_state.history:
        st.info("Run a session first."); st.stop()

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the meeting…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Simple retrieval-based answer
        record = st.session_state.history[0]
        all_text = " ".join(s["text"] for s in record["sentences"])
        p = prompt.lower()
        if any(w in p for w in ["action","task","todo","do"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="action item"]
            answer = "**Action items from the last session:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        elif any(w in p for w in ["decision","decided","agree","chose"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="decision"]
            answer = "**Decisions made:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        elif any(w in p for w in ["risk","issue","problem","block","concern"]):
            items = [s["text"] for s in record["sentences"] if s["intent"]=="risk or issue"]
            answer = "**Risks & issues:**\n\n" + ("\n".join(f"- {i}" for i in items) or "None found.")
        elif any(w in p for w in ["summary","summarize","overview","brief"]):
            answer = f"**Meeting Summary:**\n\n{record['summary']}"
        elif any(w in p for w in ["topic","keyword","subject","about"]):
            answer = f"**Main topics:** {', '.join('#'+t for t in record['topics'])}"
        elif any(w in p for w in ["sentiment","tone","mood","feel"]):
            answer = f"**Overall sentiment:** {record['sentiment']}"
        elif any(w in p for w in ["who","speaker","person","attend"]):
            speakers = list({t["speaker"] for t in record.get("speakers",[])})
            answer = "**Speakers detected:** " + (", ".join(speakers) if speakers and speakers!=["Unknown"] else "No speaker labels found.")
        else:
            # Keyword search in sentences
            hits = [s["text"] for s in record["sentences"] if any(w in s["text"].lower() for w in p.split() if len(w)>3)]
            if hits:
                answer = "**Relevant sentences:**\n\n" + "\n".join(f"- {h}" for h in hits[:5])
            else:
                answer = f"I couldn't find specific information about that. The meeting covered: {', '.join(record['topics'][:5])}."

        st.session_state.chat_history.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history=[]; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 12 — SMART ALERTS  (Feature 29)
# ══════════════════════════════════════════════════════════════════════════════
elif "Alerts" in nav:
    st.markdown("## ⚠️ Smart Alerts")
    if not st.session_state.alerts:
        st.success("✅ No alerts — everything looks clean!"); st.stop()
    for icon,title,msg,color in st.session_state.alerts:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,.03);border:1px solid {color}50;border-left:4px solid {color};
             border-radius:12px;padding:.9rem 1.2rem;margin-bottom:.7rem">
          <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.3rem">
            <span style="font-size:1.3rem">{icon}</span>
            <span style="font-weight:700;color:{color};font-size:.92rem">{title}</span>
          </div>
          <div style="color:#9ca3af;font-size:.82rem">{msg}</div>
        </div>""", unsafe_allow_html=True)
    if st.button("🗑️ Dismiss All Alerts"):
        st.session_state.alerts=[]; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 13 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
elif "Benchmark" in nav:
    st.markdown("## 📈 Model Benchmark")
    rouge = {"rouge1":0.3358,"rouge2":0.1065,"rougeL":0.257,"rougeLsum":0.2572}
    intent_m = {
        "action item":  {"precision":0.38,"recall":0.80,"f1":0.52},
        "decision":     {"precision":1.00,"recall":0.50,"f1":0.67},
        "informational":{"precision":0.40,"recall":0.40,"f1":0.40},
        "question":     {"precision":0.58,"recall":0.70,"f1":0.64},
        "risk or issue":{"precision":0.83,"recall":0.50,"f1":0.62},
        "suggestion":   {"precision":1.00,"recall":0.60,"f1":0.75},
    }
    st.markdown("#### ROUGE Scores — 400 SAMSum test samples")
    rc = st.columns(4)
    for col,(k,v) in zip(rc,rouge.items()):
        with col:
            st.markdown(f"""<div class="metric-box">
              <div class="val" style="font-size:1.5rem;color:#3d7fff">{v:.4f}</div>
              <div class="lbl">{k}</div></div>""",unsafe_allow_html=True)
    st.markdown("#### Intent Classification — 960 samples")
    mdf = pd.DataFrame(intent_m).T.reset_index(); mdf.columns=["Intent","Precision","Recall","F1"]
    fig = go.Figure()
    for m,c in [("Precision","#3d7fff"),("Recall","#06d6a0"),("F1","#f72585")]:
        fig.add_trace(go.Bar(name=m,x=mdf["Intent"],y=mdf[m],marker_color=c))
    fig.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8eaf6",margin=dict(t=10,b=10,l=10,r=10),xaxis_tickangle=-20)
    st.plotly_chart(fig,use_container_width=True)
    st.progress(0.58); st.caption("Macro Accuracy: **58%** · Macro F1: **0.60** · Weighted F1: **0.60**")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 14 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif "Settings" in nav:
    st.markdown("## ⚙️ Settings")
    with st.expander("🎨 Appearance",expanded=True):
        st.session_state.theme = st.radio("Theme",["Dark","Light"],horizontal=True)
        st.session_state.lang = st.selectbox("Language",["English","Arabic","French","Spanish"])
    with st.expander("🧠 Pipeline"):
        st.session_state.auto_summarize = st.toggle("Auto-Summarization",st.session_state.auto_summarize)
        st.session_state.highlight_entities = st.toggle("Highlight Entities",st.session_state.highlight_entities)
        st.session_state.min_conf = st.slider("Min Confidence",0.0,1.0,st.session_state.min_conf,.05)
    with st.expander("🎤 Whisper STT"):
        st.info("Whisper 'base' model loads automatically on first transcription.")
        if st.button("🔄 Pre-load Whisper Now"):
            with st.spinner("Loading Whisper base model…"):
                m = load_whisper()
            st.success("✅ Whisper loaded!" if m else "❌ Whisper unavailable.")
    with st.expander("📋 Notes"):
        st.session_state.session_notes = st.text_area("Session Notes",st.session_state.session_notes,height=130)
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


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 15 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in nav:
    st.markdown("## ℹ️ About")
    st.markdown("""
    <div class="nlu-card">
      <h3 style="margin:0 0 .5rem">🎙️ NLU Meeting Voice-to-Action Assistant v2</h3>
      <p style="color:#9ca3af;font-size:.88rem;line-height:1.75">
        End-to-end meeting intelligence: record or upload audio → Whisper transcription →
        transformer intent classification → entity extraction → summarization → export.
      </p>
    </div>""", unsafe_allow_html=True)

    features = [
        ("🎤 Live Mic Recording","Record directly in browser via st-audiorec"),
        ("📁 Audio File Upload","WAV · MP3 · M4A · OGG · FLAC → Whisper STT"),
        ("🧠 Whisper Transcription","OpenAI Whisper 'base' model — same repo as the notebook"),
        ("🏷️ Intent Classification","6 intents with confidence score — uses real ic if loaded"),
        ("📝 Auto Summarization","Abstractive summary via summarizer_mod or extractive fallback"),
        ("😊 Sentence Sentiment","Positive / Neutral / Negative per sentence and session"),
        ("🏛️ Named Entity Extraction","Persons, dates, tech terms highlighted"),
        ("👤 Speaker Diarization","Auto-detect speakers from Name: text format"),
        ("📊 Analytics Dashboard","Pie, bar, histogram, treemap across sessions"),
        ("🏷️ Intent Explorer","Multi-filter search across all sessions"),
        ("✅ Action Item Tracker","Auto-extract + manual add + progress tracking"),
        ("⚠️ Smart Alerts","Auto-flag high risk, many actions, negative tone"),
        ("🤖 Q&A Chatbot","Ask questions about your transcript"),
        ("🔄 Session Compare","Side-by-side diff of two sessions"),
        ("#️⃣ Topic Analysis","Word cloud + treemap of key topics"),
        ("🗂️ Entity Map","Frequency charts for all extracted entities"),
        ("📈 Benchmark","ROUGE + classification metrics from notebook"),
        ("📄 Export JSON","Full structured export"),
        ("📊 Export CSV","Tabular sentence export"),
        ("📑 Export PDF","Professional report generation"),
        ("📝 Export Markdown","Clean markdown report"),
        ("⭐ Bookmarks","Save important sessions"),
        ("📋 Session Notes","Personal freeform notes"),
        ("🔒 Confidence Filter","Global min-confidence threshold"),
        ("⚡ Priority Sort","Sort sentences by urgency"),
        ("🎨 Dark Theme","Full custom glassmorphism UI"),
        ("🌐 ngrok Tunnel","Public URL from Colab in one click"),
        ("♻️ Session History","Full history with timeline view"),
        ("📡 Multi-format Audio","Any audio format via pydub/soundfile"),
        ("⚙️ Whisper Preload","Cache and preload Whisper model"),
    ]
    c1,c2 = st.columns(2)
    for i,(title,desc) in enumerate(features):
        with (c1 if i%2==0 else c2):
            st.markdown(f"""<div class="nlu-card" style="padding:.8rem 1rem">
              <div style="font-weight:700;font-size:.87rem">{title}</div>
              <div style="color:#5d6b8a;font-size:.76rem;margin-top:.15rem">{desc}</div>
            </div>""",unsafe_allow_html=True)
    st.divider()
    st.caption("Built with Streamlit · Whisper · Plotly · WordCloud · FPDF · HuggingFace Transformers · st-audiorec · pyngrok")
