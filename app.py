import streamlit as st
import time
import json
import os
import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLU Intelligence Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DARK THEME CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:    #0a0b0e;
    --bg-secondary:  #111318;
    --bg-card:       #161922;
    --bg-hover:      #1e2230;
    --border:        #252a38;
    --accent:        #4f8cff;
    --accent-dim:    #2a4a8a;
    --accent-glow:   rgba(79,140,255,0.15);
    --text-primary:  #e8eaf0;
    --text-secondary:#8b93a8;
    --text-muted:    #505870;
    --success:       #34d399;
    --warning:       #fbbf24;
    --danger:        #f87171;
    --mono:          'JetBrains Mono', monospace;
    --sans:          'Syne', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp { background-color: var(--bg-primary); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label { color: var(--text-primary) !important; }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* Buttons */
.stButton > button {
    background-color: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #3a74e8 !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background-color: var(--accent-glow) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 0 !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background-color: var(--accent-glow) !important;
}

/* Dataframe */
.stDataFrame { border-radius: 8px; overflow: hidden; }
.stDataFrame iframe { background-color: var(--bg-card) !important; }

/* Metrics */
[data-testid="metric-container"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}
[data-testid="metric-container"] label { color: var(--text-secondary) !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; }

/* Selectbox */
.stSelectbox > div > div {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { background-color: var(--border) !important; }

/* Expander */
details { background-color: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
summary { color: var(--text-primary) !important; font-weight: 600 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* Info / Success / Warning boxes */
.stAlert { border-radius: 8px !important; border: none !important; }
[data-baseweb="notification"] { background-color: var(--bg-card) !important; }

/* Custom card */
.nlp-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.nlp-card-accent { border-left: 3px solid var(--accent); }

/* Header */
.page-header {
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.page-subheader {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Entity highlight */
.entity-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-family: var(--mono);
    font-weight: 600;
    margin: 2px;
}
.entity-PER { background: rgba(79,140,255,0.2); color: #4f8cff; border: 1px solid rgba(79,140,255,0.4); }
.entity-ORG { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.entity-LOC { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
.entity-MISC { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

/* Progress bar override */
.stProgress > div > div > div { background-color: var(--accent) !important; }

/* Timing badge */
.timing-badge {
    display: inline-block;
    background: var(--bg-hover);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
}
.confidence-bar-container { width: 100%; background: var(--border); border-radius: 4px; height: 6px; margin: 4px 0; }
.confidence-bar { height: 6px; border-radius: 4px; background: var(--accent); }

/* Scrollable output */
.output-block {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-secondary);
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Sidebar nav */
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 2px 0;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
.nav-item.active { background: var(--accent-glow); color: var(--accent); border-left: 2px solid var(--accent); }

/* Device badge */
.device-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-family: var(--mono);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.device-gpu { background: rgba(52,211,153,0.15); color: var(--success); border: 1px solid rgba(52,211,153,0.3); }
.device-cpu { background: rgba(251,191,36,0.15); color: var(--warning); border: 1px solid rgba(251,191,36,0.3); }
</style>
""", unsafe_allow_html=True)

# ─── DEVICE DETECTION ─────────────────────────────────────────────────────────
@st.cache_resource
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

# ─── SESSION STATE DEFAULTS ───────────────────────────────────────────────────
defaults = {
    "transcript": "",
    "ner_results": [],
    "summary": "",
    "intents": [],
    "actions": [],
    "embeddings": [],
    "embed_labels": [],
    "retrieval_results": [],
    "similarity_score": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── LAZY MODEL LOADERS ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper():
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if DEVICE == "cuda" else -1,
        return_timestamps=True,
    )

@st.cache_resource(show_spinner=False)
def load_intent_classifier():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def load_ner():
    from transformers import pipeline
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def load_summarizer():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def load_flan():
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def timing(fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, round(time.time() - t0, 2)

def render_header(title, subtitle):
    st.markdown(f'<div class="page-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subheader">{subtitle}</div>', unsafe_allow_html=True)

def card(content_fn, accent=True):
    cls = "nlp-card nlp-card-accent" if accent else "nlp-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

def json_download(data, filename):
    blob = json.dumps(data, indent=2, ensure_ascii=False)
    st.download_button(
        "Download JSON",
        data=blob,
        file_name=filename,
        mime="application/json",
    )

def device_badge():
    if DEVICE == "cuda":
        st.markdown('<span class="device-badge device-gpu">GPU Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="device-badge device-cpu">CPU Mode</span>', unsafe_allow_html=True)

def confidence_bar(score, label=""):
    pct = int(score * 100)
    st.markdown(
        f"""<div style="margin:4px 0">
        <span style="font-family:var(--mono);font-size:11px;color:var(--text-secondary)">{label}</span>
        <div class="confidence-bar-container">
          <div class="confidence-bar" style="width:{pct}%"></div>
        </div>
        <span style="font-family:var(--mono);font-size:11px;color:var(--text-muted)">{pct}%</span>
        </div>""",
        unsafe_allow_html=True,
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
PAGES = {
    "Whisper STT": ["Audio Transcription", "Language Detection", "Transcript Export"],
    "Intent Classification": ["Intent Prediction", "Multi-Intent Ranking", "Custom Labels"],
    "Named Entity Recognition": ["Entity Extraction", "Entity Highlighting", "Entity Export"],
    "Summarization": ["Long Text Summary", "Meeting Summary", "Summary Analytics"],
    "Action Extraction": ["Action Items", "Instruction to Task", "Workflow Generation"],
    "Semantic Embeddings": ["Semantic Similarity", "Semantic Retrieval", "Embedding Visualization"],
}

with st.sidebar:
    st.markdown("""
    <div style="padding:1.25rem 0.5rem 0.75rem">
      <div style="font-family:var(--sans);font-size:1.2rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.02em">
        NLU Suite
      </div>
      <div style="font-family:var(--mono);font-size:10px;color:var(--text-muted);letter-spacing:0.1em;text-transform:uppercase;margin-top:2px">
        Intelligence Platform
      </div>
    </div>
    <hr style="margin:0.5rem 0 1rem 0">
    """, unsafe_allow_html=True)

    device_badge()
    st.markdown("<br>", unsafe_allow_html=True)

    selected_section = st.radio(
        "Module",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:var(--mono);font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em">Models</div>',
        unsafe_allow_html=True,
    )
    model_info = [
        ("Whisper", "openai/whisper-small"),
        ("Intent", "bart-large-mnli"),
        ("NER", "bert-base-NER"),
        ("Summary", "bart-large-cnn"),
        ("Action", "flan-t5-base"),
        ("Embed", "all-MiniLM-L6-v2"),
    ]
    for short, full in model_info:
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:11px;color:var(--text-secondary);padding:2px 0">'
            f'<span style="color:var(--accent)">{short}</span> {full}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Session started {datetime.now().strftime('%H:%M')}")

# ─── WHISPER STT ──────────────────────────────────────────────────────────────
if selected_section == "Whisper STT":
    render_header("Speech Recognition", "Whisper STT — openai/whisper-small")

    tabs = st.tabs(["Audio Transcription", "Language Detection", "Transcript Export"])

    # TAB 1: Audio Transcription
    with tabs[0]:
        st.markdown("#### Upload Audio File")
        audio_file = st.file_uploader("Supported: mp3, wav, m4a, ogg, flac", type=["mp3", "wav", "m4a", "ogg", "flac"], key="asr_upload")
        col1, col2 = st.columns([3, 1])
        with col1:
            task_type = st.selectbox("Task", ["transcribe", "translate"], key="asr_task")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_asr = st.button("Transcribe", key="run_asr")

        if audio_file and run_asr:
            with st.spinner("Loading Whisper model..."):
                asr = load_whisper()
            audio_bytes = audio_file.read()
            tmp_path = f"/tmp/upload_{int(time.time())}.wav"
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            with st.spinner("Transcribing audio..."):
                result, elapsed = timing(asr, tmp_path, generate_kwargs={"task": task_type})
            transcript = result["text"]
            st.session_state["transcript"] = transcript
            st.session_state["chunks"] = result.get("chunks", [])
            st.success(f"Transcription complete in {elapsed}s")
            st.audio(audio_bytes)
            st.markdown("**Transcript**")
            st.markdown(f'<div class="output-block">{transcript}</div>', unsafe_allow_html=True)

        elif st.session_state["transcript"]:
            st.info("Showing previous transcript")
            st.markdown(f'<div class="output-block">{st.session_state["transcript"]}</div>', unsafe_allow_html=True)
        else:
            st.info("Upload an audio file and click Transcribe to begin.")

    # TAB 2: Language Detection
    with tabs[1]:
        render_header("Language Detection", "Auto-detect spoken language from audio")
        audio_file2 = st.file_uploader("Upload audio for language detection", type=["mp3", "wav", "m4a", "ogg", "flac"], key="lang_upload")
        if audio_file2 and st.button("Detect Language"):
            with st.spinner("Loading Whisper..."):
                asr = load_whisper()
            audio_bytes2 = audio_file2.read()
            tmp2 = f"/tmp/lang_{int(time.time())}.wav"
            with open(tmp2, "wb") as f:
                f.write(audio_bytes2)
            with st.spinner("Detecting language..."):
                import torch
                import librosa
                audio_np, sr = librosa.load(tmp2, sr=16000, mono=True, duration=30)
                processor = asr.feature_extractor
                inputs = processor(audio_np, return_tensors="pt", sampling_rate=16000)
                model = asr.model
                with torch.no_grad():
                    predicted_ids = model.generate(
                        inputs["input_features"].to(model.device),
                        task="transcribe",
                        return_timestamps=False,
                    )
                lang_token = asr.tokenizer.decode(predicted_ids[0][:3])

            cols = st.columns(3)
            cols[0].metric("Detected Language Token", lang_token[:20])
            cols[1].metric("Audio Duration", f"{round(len(audio_np)/sr, 1)}s")
            cols[2].metric("Sample Rate", f"{sr} Hz")
            st.markdown("**Note:** Whisper detects language automatically; use `translate` task to convert non-English to English.")

        elif st.session_state["transcript"]:
            st.info("Language detection uses a freshly uploaded audio file.")
        else:
            st.info("Upload audio to detect the spoken language.")

    # TAB 3: Transcript Export
    with tabs[2]:
        render_header("Transcript Export", "Download and format transcripts")
        if st.session_state["transcript"]:
            txt = st.session_state["transcript"]
            chunks = st.session_state.get("chunks", [])

            st.markdown("**Formatted Transcript**")
            st.text_area("Full text", txt, height=200, key="export_txt")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download TXT", txt, file_name="transcript.txt", mime="text/plain")
            with col2:
                export_data = {"transcript": txt, "chunks": chunks, "timestamp": datetime.now().isoformat()}
                json_download(export_data, "transcript.json")

            if chunks:
                st.markdown("**Timestamped Segments**")
                rows = [{"start": c.get("timestamp", [None, None])[0], "end": c.get("timestamp", [None, None])[1], "text": c.get("text", "")} for c in chunks]
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Transcribe audio in the first tab to generate an exportable transcript.")

# ─── INTENT CLASSIFICATION ────────────────────────────────────────────────────
elif selected_section == "Intent Classification":
    render_header("Intent Classification", "Zero-shot classification — facebook/bart-large-mnli")

    DEFAULT_LABELS = ["schedule meeting", "send email", "create task", "request update", "approve document", "cancel appointment", "provide feedback", "escalate issue"]
    tabs = st.tabs(["Intent Prediction", "Multi-Intent Ranking", "Custom Labels"])

    def get_intent_text():
        return st.text_area("Input text", value=st.session_state.get("intent_text", ""), placeholder="Enter a sentence or paragraph...", key="intent_input", height=100)

    with tabs[0]:
        text = get_intent_text()
        st.session_state["intent_text"] = text
        if st.button("Predict Intent", key="predict_single"):
            if text.strip():
                with st.spinner("Loading classifier..."):
                    clf = load_intent_classifier()
                with st.spinner("Classifying..."):
                    res, elapsed = timing(clf, text, candidate_labels=DEFAULT_LABELS[:4], multi_label=False)
                st.session_state["intents"] = list(zip(res["labels"], res["scores"]))
                st.markdown(f'<span class="timing-badge">{elapsed}s inference</span>', unsafe_allow_html=True)
                top_label, top_score = res["labels"][0], res["scores"][0]
                st.metric("Top Intent", top_label, f"{round(top_score*100, 1)}% confidence")
                st.markdown("**Confidence Breakdown**")
                for label, score in zip(res["labels"], res["scores"]):
                    confidence_bar(score, label)
            else:
                st.warning("Enter text first.")

    with tabs[1]:
        text = st.text_area("Input text for multi-intent ranking", value=st.session_state.get("intent_text", ""), height=100, key="multi_intent_input")
        top_k = st.slider("Top K intents to show", 2, 8, 5)
        if st.button("Rank Intents", key="rank_intents"):
            if text.strip():
                with st.spinner("Loading classifier..."):
                    clf = load_intent_classifier()
                with st.spinner("Running multi-label classification..."):
                    res, elapsed = timing(clf, text, candidate_labels=DEFAULT_LABELS, multi_label=True)
                pairs = list(zip(res["labels"], res["scores"]))[:top_k]
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                df = pd.DataFrame(pairs, columns=["Intent", "Score"])
                df["Score (%)"] = (df["Score"] * 100).round(1)
                fig = px.bar(df, x="Score (%)", y="Intent", orientation="h",
                             color="Score (%)", color_continuous_scale=["#252a38", "#4f8cff"],
                             template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_family="JetBrains Mono", showlegend=False,
                                  coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df[["Intent", "Score (%)"]], use_container_width=True)
                json_download(pairs, "intent_ranking.json")
            else:
                st.warning("Enter text first.")

    with tabs[2]:
        text = st.text_area("Input text", value=st.session_state.get("intent_text", ""), height=80, key="custom_label_input")
        custom_raw = st.text_input("Custom labels (comma-separated)", value="approve budget, reject proposal, request clarification, defer decision")
        if st.button("Classify with Custom Labels"):
            labels = [l.strip() for l in custom_raw.split(",") if l.strip()]
            if text.strip() and labels:
                with st.spinner("Loading classifier..."):
                    clf = load_intent_classifier()
                with st.spinner("Classifying..."):
                    res, elapsed = timing(clf, text, candidate_labels=labels, multi_label=True)
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                df = pd.DataFrame({"Label": res["labels"], "Score": [round(s, 4) for s in res["scores"]]})
                st.dataframe(df, use_container_width=True)
                fig = px.pie(df, names="Label", values="Score", template="plotly_dark",
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_family="JetBrains Mono")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Provide text and at least one label.")

# ─── NAMED ENTITY RECOGNITION ─────────────────────────────────────────────────
elif selected_section == "Named Entity Recognition":
    render_header("Named Entity Recognition", "Token classification — dslim/bert-base-NER")

    tabs = st.tabs(["Entity Extraction", "Entity Highlighting", "Entity Export"])

    sample_text = "Dr. Sarah Chen from OpenAI met with Elon Musk at Google headquarters in San Francisco to discuss the future of artificial intelligence on Monday."

    with tabs[0]:
        text = st.text_area("Input text", value=sample_text, height=120, key="ner_input")
        if st.button("Extract Entities"):
            if text.strip():
                with st.spinner("Loading NER model..."):
                    ner = load_ner()
                with st.spinner("Extracting entities..."):
                    results, elapsed = timing(ner, text)
                st.session_state["ner_results"] = results
                st.session_state["ner_text"] = text
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                if results:
                    df = pd.DataFrame([{
                        "Entity": r["word"],
                        "Type": r["entity_group"],
                        "Score": round(r["score"], 4),
                        "Start": r["start"],
                        "End": r["end"],
                    } for r in results])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Entities", len(results))
                    col2.metric("Unique Types", df["Type"].nunique())
                    col3.metric("Avg Confidence", f"{df['Score'].mean():.1%}")
                    st.dataframe(df, use_container_width=True)
                    type_counts = df["Type"].value_counts().reset_index()
                    type_counts.columns = ["Type", "Count"]
                    fig = px.bar(type_counts, x="Type", y="Count", template="plotly_dark",
                                 color="Count", color_continuous_scale=["#252a38", "#4f8cff"])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_family="JetBrains Mono", showlegend=False,
                                      coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No entities found.")

    with tabs[1]:
        text2 = st.text_area("Input text for highlighting", value=sample_text, height=120, key="ner_highlight_input")
        if st.button("Highlight Entities"):
            if text2.strip():
                with st.spinner("Loading NER..."):
                    ner = load_ner()
                with st.spinner("Processing..."):
                    results2, _ = timing(ner, text2)

                if results2:
                    highlighted = text2
                    for ent in sorted(results2, key=lambda x: x["start"], reverse=True):
                        word = ent["word"]
                        etype = ent["entity_group"]
                        tag = f'<span class="entity-tag entity-{etype}">{word} <sup>{etype}</sup></span>'
                        start, end = ent["start"], ent["end"]
                        highlighted = highlighted[:start] + tag + highlighted[end:]

                    st.markdown("**Highlighted Output**")
                    st.markdown(
                        f'<div class="nlp-card" style="line-height:2.2;font-size:14px">{highlighted}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("""
                    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem">
                      <span class="entity-tag entity-PER">PER Person</span>
                      <span class="entity-tag entity-ORG">ORG Organization</span>
                      <span class="entity-tag entity-LOC">LOC Location</span>
                      <span class="entity-tag entity-MISC">MISC Miscellaneous</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.info("No entities detected.")

    with tabs[2]:
        if st.session_state["ner_results"]:
            results = st.session_state["ner_results"]
            df = pd.DataFrame([{
                "entity": r["word"],
                "type": r["entity_group"],
                "confidence": round(r["score"], 4),
                "start": r["start"],
                "end": r["end"],
            } for r in results])
            st.dataframe(df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "entities.csv", "text/csv")
            with col2:
                json_download(results, "entities.json")
        else:
            st.info("Run Entity Extraction first to generate exportable data.")

# ─── SUMMARIZATION ────────────────────────────────────────────────────────────
elif selected_section == "Summarization":
    render_header("Summarization", "Abstractive summarization — facebook/bart-large-cnn")

    SAMPLE_MEETING = """The quarterly review meeting commenced at 9 AM with all department heads present. The CEO opened by acknowledging strong Q3 results with revenue up 23% year-over-year. The marketing director reported that the new campaign generated 45,000 leads with a conversion rate of 8.2%. The engineering team completed migration to the new cloud infrastructure two weeks ahead of schedule, reducing operational costs by 31%. However, the sales team flagged concerns about supply chain delays affecting product delivery timelines. HR presented the new remote work policy effective January 1st. The board unanimously approved the $12M budget for R&D expansion in AI capabilities. Action items: CFO to finalize Q4 projections by Friday, Sales VP to negotiate with logistics partners, and HR to distribute updated policy documents by end of week."""

    tabs = st.tabs(["Long Text Summary", "Meeting Summary", "Summary Analytics"])

    with tabs[0]:
        text = st.text_area("Paste long text to summarize", value=SAMPLE_MEETING, height=180, key="summary_input")
        col1, col2 = st.columns(2)
        with col1:
            min_len = st.slider("Min summary length", 30, 150, 50)
        with col2:
            max_len = st.slider("Max summary length", 100, 400, 200)

        if st.button("Summarize Text"):
            if len(text.split()) < 20:
                st.warning("Text is too short to summarize meaningfully.")
            else:
                with st.spinner("Loading summarizer (BART-large-cnn)..."):
                    summ = load_summarizer()
                with st.spinner("Generating summary..."):
                    res, elapsed = timing(summ, text, max_length=max_len, min_length=min_len, do_sample=False)
                summary_text = res[0]["summary_text"]
                st.session_state["summary"] = summary_text
                st.session_state["original_text"] = text
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                st.markdown("**Summary**")
                st.markdown(f'<div class="nlp-card nlp-card-accent" style="font-size:14px;line-height:1.8">{summary_text}</div>', unsafe_allow_html=True)
                ratio = round(len(summary_text.split()) / len(text.split()) * 100, 1)
                c1, c2, c3 = st.columns(3)
                c1.metric("Original Words", len(text.split()))
                c2.metric("Summary Words", len(summary_text.split()))
                c3.metric("Compression Ratio", f"{ratio}%")

    with tabs[1]:
        meeting_text = st.text_area("Meeting transcript or notes", value=SAMPLE_MEETING, height=180, key="meeting_sum_input")
        if st.button("Summarize Meeting"):
            if len(meeting_text.split()) >= 20:
                with st.spinner("Loading summarizer..."):
                    summ = load_summarizer()
                prompt = meeting_text
                with st.spinner("Summarizing..."):
                    res, elapsed = timing(summ, prompt, max_length=250, min_length=60, do_sample=False)
                summary_text = res[0]["summary_text"]
                st.session_state["summary"] = summary_text
                st.markdown(f'<span class="timing-badge">{elapsed}s inference</span>', unsafe_allow_html=True)
                st.markdown("**Meeting Summary**")
                st.markdown(f'<div class="nlp-card nlp-card-accent" style="font-size:14px;line-height:1.8">{summary_text}</div>', unsafe_allow_html=True)
                st.download_button("Export Summary", summary_text, "meeting_summary.txt", "text/plain")
            else:
                st.warning("Enter a longer meeting transcript.")

    with tabs[2]:
        if st.session_state["summary"]:
            orig = st.session_state.get("original_text", "")
            summ = st.session_state["summary"]
            st.markdown("**Summary Analytics**")
            orig_words = len(orig.split()) if orig else 0
            summ_words = len(summ.split())
            orig_sents = orig.count(".") + orig.count("!") + orig.count("?") if orig else 0
            summ_sents = summ.count(".") + summ.count("!") + summ.count("?")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Words", orig_words)
            c2.metric("Summary Words", summ_words)
            c3.metric("Original Sentences", orig_sents)
            c4.metric("Summary Sentences", summ_sents)

            if orig_words > 0:
                fig = go.Figure(go.Bar(
                    x=["Original", "Summary"],
                    y=[orig_words, summ_words],
                    marker_color=["#252a38", "#4f8cff"],
                    text=[orig_words, summ_words],
                    textposition="auto",
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="JetBrains Mono",
                    title="Word Count Comparison",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            json_download({"original": orig, "summary": summ, "stats": {"original_words": orig_words, "summary_words": summ_words}}, "summary_analytics.json")
        else:
            st.info("Run a summarization in the previous tabs first.")

# ─── ACTION EXTRACTION ────────────────────────────────────────────────────────
elif selected_section == "Action Extraction":
    render_header("Action Extraction", "Seq2Seq generation — google/flan-t5-base")

    SAMPLE = "The team agreed to review the Q4 budget by Thursday. Alice will send the updated report to Bob. Everyone needs to complete the training module before the Friday deadline. John should schedule a follow-up meeting with the client next week."

    tabs = st.tabs(["Action Items", "Instruction to Task", "Workflow Generation"])

    with tabs[0]:
        text = st.text_area("Meeting notes or text", value=SAMPLE, height=150, key="action_input")
        if st.button("Extract Action Items"):
            if text.strip():
                with st.spinner("Loading Flan-T5..."):
                    flan = load_flan()
                prompt = f"Extract all action items, tasks, and responsibilities from the following text as a numbered list:\n\n{text}"
                with st.spinner("Extracting actions..."):
                    res, elapsed = timing(flan, prompt, max_new_tokens=200, do_sample=False)
                output = res[0]["generated_text"]
                st.session_state["actions"] = output
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                st.markdown("**Extracted Action Items**")
                st.markdown(f'<div class="output-block">{output}</div>', unsafe_allow_html=True)
                st.download_button("Export Actions", output, "action_items.txt", "text/plain")

    with tabs[1]:
        instruction = st.text_area("Enter an instruction or goal", value="Improve customer onboarding process to reduce churn", height=100, key="inst_input")
        if st.button("Convert to Tasks"):
            if instruction.strip():
                with st.spinner("Loading Flan-T5..."):
                    flan = load_flan()
                prompt = f"Break down this goal into specific actionable tasks:\n\nGoal: {instruction}\n\nTasks:"
                with st.spinner("Generating tasks..."):
                    res, elapsed = timing(flan, prompt, max_new_tokens=200, do_sample=False)
                output = res[0]["generated_text"]
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="output-block">{output}</div>', unsafe_allow_html=True)

    with tabs[2]:
        workflow_goal = st.text_area("Describe a process or goal", value="Launch a new software product to market", height=100, key="workflow_input")
        steps_n = st.slider("Number of workflow steps", 3, 8, 5)
        if st.button("Generate Workflow"):
            if workflow_goal.strip():
                with st.spinner("Loading Flan-T5..."):
                    flan = load_flan()
                prompt = f"Create a step-by-step workflow with {steps_n} steps for: {workflow_goal}"
                with st.spinner("Generating workflow..."):
                    res, elapsed = timing(flan, prompt, max_new_tokens=250, do_sample=False)
                output = res[0]["generated_text"]
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)
                st.markdown("**Generated Workflow**")
                st.markdown(f'<div class="output-block">{output}</div>', unsafe_allow_html=True)

                # Visualize as a simple flow
                steps_text = [s.strip() for s in output.split("\n") if s.strip()][:steps_n]
                if steps_text:
                    fig = go.Figure()
                    for i, step in enumerate(steps_text):
                        fig.add_annotation(
                            x=0.5, y=1 - i / max(len(steps_text), 1),
                            text=f"Step {i+1}: {step[:60]}",
                            showarrow=False,
                            font=dict(size=11, color="#e8eaf0", family="JetBrains Mono"),
                            xanchor="center", align="center",
                            bgcolor="#161922", bordercolor="#4f8cff", borderwidth=1,
                        )
                    fig.update_layout(
                        height=max(200, len(steps_text) * 60),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                json_download({"goal": workflow_goal, "workflow": output}, "workflow.json")

# ─── SEMANTIC EMBEDDINGS ──────────────────────────────────────────────────────
elif selected_section == "Semantic Embeddings":
    render_header("Semantic Embeddings", "Dense retrieval — sentence-transformers/all-MiniLM-L6-v2")

    tabs = st.tabs(["Semantic Similarity", "Semantic Retrieval", "Embedding Visualization"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            s1 = st.text_area("Sentence A", value="The project deadline has been moved to next Friday.", height=100, key="sim_s1")
        with col2:
            s2 = st.text_area("Sentence B", value="We pushed the project due date to the end of next week.", height=100, key="sim_s2")

        if st.button("Compute Similarity"):
            if s1.strip() and s2.strip():
                with st.spinner("Loading embedder..."):
                    embedder = load_embedder()
                with st.spinner("Computing embeddings..."):
                    embs, elapsed = timing(embedder.encode, [s1, s2])
                    from sklearn.metrics.pairwise import cosine_similarity
                    score = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
                st.session_state["similarity_score"] = score
                st.markdown(f'<span class="timing-badge">{elapsed}s</span>', unsafe_allow_html=True)

                color = "#34d399" if score > 0.7 else "#fbbf24" if score > 0.4 else "#f87171"
                st.markdown(
                    f'<div class="nlp-card" style="text-align:center">'
                    f'<div style="font-size:3rem;font-weight:800;color:{color};font-family:JetBrains Mono">{score:.4f}</div>'
                    f'<div style="color:var(--text-secondary);font-size:13px;margin-top:0.5rem">Cosine Similarity Score</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                interp = "Very similar" if score > 0.7 else "Moderately similar" if score > 0.4 else "Semantically different"
                st.markdown(f"**Interpretation:** {interp} (threshold: >0.7 = high, 0.4-0.7 = medium, <0.4 = low)")

    with tabs[1]:
        query = st.text_input("Search query", value="machine learning optimization", key="retrieval_query")
        corpus_raw = st.text_area(
            "Corpus (one sentence per line)",
            value="\n".join([
                "Deep learning requires large amounts of labeled training data.",
                "The project was approved by the executive committee last Tuesday.",
                "Neural network training can be accelerated using GPU hardware.",
                "Our quarterly revenue exceeded expectations for three consecutive quarters.",
                "Gradient descent is a core optimization algorithm in machine learning.",
                "The team celebrated their successful product launch in Paris.",
                "Transformers use self-attention mechanisms to process sequences.",
                "The new office policy requires all employees to badge in daily.",
            ]),
            height=180,
            key="corpus_input",
        )
        top_k = st.slider("Top K results", 1, 5, 3, key="retrieval_k")

        if st.button("Search") and query.strip():
            corpus = [line.strip() for line in corpus_raw.split("\n") if line.strip()]
            with st.spinner("Loading embedder..."):
                embedder = load_embedder()
            with st.spinner("Computing embeddings..."):
                q_emb = embedder.encode([query])
                c_embs = embedder.encode(corpus)
                from sklearn.metrics.pairwise import cosine_similarity
                scores = cosine_similarity(q_emb, c_embs)[0]
            ranked = sorted(enumerate(corpus), key=lambda x: scores[x[0]], reverse=True)[:top_k]
            st.session_state["retrieval_results"] = [(corpus[i], float(scores[i])) for i, _ in ranked]
            st.markdown("**Top Matches**")
            for rank, (i, sent) in enumerate(ranked):
                score_val = scores[i]
                st.markdown(
                    f'<div class="nlp-card" style="margin-bottom:0.5rem">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                    f'<span style="font-family:var(--mono);font-size:11px;color:var(--accent)">Rank {rank+1}</span>'
                    f'<span style="font-family:var(--mono);font-size:11px;color:var(--text-muted)">{score_val:.4f}</span>'
                    f'</div>'
                    f'<div style="font-size:13px;color:var(--text-primary)">{sent}</div>'
                    f'<div class="confidence-bar-container"><div class="confidence-bar" style="width:{int(score_val*100)}%"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tabs[2]:
        viz_raw = st.text_area(
            "Sentences to visualize (one per line)",
            value="\n".join([
                "Machine learning optimizes model parameters",
                "Deep learning uses neural networks",
                "The stock market crashed yesterday",
                "Financial markets are volatile",
                "Python is great for data science",
                "JavaScript dominates web development",
                "The board approved the Q4 budget",
                "Revenue grew 23% year-over-year",
            ]),
            height=180,
            key="viz_input",
        )
        if st.button("Visualize Embeddings"):
            sentences = [s.strip() for s in viz_raw.split("\n") if s.strip()]
            if len(sentences) >= 3:
                with st.spinner("Loading embedder..."):
                    embedder = load_embedder()
                with st.spinner("Computing and projecting embeddings..."):
                    embs = embedder.encode(sentences)
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(embs)
                df_viz = pd.DataFrame({
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "label": sentences,
                    "short": [s[:30] + "..." if len(s) > 30 else s for s in sentences],
                })
                fig = px.scatter(
                    df_viz, x="x", y="y", text="short",
                    template="plotly_dark",
                    title="Embedding Space (PCA 2D projection)",
                )
                fig.update_traces(
                    marker=dict(size=12, color="#4f8cff", line=dict(color="#252a38", width=1)),
                    textfont=dict(size=9, color="#8b93a8", family="JetBrains Mono"),
                    textposition="top center",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="JetBrains Mono",
                    showlegend=False,
                    xaxis=dict(gridcolor="#252a38", zerolinecolor="#252a38"),
                    yaxis=dict(gridcolor="#252a38", zerolinecolor="#252a38"),
                )
                st.plotly_chart(fig, use_container_width=True)
                var_explained = pca.explained_variance_ratio_
                c1, c2 = st.columns(2)
                c1.metric("PC1 Variance", f"{var_explained[0]:.1%}")
                c2.metric("PC2 Variance", f"{var_explained[1]:.1%}")
                json_download(df_viz.to_dict(orient="records"), "embeddings_2d.json")
            else:
                st.warning("Enter at least 3 sentences.")
