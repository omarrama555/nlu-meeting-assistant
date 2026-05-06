"""
NLU Meeting Voice-to-Action — Streamlit App
6 AI Models Pipeline: STT → Intent → NER → Summarization → Action Extraction → Embeddings
"""

import streamlit as st
import torch
import numpy as np
import re
import json
import time
import io
import os
import tempfile
from collections import Counter
from pathlib import Path

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLU Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:    #0a0f1e;
    --bg-card:       #0f1629;
    --bg-card2:      #131c35;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-violet: #8b5cf6;
    --accent-amber:  #f59e0b;
    --accent-green:  #10b981;
    --accent-rose:   #f43f5e;
    --text-primary:  #e2e8f0;
    --text-muted:    #64748b;
    --border:        #1e2d4a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── HERO HEADER ── */
.hero-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b3e 50%, #0a0f1e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}
.model-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.2rem;
}
.model-pill {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    color: #93c5fd;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 1.5rem;
}

/* ── METRIC CARDS ── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(59,130,246,0.4); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.metric-card.violet::before { background: linear-gradient(90deg, #8b5cf6, #ec4899); }
.metric-card.green::before  { background: linear-gradient(90deg, #10b981, #06b6d4); }
.metric-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #f97316); }
.metric-card.rose::before   { background: linear-gradient(90deg, #f43f5e, #f97316); }
.metric-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, #8b5cf6); }
.metric-label { color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.4rem; font-family: 'DM Mono', monospace; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: var(--text-primary); line-height: 1; }
.metric-icon  { position: absolute; top: 1.2rem; right: 1.2rem; font-size: 1.5rem; opacity: 0.4; }

/* ── SECTION CARDS ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title.blue   { color: #60a5fa; }
.section-title.violet { color: #a78bfa; }
.section-title.green  { color: #34d399; }
.section-title.amber  { color: #fbbf24; }
.section-title.cyan   { color: #22d3ee; }
.section-title.rose   { color: #fb7185; }

/* ── SUMMARY BOX ── */
.summary-box {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.05));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #cbd5e1;
    font-style: italic;
}

/* ── ENTITY TAGS ── */
.entity-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.entity-tag {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem; border-radius: 8px;
    font-size: 0.8rem; font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.entity-PER  { background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.4); color: #c4b5fd; }
.entity-ORG  { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); color: #93c5fd; }
.entity-LOC  { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7; }
.entity-MISC { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #fcd34d; }
.entity-DATE { background: rgba(244,63,94,0.15);  border: 1px solid rgba(244,63,94,0.4);  color: #fda4af; }
.entity-default { background: rgba(100,116,139,0.15); border: 1px solid rgba(100,116,139,0.3); color: #94a3b8; }

/* ── INTENT BADGE ── */
.intent-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0.9rem; border-radius: 10px;
    background: var(--bg-card2); border: 1px solid var(--border);
    margin-bottom: 0.4rem; font-size: 0.82rem;
}
.intent-label { color: #cbd5e1; font-family: 'DM Mono', monospace; }
.intent-count {
    background: rgba(59,130,246,0.2); color: #93c5fd;
    padding: 0.1rem 0.6rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}

/* ── ACTION ITEMS ── */
.action-item {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.8rem 1rem; border-radius: 10px;
    background: var(--bg-card2); border: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.action-bullet {
    width: 22px; height: 22px; border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: white;
    flex-shrink: 0; margin-top: 1px;
}
.action-text { color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; }

/* ── EMBEDDING VIZ ── */
.embed-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 0.4rem; margin-top: 0.5rem;
}
.embed-cell {
    height: 36px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.5);
    border: 1px solid rgba(255,255,255,0.05);
}

/* ── PIPELINE FLOW ── */
.pipeline-flow {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: 0; margin-bottom: 1.5rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem 1.5rem; overflow: hidden;
}
.pipeline-step {
    display: flex; flex-direction: column; align-items: center;
    padding: 0.5rem 1.2rem; text-align: center; flex: 1;
    min-width: 100px;
}
.pipeline-step-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
.pipeline-step-label { font-size: 0.7rem; color: var(--text-muted); font-family: 'DM Mono', monospace; }
.pipeline-step-name { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin-top: 0.2rem; }
.pipeline-arrow { color: var(--text-muted); font-size: 1.2rem; padding: 0 0.2rem; }

/* ── PROCESSING STATUS ── */
.status-bar {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.status-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.status-dot.green { background: #10b981; box-shadow: 0 0 6px #10b981; }
.status-dot.blue  { background: #3b82f6; animation: pulse 1.5s infinite; }
.status-text { font-size: 0.85rem; color: #94a3b8; font-family: 'DM Mono', monospace; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── WAVEFORM PLACEHOLDER ── */
.waveform-box {
    background: var(--bg-card2); border: 2px dashed var(--border);
    border-radius: 12px; padding: 2rem; text-align: center;
    color: var(--text-muted); font-size: 0.85rem;
}
.waveform-bars {
    display: flex; align-items: flex-end; justify-content: center;
    gap: 3px; height: 40px; margin-bottom: 0.8rem;
}
.waveform-bar {
    width: 4px; border-radius: 2px;
    background: linear-gradient(to top, #3b82f6, #8b5cf6);
    opacity: 0.7;
}

/* ── TRANSCRIPT BOX ── */
.transcript-box {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    font-family: 'DM Sans', sans-serif; font-size: 0.9rem;
    line-height: 1.8; color: #cbd5e1;
    max-height: 220px; overflow-y: auto;
}
.transcript-box::-webkit-scrollbar { width: 4px; }
.transcript-box::-webkit-scrollbar-track { background: transparent; }
.transcript-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── CONFIDENCE BAR ── */
.conf-bar-wrap { margin-bottom: 0.5rem; }
.conf-bar-label { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.25rem; }
.conf-bar-name { color: #cbd5e1; font-family: 'DM Mono', monospace; }
.conf-bar-val  { color: var(--text-muted); }
.conf-bar-track { background: var(--bg-card2); border-radius: 4px; height: 6px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); transition: width 0.8s ease; }

/* ── SIMILARITY MATRIX ── */
.sim-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 1rem; border-radius: 20px;
    font-size: 0.8rem; background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7;
    margin: 0.2rem; font-family: 'DM Mono', monospace;
}
.sim-score { font-weight: 700; }

/* ── DOWNLOAD BTN ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.85 !important; }

/* ── STREAMLIT OVERRIDES ── */
.stTextArea textarea {
    background: var(--bg-card2) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important; transition: transform 0.1s, opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stFileUploader { background: var(--bg-card2) !important; border: 2px dashed var(--border) !important; border-radius: 12px !important; }
.stSelectbox select, .stSlider { color: var(--text-primary) !important; }
div[data-testid="stExpander"] { background: var(--bg-card2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 10px !important; padding: 0.2rem !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { background: rgba(59,130,246,0.2) !important; color: #93c5fd !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all 6 models with lazy imports."""
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForSeq2SeqLM,
        pipeline, WhisperProcessor, WhisperForConditionalGeneration
    )
    from sentence_transformers import SentenceTransformer

    device = 0 if torch.cuda.is_available() else -1
    models = {}

    # 1. Intent Classifier — roberta-base
    with st.spinner("⚡ Loading Intent Classifier (RoBERTa)..."):
        ic_tok   = AutoTokenizer.from_pretrained("roberta-base")
        ic_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        models["intent"] = pipeline(
            "text-classification", model=ic_model, tokenizer=ic_tok, device=device
        )

    # 2. NER — bert-base-cased
    with st.spinner("🏷️ Loading NER Model (BERT)..."):
        ner_tok   = AutoTokenizer.from_pretrained("bert-base-cased")
        ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        models["ner"] = pipeline(
            "ner", model=ner_model, tokenizer=ner_tok,
            aggregation_strategy="simple", device=device
        )

    # 3. Summarizer — facebook/bart-base
    with st.spinner("📋 Loading Summarizer (BART)..."):
        models["summarizer"] = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=device
        )

    # 4. Action Extractor — facebook/bart-base (text2text)
    with st.spinner("✅ Loading Action Extractor (BART)..."):
        models["action"] = pipeline(
            "text2text-generation", model="google/flan-t5-base", device=device,
            max_new_tokens=200
        )

    # 5. Sentence Embedder — all-mpnet-base-v2
    with st.spinner("🔗 Loading Sentence Embedder (MPNet)..."):
        models["embedding"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 6. Whisper STT — openai/whisper-medium → using base for size
    with st.spinner("🎙️ Loading Whisper STT..."):
        stt_proc  = WhisperProcessor.from_pretrained("openai/whisper-base")
        stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        models["stt"] = pipeline(
            "automatic-speech-recognition",
            model=stt_model,
            tokenizer=stt_proc.tokenizer,
            feature_extractor=stt_proc.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            device=device,
        )

    return models


# ─── Pipeline Functions ───────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, stt_pipe) -> str:
    """Convert audio bytes to text using Whisper."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio_array, sr = sf.read(tmp_path)
        result = stt_pipe({"array": audio_array.astype(np.float32), "sampling_rate": sr})
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def run_intent_classification(sentences: list, ic_pipe) -> list:
    """Classify intent for each sentence."""
    if not sentences:
        return []
    batch = sentences[:20]
    results = ic_pipe(batch, truncation=True, max_length=128)
    return [
        {"sentence": s, "intent": r["label"], "confidence": round(r["score"], 4)}
        for s, r in zip(batch, results)
    ]


def run_ner(sentences: list, ner_pipe) -> list:
    """Extract named entities from sentences."""
    entities = []
    for sent in sentences[:15]:
        ents = ner_pipe(sent)
        for e in ents:
            entities.append({
                "entity": e["entity_group"],
                "word": e["word"],
                "score": round(e["score"], 4),
                "sentence": sent[:80],
            })
    return entities


def run_summarization(transcript: str, summ_pipe) -> str:
    """Generate meeting summary."""
    text = transcript[:1024]
    result = summ_pipe(text, max_length=130, min_length=30, do_sample=False)
    return result[0]["summary_text"]


def make_action_prompt(transcript: str) -> str:
    return (
        "Extract action items from this meeting transcript. "
        "List each action item on a new line with the responsible person and deadline if mentioned.\n\n"
        f"Transcript:\n{transcript[:800]}\n\nAction Items:"
    )


def run_action_extraction(transcript: str, act_pipe) -> list:
    """Extract action items from transcript."""
    prompt = make_action_prompt(transcript)
    result = act_pipe(prompt)[0]["generated_text"]
    # Parse into list
    lines = [l.strip().lstrip("-•*123456789. ") for l in result.split("\n") if l.strip()]
    return [l for l in lines if len(l) > 10][:10]


def run_embeddings(sentences: list, embed_model) -> np.ndarray:
    """Generate sentence embeddings."""
    if not sentences:
        return np.array([])
    return embed_model.encode(sentences[:10], normalize_embeddings=True)


def compute_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity."""
    if len(embeddings) < 2:
        return np.array([])
    return (embeddings @ embeddings.T)


def run_full_pipeline(transcript: str, models: dict) -> dict:
    """Run all 6 model pipeline on transcript."""
    results = {"transcript": transcript, "timing": {}}

    # Sentence splitting
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if len(s.strip()) > 5]
    results["sentences"] = sentences

    t0 = time.time()
    results["intents"] = run_intent_classification(sentences, models["intent"])
    results["timing"]["intent"] = round(time.time() - t0, 2)

    t0 = time.time()
    results["entities"] = run_ner(sentences, models["ner"])
    results["timing"]["ner"] = round(time.time() - t0, 2)

    t0 = time.time()
    results["summary"] = run_summarization(transcript, models["summarizer"])
    results["timing"]["summarizer"] = round(time.time() - t0, 2)

    t0 = time.time()
    results["action_items"] = run_action_extraction(transcript, models["action"])
    results["timing"]["action"] = round(time.time() - t0, 2)

    t0 = time.time()
    embeddings = run_embeddings(sentences, models["embedding"])
    results["embeddings"] = embeddings
    results["similarity"] = compute_similarity(embeddings)
    results["timing"]["embedding"] = round(time.time() - t0, 2)

    return results


# ─── UI Helpers ───────────────────────────────────────────────────────────────
def render_metric_cards(results: dict):
    n_sentences = len(results.get("sentences", []))
    n_entities  = len(results.get("entities", []))
    n_intents   = len(set(r["intent"] for r in results.get("intents", [])))
    n_actions   = len(results.get("action_items", []))
    n_words     = len(results["transcript"].split())
    embed_shape = results["embeddings"].shape if len(results.get("embeddings", [])) > 0 else (0, 0)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-icon">💬</div>
            <div class="metric-label">Total Words</div>
            <div class="metric-value">{n_words:,}</div>
        </div>
        <div class="metric-card violet">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Intent Types</div>
            <div class="metric-value">{n_intents}</div>
        </div>
        <div class="metric-card green">
            <div class="metric-icon">🏷️</div>
            <div class="metric-label">Entities Found</div>
            <div class="metric-value">{n_entities}</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-icon">✅</div>
            <div class="metric-label">Action Items</div>
            <div class="metric-value">{n_actions}</div>
        </div>
        <div class="metric-card rose">
            <div class="metric-icon">📝</div>
            <div class="metric-label">Sentences</div>
            <div class="metric-value">{n_sentences}</div>
        </div>
        <div class="metric-card cyan">
            <div class="metric-icon">🔢</div>
            <div class="metric-label">Embed Dims</div>
            <div class="metric-value">{embed_shape[1] if len(embed_shape) > 1 else 0}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_entity_tag(entity: str, word: str, score: float) -> str:
    cls = f"entity-{entity}" if entity in ["PER", "ORG", "LOC", "MISC", "DATE"] else "entity-default"
    return f'<span class="entity-tag {cls}"><b>{entity}</b> {word} <small>({score:.2f})</small></span>'


def render_confidence_bar(label: str, value: float, color: str = "#3b82f6"):
    pct = int(value * 100)
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-label">
            <span class="conf-bar-name">{label}</span>
            <span class="conf-bar-val">{pct}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{pct}%; background: linear-gradient(90deg, {color}, #8b5cf6);"></div>
        </div>
    </div>"""


SAMPLE_TRANSCRIPT = """Alice: Good morning everyone. Let's get started with our weekly product sync.
John, can you give us the engineering update?
John: Sure. We completed the user authentication module this week.
The database migration to PostgreSQL is 90% done, we expect to finish by Thursday.
Sarah: That's great news. On the design side, we finalized the new dashboard mockups.
I'll share them with the team by end of day today.
Alice: Perfect. What about the security audit? Has that been addressed?
John: Mike is leading that. Mike, do you want to update the team?
Mike: Yes, we identified three medium-priority vulnerabilities. I'll send a detailed report to all stakeholders by tomorrow morning.
Alice: Thanks Mike. We need to schedule a follow-up meeting with the security team next week.
Sarah, can you set that up?
Sarah: Absolutely. I'll send out the calendar invite today.
Alice: One more thing — we need to finalize the Q4 budget proposal. John, please coordinate with finance to get the numbers ready before the board meeting on December 5th.
John: Got it. I'll set up a meeting with the CFO this week.
Alice: Great. Let's wrap up. Any other blockers?
Mike: No blockers from my side.
Sarah: Same here.
Alice: Perfect. Thanks everyone. Talk to you next week."""


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # ── HERO HEADER ──
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🎙️ NLU Meeting Intelligence</h1>
        <p class="hero-sub">Voice-to-Action AI Pipeline · 6 Transformer Models · End-to-End NLP</p>
        <div class="model-pills">
            <span class="model-pill">🎙️ Whisper STT</span>
            <span class="model-pill">🎯 RoBERTa Intent</span>
            <span class="model-pill">🏷️ BERT NER</span>
            <span class="model-pill">📋 BART Summarizer</span>
            <span class="model-pill">✅ T5 Action Extractor</span>
            <span class="model-pill">🔗 MPNet Embeddings</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PIPELINE FLOW ──
    st.markdown("""
    <div class="pipeline-flow">
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🎙️</div>
            <div class="pipeline-step-name">Whisper</div>
            <div class="pipeline-step-label">Speech-to-Text</div>
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🎯</div>
            <div class="pipeline-step-name">RoBERTa</div>
            <div class="pipeline-step-label">Intent Class.</div>
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🏷️</div>
            <div class="pipeline-step-name">BERT</div>
            <div class="pipeline-step-label">Named Entity</div>
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">📋</div>
            <div class="pipeline-step-name">BART</div>
            <div class="pipeline-step-label">Summarization</div>
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">✅</div>
            <div class="pipeline-step-name">T5</div>
            <div class="pipeline-step-label">Action Items</div>
        </div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step">
            <div class="pipeline-step-icon">🔗</div>
            <div class="pipeline-step-name">MPNet</div>
            <div class="pipeline-step-label">Embeddings</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### ⚙️ Input Mode")
        input_mode = st.radio(
            "Choose input type",
            ["📝 Text Transcript", "🎵 Audio File"],
            label_visibility="collapsed"
        )

        st.markdown("### 🔧 Pipeline Settings")
        max_sentences = st.slider("Max sentences for intent", 5, 20, 12)
        max_ner_sents = st.slider("Max sentences for NER", 5, 15, 10)
        summary_max_len = st.slider("Summary max length", 50, 200, 130)

        st.markdown("### 📊 Analysis Options")
        show_similarity = st.checkbox("Show similarity matrix", True)
        show_raw_json   = st.checkbox("Show raw JSON output", False)
        show_timing     = st.checkbox("Show processing times", True)

        st.markdown("---")
        st.markdown("""
        <div style="color:#475569; font-size:0.75rem; font-family:'DM Mono',monospace; line-height:1.8;">
        <b style="color:#64748b;">6 Models Pipeline</b><br>
        • openai/whisper-base<br>
        • roberta-base<br>
        • bert-base-cased<br>
        • facebook/bart-large-cnn<br>
        • google/flan-t5-base<br>
        • all-MiniLM-L6-v2
        </div>
        """, unsafe_allow_html=True)

    # ── INPUT SECTION ──
    col_input, col_run = st.columns([4, 1])

    transcript = ""
    audio_bytes = None

    if "📝 Text Transcript" in input_mode:
        with col_input:
            transcript_input = st.text_area(
                "Meeting Transcript",
                value=SAMPLE_TRANSCRIPT,
                height=200,
                label_visibility="collapsed",
                placeholder="Paste your meeting transcript here...",
            )
        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)
            use_sample = st.button("📋 Use Sample")
            run_btn    = st.button("🚀 Analyze Meeting")

        transcript = transcript_input

    else:
        st.markdown("""
        <div class="waveform-box">
            <div class="waveform-bars">
        """ + "".join([
            f'<div class="waveform-bar" style="height:{h}px;"></div>'
            for h in [8,15,25,18,30,22,12,28,20,16,35,25,14,22,18,30,12,25,20,16]
        ]) + """
            </div>
            Upload your meeting audio file (WAV, MP3, M4A)
        </div>
        """, unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Upload Audio", type=["wav", "mp3", "m4a", "ogg"],
            label_visibility="collapsed"
        )
        run_btn = st.button("🎙️ Transcribe & Analyze")
        use_sample = False
        if audio_file:
            audio_bytes = audio_file.read()

    if use_sample:
        transcript = SAMPLE_TRANSCRIPT

    # ── RUN ANALYSIS ──
    if run_btn and (transcript.strip() or audio_bytes):
        # Load models
        with st.spinner("🔄 Initializing AI models..."):
            try:
                models = load_models()
                st.markdown('<div class="status-bar"><div class="status-dot green"></div><div class="status-text">All 6 models loaded successfully</div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Model loading error: {e}")
                st.stop()

        # Transcribe audio if needed
        if audio_bytes:
            with st.spinner("🎙️ Transcribing audio with Whisper..."):
                try:
                    transcript = transcribe_audio(audio_bytes, models["stt"])
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    st.stop()

        if not transcript.strip():
            st.warning("⚠️ No transcript to analyze.")
            st.stop()

        # Run pipeline
        progress_bar = st.progress(0, text="Running pipeline...")
        steps = ["Intent Classification", "Named Entity Recognition",
                 "Summarization", "Action Extraction", "Embeddings"]

        results = {"transcript": transcript, "timing": {}, "sentences": []}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if len(s.strip()) > 5]
        results["sentences"] = sentences

        # Step 1: Intent
        progress_bar.progress(10, text="🎯 Running Intent Classification...")
        t0 = time.time()
        results["intents"] = run_intent_classification(sentences[:max_sentences], models["intent"])
        results["timing"]["intent"] = round(time.time() - t0, 2)
        progress_bar.progress(30, text="🏷️ Running NER...")

        # Step 2: NER
        t0 = time.time()
        results["entities"] = run_ner(sentences[:max_ner_sents], models["ner"])
        results["timing"]["ner"] = round(time.time() - t0, 2)
        progress_bar.progress(50, text="📋 Running Summarization...")

        # Step 3: Summarization
        t0 = time.time()
        results["summary"] = run_summarization(transcript, models["summarizer"])
        results["timing"]["summarizer"] = round(time.time() - t0, 2)
        progress_bar.progress(65, text="✅ Extracting Action Items...")

        # Step 4: Action Extraction
        t0 = time.time()
        results["action_items"] = run_action_extraction(transcript, models["action"])
        results["timing"]["action"] = round(time.time() - t0, 2)
        progress_bar.progress(85, text="🔗 Computing Embeddings...")

        # Step 5: Embeddings
        t0 = time.time()
        embeddings = run_embeddings(sentences, models["embedding"])
        results["embeddings"] = embeddings
        results["similarity"] = compute_similarity(embeddings)
        results["timing"]["embedding"] = round(time.time() - t0, 2)
        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.3)
        progress_bar.empty()

        st.session_state["results"] = results

    # ── DISPLAY RESULTS ──
    if "results" in st.session_state:
        results = st.session_state["results"]

        # Metric cards
        render_metric_cards(results)

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Summary", "🎯 Intents", "🏷️ Entities", "✅ Actions", "🔗 Embeddings", "📄 Transcript"
        ])

        # ── TAB 1: SUMMARY ──
        with tab1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title blue">📋 Meeting Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{results.get("summary", "No summary generated.")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if show_timing:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title cyan">⏱️ Processing Times</div>', unsafe_allow_html=True)
                timing = results.get("timing", {})
                timing_labels = {
                    "intent": "Intent Classification",
                    "ner": "Named Entity Recognition",
                    "summarizer": "Summarization",
                    "action": "Action Extraction",
                    "embedding": "Embeddings",
                }
                total = sum(timing.values())
                for key, label in timing_labels.items():
                    val = timing.get(key, 0)
                    st.markdown(render_confidence_bar(f"{label} ({val}s)", val / max(total, 1)), unsafe_allow_html=True)
                st.markdown(f'<p style="color:#64748b; font-size:0.8rem; font-family:\'DM Mono\',monospace; margin-top:0.5rem;">Total: {total:.2f}s</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── TAB 2: INTENTS ──
        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title violet">🎯 Intent Distribution</div>', unsafe_allow_html=True)
            intents = results.get("intents", [])
            if intents:
                intent_counts = Counter(r["intent"] for r in intents)
                total_sents = len(intents)
                for intent, count in intent_counts.most_common():
                    pct = count / total_sents
                    st.markdown(f"""
                    <div class="intent-item">
                        <span class="intent-label">{intent}</span>
                        <span class="intent-count">{count} / {total_sents}</span>
                    </div>
                    {render_confidence_bar(intent, pct, "#8b5cf6")}
                    """, unsafe_allow_html=True)
            else:
                st.info("No intent results available.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Per-sentence intents
            with st.expander("📄 Per-Sentence Intent Details"):
                for item in intents[:15]:
                    conf = item["confidence"]
                    color = "#10b981" if conf > 0.8 else "#f59e0b" if conf > 0.6 else "#f43f5e"
                    st.markdown(f"""
                    <div style="padding:0.5rem 0.8rem; border-left: 3px solid {color}; margin-bottom:0.4rem; background:var(--bg-card2); border-radius:0 8px 8px 0;">
                        <div style="font-size:0.78rem; color:#64748b; font-family:'DM Mono',monospace; margin-bottom:0.2rem;">
                            {item['intent']} · {int(conf*100)}% confidence
                        </div>
                        <div style="font-size:0.85rem; color:#cbd5e1;">{item['sentence'][:120]}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── TAB 3: ENTITIES ──
        with tab3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title green">🏷️ Named Entities</div>', unsafe_allow_html=True)
            entities = results.get("entities", [])
            if entities:
                # Group by type
                entity_groups = {}
                for e in entities:
                    etype = e["entity"]
                    entity_groups.setdefault(etype, []).append(e)

                for etype, ents in sorted(entity_groups.items()):
                    cls = f"entity-{etype}" if etype in ["PER", "ORG", "LOC", "MISC", "DATE"] else "entity-default"
                    st.markdown(f'<div style="margin-bottom:1rem;"><div style="font-size:0.75rem; color:#64748b; font-family:\'DM Mono\',monospace; margin-bottom:0.4rem; text-transform:uppercase; letter-spacing:1px;">{etype} ({len(ents)})</div><div class="entity-row">', unsafe_allow_html=True)
                    for e in ents[:12]:
                        st.markdown(f'<span class="entity-tag {cls}"><b>{etype}</b> {e["word"]} <small>({e["score"]:.2f})</small></span>', unsafe_allow_html=True)
                    st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.info("No entities detected.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── TAB 4: ACTIONS ──
        with tab4:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title amber">✅ Action Items</div>', unsafe_allow_html=True)
            action_items = results.get("action_items", [])
            if action_items:
                for i, item in enumerate(action_items, 1):
                    st.markdown(f"""
                    <div class="action-item">
                        <div class="action-bullet">{i}</div>
                        <div class="action-text">{item}</div>
                    </div>
                    """, unsafe_allow_html=True)
                # Export actions
                actions_text = "\n".join(f"{i}. {a}" for i, a in enumerate(action_items, 1))
                st.download_button(
                    "⬇️ Export Action Items",
                    data=actions_text,
                    file_name="action_items.txt",
                    mime="text/plain"
                )
            else:
                st.info("No action items extracted.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── TAB 5: EMBEDDINGS ──
        with tab5:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title cyan">🔗 Sentence Embeddings & Similarity</div>', unsafe_allow_html=True)
            embeddings = results.get("embeddings", np.array([]))
            sentences_list = results.get("sentences", [])

            if len(embeddings) > 0:
                emb_shape = embeddings.shape
                st.markdown(f"""
                <div style="display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
                    <div style="background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.3); border-radius:8px; padding:0.5rem 1rem;">
                        <span style="color:#64748b; font-size:0.72rem; font-family:'DM Mono',monospace;">SENTENCES</span>
                        <div style="color:#22d3ee; font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;">{emb_shape[0]}</div>
                    </div>
                    <div style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.3); border-radius:8px; padding:0.5rem 1rem;">
                        <span style="color:#64748b; font-size:0.72rem; font-family:'DM Mono',monospace;">DIMENSIONS</span>
                        <div style="color:#a78bfa; font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;">{emb_shape[1]}</div>
                    </div>
                    <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:0.5rem 1rem;">
                        <span style="color:#64748b; font-size:0.72rem; font-family:'DM Mono',monospace;">TOTAL VALUES</span>
                        <div style="color:#34d399; font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;">{emb_shape[0]*emb_shape[1]:,}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Embedding heatmap visualization (first 8 dims of first 5 sentences)
                st.markdown('<div style="font-size:0.78rem; color:#64748b; font-family:\'DM Mono\',monospace; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:1px;">Embedding Values (first 5 sentences × 20 dims)</div>', unsafe_allow_html=True)
                preview = embeddings[:5, :20]
                vmin, vmax = preview.min(), preview.max()
                grid_html = '<div class="embed-grid">'
                for row in preview:
                    for val in row:
                        norm = (val - vmin) / max(vmax - vmin, 1e-8)
                        r = int(59 + norm * (6 - 59))
                        g = int(130 + norm * (182 - 130))
                        b = int(246 + norm * (212 - 246))
                        grid_html += f'<div class="embed-cell" style="background:rgba({r},{g},{b},0.2);">{val:.2f}</div>'
                grid_html += "</div>"
                st.markdown(grid_html, unsafe_allow_html=True)

                # Similarity
                if show_similarity and len(embeddings) > 1:
                    st.markdown('<br><div class="section-title cyan" style="font-size:0.8rem;">🔁 Top Sentence Pairs by Similarity</div>', unsafe_allow_html=True)
                    sim_matrix = results.get("similarity", np.array([]))
                    if len(sim_matrix) > 1:
                        pairs = []
                        n = min(len(sentences_list), len(sim_matrix))
                        for i in range(n):
                            for j in range(i + 1, n):
                                pairs.append((sim_matrix[i, j], i, j))
                        pairs.sort(reverse=True)
                        for score, i, j in pairs[:5]:
                            s1 = sentences_list[i][:60] + "..." if len(sentences_list[i]) > 60 else sentences_list[i]
                            s2 = sentences_list[j][:60] + "..." if len(sentences_list[j]) > 60 else sentences_list[j]
                            color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.6 else "#64748b"
                            st.markdown(f"""
                            <div style="padding:0.7rem 1rem; background:var(--bg-card2); border:1px solid var(--border); border-radius:10px; margin-bottom:0.5rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                                    <span style="color:#64748b; font-size:0.72rem; font-family:'DM Mono',monospace;">SIMILARITY SCORE</span>
                                    <span style="color:{color}; font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;">{score:.3f}</span>
                                </div>
                                <div style="font-size:0.8rem; color:#94a3b8;">"{s1}"</div>
                                <div style="font-size:0.75rem; color:#475569; margin:0.25rem 0;">↔</div>
                                <div style="font-size:0.8rem; color:#94a3b8;">"{s2}"</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No embeddings computed.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── TAB 6: TRANSCRIPT ──
        with tab6:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title rose">📄 Full Transcript</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="transcript-box">{results["transcript"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "⬇️ Download Transcript",
                    data=results["transcript"],
                    file_name="transcript.txt",
                    mime="text/plain"
                )
            with col_dl2:
                if show_raw_json:
                    # Serialize results (exclude numpy arrays)
                    json_results = {k: v for k, v in results.items() if k not in ["embeddings", "similarity"]}
                    st.download_button(
                        "⬇️ Export Full JSON",
                        data=json.dumps(json_results, indent=2, ensure_ascii=False),
                        file_name="meeting_analysis.json",
                        mime="application/json"
                    )

        # ── RAW JSON ──
        if show_raw_json:
            with st.expander("🔍 Raw JSON Output"):
                json_results = {k: v for k, v in results.items() if k not in ["embeddings", "similarity"]}
                st.json(json_results)

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center; padding:3rem 2rem; color:#334155;">
            <div style="font-size:3.5rem; margin-bottom:1rem;">🎙️</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#475569; margin-bottom:0.5rem;">
                Ready to Analyze
            </div>
            <div style="font-size:0.9rem; color:#334155; font-family:'DM Mono',monospace;">
                Paste a transcript or upload audio · Click Analyze Meeting
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
