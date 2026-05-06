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
import os
import tempfile
from collections import Counter

st.set_page_config(
    page_title="NLU Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg:      #060d1f;
    --card:    #0b1428;
    --card2:   #0f1a33;
    --border:  #172040;
    --blue:    #3b82f6;
    --violet:  #7c3aed;
    --cyan:    #06b6d4;
    --green:   #10b981;
    --amber:   #f59e0b;
    --rose:    #f43f5e;
    --text:    #dde4f0;
    --muted:   #4a5a7a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1300px; }

/* HERO */
.hero {
    background: linear-gradient(120deg, #070e22 0%, #0c1530 60%, #070e22 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:320px; height:320px; border-radius:50%;
    background: radial-gradient(circle, rgba(59,130,246,.1) 0%, transparent 70%);
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
    background: linear-gradient(135deg,#60a5fa,#a78bfa,#22d3ee);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0 0 .3rem;
}
.hero-sub { color:var(--muted); font-size:.85rem; font-family:'DM Mono',monospace; margin:0 0 1rem; }
.pills { display:flex; flex-wrap:wrap; gap:.4rem; }
.pill {
    background:rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.25);
    color:#93c5fd; padding:.2rem .7rem; border-radius:30px;
    font-size:.72rem; font-family:'DM Mono',monospace;
}

/* PIPELINE */
.pipe {
    display:flex; align-items:center; flex-wrap:wrap;
    background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:.8rem 1.2rem; margin-bottom:1.4rem; gap:0;
}
.pipe-step { display:flex; flex-direction:column; align-items:center; flex:1; min-width:80px; padding:.3rem .5rem; text-align:center; }
.pipe-icon { font-size:1.3rem; margin-bottom:.2rem; }
.pipe-name { font-size:.75rem; font-weight:700; color:var(--text); font-family:'Syne',sans-serif; }
.pipe-sub  { font-size:.65rem; color:var(--muted); font-family:'DM Mono',monospace; }
.pipe-arr  { color:var(--muted); font-size:1rem; padding:0 .1rem; }

/* METRIC GRID */
.mgrid { display:grid; grid-template-columns:repeat(6,1fr); gap:.8rem; margin-bottom:1.4rem; }
.mcard {
    background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:1rem 1.1rem; position:relative; overflow:hidden;
}
.mcard::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.mcard.b::before { background:linear-gradient(90deg,#3b82f6,#06b6d4); }
.mcard.v::before { background:linear-gradient(90deg,#7c3aed,#ec4899); }
.mcard.g::before { background:linear-gradient(90deg,#10b981,#06b6d4); }
.mcard.a::before { background:linear-gradient(90deg,#f59e0b,#f97316); }
.mcard.r::before { background:linear-gradient(90deg,#f43f5e,#f97316); }
.mcard.c::before { background:linear-gradient(90deg,#06b6d4,#7c3aed); }
.mlabel { color:var(--muted); font-size:.65rem; text-transform:uppercase; letter-spacing:1px; font-family:'DM Mono',monospace; }
.mval   { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:var(--text); line-height:1.1; }
.micon  { position:absolute; top:.9rem; right:.9rem; font-size:1.3rem; opacity:.3; }

/* SECTION CARD */
.scard {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1.3rem 1.6rem; margin-bottom:1.2rem;
}
.stitle {
    font-family:'Syne',sans-serif; font-size:.8rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1.5px;
    display:flex; align-items:center; gap:.4rem; margin-bottom:.9rem;
}
.stitle.b { color:#60a5fa; } .stitle.v { color:#a78bfa; }
.stitle.g { color:#34d399; } .stitle.a { color:#fbbf24; }
.stitle.c { color:#22d3ee; } .stitle.r { color:#fb7185; }

/* SUMMARY */
.summary {
    background:linear-gradient(135deg,rgba(59,130,246,.07),rgba(124,58,237,.04));
    border:1px solid rgba(59,130,246,.18); border-radius:10px;
    padding:1rem 1.3rem; font-size:.92rem; line-height:1.75;
    color:#c8d4e8; font-style:italic;
}

/* ENTITY */
.etag {
    display:inline-flex; align-items:center; gap:.3rem;
    padding:.25rem .7rem; border-radius:7px;
    font-size:.77rem; font-family:'DM Mono',monospace;
    margin:.2rem;
}
.PER  { background:rgba(124,58,237,.15);  border:1px solid rgba(124,58,237,.35); color:#c4b5fd; }
.ORG  { background:rgba(59,130,246,.15);  border:1px solid rgba(59,130,246,.35); color:#93c5fd; }
.LOC  { background:rgba(16,185,129,.15);  border:1px solid rgba(16,185,129,.35); color:#6ee7b7; }
.MISC { background:rgba(245,158,11,.15);  border:1px solid rgba(245,158,11,.35); color:#fcd34d; }
.DATE { background:rgba(244,63,94,.15);   border:1px solid rgba(244,63,94,.35);  color:#fda4af; }
.EDEF { background:rgba(74,90,122,.15);   border:1px solid rgba(74,90,122,.35);  color:#94a3b8; }

/* ACTION */
.aitem {
    display:flex; align-items:flex-start; gap:.7rem;
    padding:.7rem .9rem; border-radius:9px;
    background:var(--card2); border:1px solid var(--border); margin-bottom:.4rem;
}
.anum {
    width:20px; height:20px; border-radius:50%; flex-shrink:0; margin-top:1px;
    background:linear-gradient(135deg,#3b82f6,#7c3aed);
    display:flex; align-items:center; justify-content:center;
    font-size:.65rem; font-weight:700; color:#fff;
}
.atext { font-size:.85rem; color:#c8d4e8; line-height:1.5; }

/* INTENT */
.iitem {
    display:flex; justify-content:space-between; align-items:center;
    padding:.5rem .8rem; border-radius:8px;
    background:var(--card2); border:1px solid var(--border); margin-bottom:.35rem;
}
.ilabel { color:#c8d4e8; font-size:.8rem; font-family:'DM Mono',monospace; }
.ibadge {
    background:rgba(59,130,246,.18); color:#93c5fd;
    padding:.1rem .55rem; border-radius:20px; font-size:.72rem; font-weight:600;
}

/* CONF BAR */
.cbar { margin-bottom:.45rem; }
.cbar-top { display:flex; justify-content:space-between; font-size:.74rem; margin-bottom:.2rem; }
.cbar-name { color:#c8d4e8; font-family:'DM Mono',monospace; }
.cbar-val  { color:var(--muted); }
.cbar-track { background:var(--card2); border-radius:3px; height:5px; overflow:hidden; }
.cbar-fill  { height:100%; border-radius:3px; background:linear-gradient(90deg,#3b82f6,#7c3aed); }

/* TRANSCRIPT */
.tbox {
    background:var(--card2); border:1px solid var(--border); border-radius:10px;
    padding:1rem 1.3rem; font-size:.87rem; line-height:1.8; color:#c8d4e8;
    max-height:240px; overflow-y:auto;
}
.tbox::-webkit-scrollbar { width:3px; }
.tbox::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

/* EMPTY STATE */
.empty {
    text-align:center; padding:3rem 2rem; color:var(--muted);
}
.empty-icon { font-size:3rem; margin-bottom:.8rem; }
.empty-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#2d3e5e; margin-bottom:.4rem; }
.empty-sub   { font-size:.83rem; font-family:'DM Mono',monospace; color:#1e2d47; }

/* STREAMLIT OVERRIDES */
[data-testid="stSidebar"] { background:var(--card) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] h3 { font-family:'Syne',sans-serif !important; color:#60a5fa !important; font-size:.78rem !important; text-transform:uppercase; letter-spacing:1.5px; margin-top:1.2rem; }
.stTextArea textarea { background:var(--card2) !important; color:var(--text) !important; border:1px solid var(--border) !important; border-radius:9px !important; font-family:'DM Sans',sans-serif !important; font-size:.88rem !important; }
.stButton > button { background:linear-gradient(135deg,#1e40af,#6d28d9) !important; color:#fff !important; border:none !important; border-radius:9px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:.9rem !important; width:100% !important; padding:.55rem 1.5rem !important; transition:opacity .2s !important; }
.stButton > button:hover { opacity:.82 !important; }
.stDownloadButton > button { background:linear-gradient(135deg,#1e40af,#6d28d9) !important; color:#fff !important; border:none !important; border-radius:9px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; }
.stTabs [data-baseweb="tab-list"] { background:var(--card) !important; border-radius:9px !important; padding:.2rem !important; }
.stTabs [data-baseweb="tab"] { color:var(--muted) !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; font-size:.82rem !important; }
.stTabs [aria-selected="true"] { background:rgba(59,130,246,.18) !important; color:#93c5fd !important; border-radius:7px !important; }
div[data-testid="stExpander"] { background:var(--card2) !important; border:1px solid var(--border) !important; border-radius:9px !important; }
.stRadio > div { gap:.4rem !important; }
.stCheckbox span { color:var(--text) !important; font-size:.85rem !important; }
p, li, label { color:var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForSeq2SeqLM,
        pipeline, WhisperProcessor, WhisperForConditionalGeneration
    )
    from sentence_transformers import SentenceTransformer

    device = 0 if torch.cuda.is_available() else -1
    models = {}

    # 1. Intent — roberta-base
    ic_tok   = AutoTokenizer.from_pretrained("roberta-base")
    ic_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    models["intent"] = pipeline("text-classification", model=ic_model, tokenizer=ic_tok, device=device)

    # 2. NER — bert finetuned conll03
    models["ner"] = pipeline(
        "token-classification",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=device,
    )

    # 3. Summarizer — bart-large-cnn via AutoModel (no pipeline task string)
    summ_tok   = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summ_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summ_model.eval()
    if device == 0:
        summ_model = summ_model.cuda()
    models["summ_tok"]   = summ_tok
    models["summ_model"] = summ_model

    # 4. Action Extractor — flan-t5-base via AutoModel
    act_tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
    act_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    act_model.eval()
    if device == 0:
        act_model = act_model.cuda()
    models["act_tok"]   = act_tok
    models["act_model"] = act_model

    # 5. Embeddings — MiniLM
    models["embedding"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 6. Whisper STT — base
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


# ─── Pipeline Helpers ─────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes, stt_pipe):
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        arr, sr = sf.read(tmp_path)
        return stt_pipe({"array": arr.astype(np.float32), "sampling_rate": sr})["text"].strip()
    finally:
        os.unlink(tmp_path)


def run_intent(sentences, ic_pipe, max_s):
    if not sentences: return []
    res = ic_pipe(sentences[:max_s], truncation=True, max_length=128)
    return [{"sentence": s, "intent": r["label"], "confidence": round(r["score"], 4)}
            for s, r in zip(sentences[:max_s], res)]


def run_ner(sentences, ner_pipe, max_s):
    out = []
    for sent in sentences[:max_s]:
        for e in ner_pipe(sent):
            out.append({"entity": e["entity_group"], "word": e["word"],
                        "score": round(e["score"], 4), "sentence": sent[:80]})
    return out


def run_summarize(text, tok, model):
    inputs = tok(text[:1024], return_tensors="pt", truncation=True, max_length=512)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=130, min_new_tokens=30,
                             num_beams=4, early_stopping=True)
    return tok.decode(ids[0], skip_special_tokens=True)


def run_action(text, tok, model):
    prompt = (
        "Extract action items from this meeting. "
        "List each item on a new line with owner and deadline.\n\n"
        f"Transcript:\n{text[:700]}\n\nAction Items:"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=200)
    raw = tok.decode(ids[0], skip_special_tokens=True)
    lines = [l.strip().lstrip("-•*0123456789. ") for l in raw.split("\n") if l.strip()]
    return [l for l in lines if len(l) > 8][:10]


def run_embeddings(sentences, embed_model):
    if not sentences: return np.array([])
    return embed_model.encode(sentences[:10], normalize_embeddings=True)


# ─── Render Helpers ───────────────────────────────────────────────────────────
def cbar(label, val, color="#3b82f6"):
    pct = int(val * 100)
    return f"""<div class="cbar">
    <div class="cbar-top"><span class="cbar-name">{label}</span><span class="cbar-val">{pct}%</span></div>
    <div class="cbar-track"><div class="cbar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},#7c3aed);"></div></div>
    </div>"""


SAMPLE = """Alice: Good morning everyone. Let's start the weekly product sync.
John, can you give us the engineering update?
John: Sure. We completed the user authentication module this week.
The database migration to PostgreSQL is 90% done — we expect to finish by Thursday.
Sarah: Great news. On the design side, we finalized the new dashboard mockups.
I'll share them with the team by end of day.
Alice: What about the security audit?
John: Mike is leading that. Mike, any update?
Mike: Yes, we found three medium-priority vulnerabilities. I'll send a detailed report to all stakeholders by tomorrow morning.
Alice: We need to schedule a follow-up with the security team next week. Sarah, can you set that up?
Sarah: Absolutely. I'll send the calendar invite today.
Alice: One more thing — we need to finalize the Q4 budget proposal. John, coordinate with finance before the board meeting on December 5th.
John: Got it. I'll set up a meeting with the CFO this week.
Alice: Great. Any blockers?
Mike: No blockers from my side.
Sarah: Same here.
Alice: Perfect. Thanks everyone."""


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # HERO
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">🎙️ NLU Meeting Intelligence</h1>
        <p class="hero-sub">Voice-to-Action · 6 Transformer Models · End-to-End NLP Pipeline</p>
        <div class="pills">
            <span class="pill">🎙️ Whisper STT</span>
            <span class="pill">🎯 RoBERTa Intent</span>
            <span class="pill">🏷️ BERT NER</span>
            <span class="pill">📋 BART Summary</span>
            <span class="pill">✅ T5 Actions</span>
            <span class="pill">🔗 MPNet Embed</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # PIPELINE FLOW
    st.markdown("""
    <div class="pipe">
        <div class="pipe-step"><div class="pipe-icon">🎙️</div><div class="pipe-name">Whisper</div><div class="pipe-sub">STT</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">🎯</div><div class="pipe-name">RoBERTa</div><div class="pipe-sub">Intent</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">🏷️</div><div class="pipe-name">BERT</div><div class="pipe-sub">NER</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">📋</div><div class="pipe-name">BART</div><div class="pipe-sub">Summarize</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">✅</div><div class="pipe-name">T5</div><div class="pipe-sub">Actions</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">🔗</div><div class="pipe-name">MPNet</div><div class="pipe-sub">Embeddings</div></div>
    </div>""", unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.markdown("### ⚙️ Input Mode")
        mode = st.radio("Mode", ["📝 Text", "🎵 Audio"], label_visibility="collapsed")
        st.markdown("### 🔧 Settings")
        max_ic  = st.slider("Intent sentences",  5, 20, 12)
        max_ner = st.slider("NER sentences",      5, 15, 10)
        st.markdown("### 📊 Display")
        show_sim    = st.checkbox("Similarity pairs",   True)
        show_timing = st.checkbox("Processing times",   True)
        show_json   = st.checkbox("Raw JSON",           False)
        st.markdown("---")
        st.markdown("""<div style="color:#1e2d47;font-size:.72rem;font-family:'DM Mono',monospace;line-height:1.9;">
        openai/whisper-base<br>roberta-base<br>bert-cased-conll03<br>
        facebook/bart-large-cnn<br>google/flan-t5-base<br>all-MiniLM-L6-v2</div>""",
        unsafe_allow_html=True)

    # INPUT
    transcript = ""
    audio_bytes = None

    if "📝 Text" in mode:
        transcript = st.text_area("Transcript", value=SAMPLE, height=180, label_visibility="collapsed")
        c1, c2 = st.columns([1, 1])
        with c1:
            sample_btn = st.button("📋 Load Sample")
        with c2:
            run_btn = st.button("🚀 Analyze Meeting")
        if sample_btn:
            transcript = SAMPLE
    else:
        af = st.file_uploader("Upload Audio (WAV/MP3/M4A)", type=["wav","mp3","m4a","ogg"], label_visibility="collapsed")
        run_btn    = st.button("🎙️ Transcribe & Analyze")
        sample_btn = False
        if af:
            audio_bytes = af.read()

    # RUN
    if run_btn and (transcript.strip() or audio_bytes):
        with st.spinner("Loading models…"):
            try:
                M = load_models()
            except Exception as e:
                st.error(f"Model loading error: {e}")
                st.stop()

        if audio_bytes:
            with st.spinner("Transcribing audio…"):
                try:
                    transcript = transcribe_audio(audio_bytes, M["stt"])
                except Exception as e:
                    st.error(f"Transcription error: {e}"); st.stop()

        if not transcript.strip():
            st.warning("No transcript to analyze."); st.stop()

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if len(s.strip()) > 5]
        res   = {"transcript": transcript, "sentences": sents, "timing": {}}
        bar   = st.progress(0, text="Running pipeline…")

        t = time.time()
        res["intents"] = run_intent(sents, M["intent"], max_ic)
        res["timing"]["Intent"]  = round(time.time()-t, 2); bar.progress(20)

        t = time.time()
        res["entities"] = run_ner(sents, M["ner"], max_ner)
        res["timing"]["NER"]     = round(time.time()-t, 2); bar.progress(40)

        t = time.time()
        res["summary"] = run_summarize(transcript, M["summ_tok"], M["summ_model"])
        res["timing"]["Summary"] = round(time.time()-t, 2); bar.progress(60)

        t = time.time()
        res["actions"] = run_action(transcript, M["act_tok"], M["act_model"])
        res["timing"]["Actions"] = round(time.time()-t, 2); bar.progress(80)

        t = time.time()
        emb = run_embeddings(sents, M["embedding"])
        res["embeddings"] = emb
        res["similarity"] = (emb @ emb.T) if len(emb) > 1 else np.array([])
        res["timing"]["Embeddings"] = round(time.time()-t, 2); bar.progress(100)
        time.sleep(.2); bar.empty()
        st.session_state["res"] = res

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if "res" not in st.session_state:
        st.markdown("""<div class="empty">
            <div class="empty-icon">🎙️</div>
            <div class="empty-title">Ready to Analyze</div>
            <div class="empty-sub">Paste a transcript · click Analyze Meeting</div>
        </div>""", unsafe_allow_html=True)
        return

    R = st.session_state["res"]
    sents = R["sentences"]

    # METRICS
    n_w   = len(R["transcript"].split())
    n_i   = len(set(x["intent"] for x in R["intents"]))
    n_e   = len(R["entities"])
    n_a   = len(R["actions"])
    n_s   = len(sents)
    n_dim = R["embeddings"].shape[1] if len(R["embeddings"]) > 0 else 0
    st.markdown(f"""
    <div class="mgrid">
        <div class="mcard b"><div class="micon">💬</div><div class="mlabel">Words</div><div class="mval">{n_w:,}</div></div>
        <div class="mcard v"><div class="micon">🎯</div><div class="mlabel">Intents</div><div class="mval">{n_i}</div></div>
        <div class="mcard g"><div class="micon">🏷️</div><div class="mlabel">Entities</div><div class="mval">{n_e}</div></div>
        <div class="mcard a"><div class="micon">✅</div><div class="mlabel">Actions</div><div class="mval">{n_a}</div></div>
        <div class="mcard r"><div class="micon">📝</div><div class="mlabel">Sentences</div><div class="mval">{n_s}</div></div>
        <div class="mcard c"><div class="micon">🔢</div><div class="mlabel">Embed Dims</div><div class="mval">{n_dim}</div></div>
    </div>""", unsafe_allow_html=True)

    # TABS
    t1, t2, t3, t4, t5, t6 = st.tabs(["📋 Summary", "🎯 Intents", "🏷️ Entities", "✅ Actions", "🔗 Embeddings", "📄 Transcript"])

    # TAB 1 — SUMMARY
    with t1:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="stitle b">📋 Meeting Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary">{R.get("summary","—")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_timing:
            st.markdown('<div class="scard">', unsafe_allow_html=True)
            st.markdown('<div class="stitle c">⏱️ Processing Times</div>', unsafe_allow_html=True)
            total = sum(R["timing"].values())
            for k, v in R["timing"].items():
                st.markdown(cbar(f"{k}  ({v}s)", v / max(total, .001)), unsafe_allow_html=True)
            st.markdown(f'<p style="color:var(--muted);font-size:.75rem;font-family:\'DM Mono\',monospace;margin-top:.4rem;">Total: {total:.2f}s</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2 — INTENTS
    with t2:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="stitle v">🎯 Intent Distribution</div>', unsafe_allow_html=True)
        intents = R.get("intents", [])
        if intents:
            counts = Counter(x["intent"] for x in intents)
            total  = len(intents)
            for lbl, cnt in counts.most_common():
                st.markdown(f'<div class="iitem"><span class="ilabel">{lbl}</span><span class="ibadge">{cnt}/{total}</span></div>', unsafe_allow_html=True)
                st.markdown(cbar(lbl, cnt/total, "#7c3aed"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Per-sentence details"):
            for x in intents[:15]:
                c = "#10b981" if x["confidence"]>.8 else "#f59e0b" if x["confidence"]>.6 else "#f43f5e"
                st.markdown(f"""<div style="padding:.45rem .8rem;border-left:3px solid {c};margin-bottom:.35rem;background:var(--card2);border-radius:0 7px 7px 0;">
                    <div style="font-size:.72rem;color:var(--muted);font-family:'DM Mono',monospace;">{x['intent']} · {int(x['confidence']*100)}%</div>
                    <div style="font-size:.83rem;color:#c8d4e8;">{x['sentence'][:120]}</div>
                </div>""", unsafe_allow_html=True)

    # TAB 3 — ENTITIES
    with t3:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="stitle g">🏷️ Named Entities</div>', unsafe_allow_html=True)
        ents = R.get("entities", [])
        if ents:
            groups = {}
            for e in ents:
                groups.setdefault(e["entity"], []).append(e)
            for etype, lst in sorted(groups.items()):
                cls = etype if etype in ["PER","ORG","LOC","MISC","DATE"] else "EDEF"
                tags = "".join(f'<span class="etag {cls}"><b>{etype}</b> {e["word"]} <small>({e["score"]:.2f})</small></span>' for e in lst[:12])
                st.markdown(f'<div style="margin-bottom:.9rem;"><div style="font-size:.67rem;color:var(--muted);font-family:\'DM Mono\',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:.3rem;">{etype} ({len(lst)})</div>{tags}</div>', unsafe_allow_html=True)
        else:
            st.info("No entities detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 4 — ACTIONS
    with t4:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="stitle a">✅ Action Items</div>', unsafe_allow_html=True)
        actions = R.get("actions", [])
        if actions:
            for i, a in enumerate(actions, 1):
                st.markdown(f'<div class="aitem"><div class="anum">{i}</div><div class="atext">{a}</div></div>', unsafe_allow_html=True)
            st.download_button("⬇️ Export Actions", "\n".join(f"{i}. {a}" for i, a in enumerate(actions,1)), "action_items.txt", "text/plain")
        else:
            st.info("No action items extracted.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 5 — EMBEDDINGS
    with t5:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="stitle c">🔗 Sentence Embeddings</div>', unsafe_allow_html=True)
        emb = R.get("embeddings", np.array([]))
        if len(emb) > 0:
            sh = emb.shape
            st.markdown(f"""<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;">
            <div style="background:rgba(6,182,212,.1);border:1px solid rgba(6,182,212,.25);border-radius:8px;padding:.5rem 1rem;">
                <div style="color:var(--muted);font-size:.65rem;font-family:'DM Mono',monospace;">SENTENCES</div>
                <div style="color:#22d3ee;font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;">{sh[0]}</div>
            </div>
            <div style="background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.25);border-radius:8px;padding:.5rem 1rem;">
                <div style="color:var(--muted);font-size:.65rem;font-family:'DM Mono',monospace;">DIMENSIONS</div>
                <div style="color:#a78bfa;font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;">{sh[1]}</div>
            </div>
            <div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);border-radius:8px;padding:.5rem 1rem;">
                <div style="color:var(--muted);font-size:.65rem;font-family:'DM Mono',monospace;">TOTAL VALUES</div>
                <div style="color:#34d399;font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;">{sh[0]*sh[1]:,}</div>
            </div></div>""", unsafe_allow_html=True)

            # Heatmap preview
            preview = emb[:min(5,sh[0]), :20]
            vmin, vmax = preview.min(), preview.max()
            cells = ""
            for row in preview:
                for val in row:
                    n = (val - vmin) / max(vmax - vmin, 1e-8)
                    bg = f"rgba(59,{int(130+n*52)},{int(246-n*34)},{.12+n*.25})"
                    cells += f'<div style="height:30px;border-radius:4px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:.6rem;font-family:\'DM Mono\',monospace;color:rgba(255,255,255,.4);">{val:.2f}</div>'
            st.markdown(f'<div style="display:grid;grid-template-columns:repeat(20,1fr);gap:2px;margin-top:.3rem;">{cells}</div>', unsafe_allow_html=True)

            if show_sim and len(emb) > 1:
                st.markdown('<div class="stitle c" style="margin-top:1.2rem;font-size:.75rem;">🔁 Top Similar Pairs</div>', unsafe_allow_html=True)
                sim = R.get("similarity", np.array([]))
                if len(sim) > 1:
                    n   = min(len(sents), len(sim))
                    pairs = sorted([(sim[i,j],i,j) for i in range(n) for j in range(i+1,n)], reverse=True)
                    for score, i, j in pairs[:4]:
                        col = "#10b981" if score>.8 else "#f59e0b" if score>.6 else "#64748b"
                        s1  = sents[i][:70]+"…" if len(sents[i])>70 else sents[i]
                        s2  = sents[j][:70]+"…" if len(sents[j])>70 else sents[j]
                        st.markdown(f"""<div style="padding:.65rem .9rem;background:var(--card2);border:1px solid var(--border);border-radius:9px;margin-bottom:.4rem;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:.35rem;">
                                <span style="color:var(--muted);font-size:.68rem;font-family:'DM Mono',monospace;">COSINE SIMILARITY</span>
                                <span style="color:{col};font-family:'Syne',sans-serif;font-size:.95rem;font-weight:800;">{score:.3f}</span>
                            </div>
                            <div style="font-size:.8rem;color:#94a3b8;">"{s1}"</div>
                            <div style="font-size:.72rem;color:var(--muted);margin:.2rem 0;">↔</div>
                            <div style="font-size:.8rem;color:#94a3b8;">"{s2}"</div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.info("No embeddings computed.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 6 — TRANSCRIPT
    with t6:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="stitle r">📄 Full Transcript</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tbox">{R["transcript"].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Transcript TXT", R["transcript"], "transcript.txt", "text/plain")
        with c2:
            if show_json:
                safe = {k:v for k,v in R.items() if k not in ["embeddings","similarity"]}
                st.download_button("⬇️ Full JSON", json.dumps(safe, indent=2, ensure_ascii=False), "analysis.json", "application/json")

    if show_json:
        with st.expander("Raw JSON"):
            safe = {k:v for k,v in R.items() if k not in ["embeddings","similarity"]}
            st.json(safe)


if __name__ == "__main__":
    main()
