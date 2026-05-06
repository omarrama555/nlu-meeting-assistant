import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
import numpy as np
import re
import json
import time
import tempfile
from collections import Counter

st.set_page_config(page_title="NLU Meeting Intelligence", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
:root{--bg:#03060e;--card:#070c1a;--card2:#0a1020;--border:#111929;--text:#e2e8f0;--muted:#334466;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem;max-width:1280px;}
.hero{background:linear-gradient(135deg,#03060e 0%,#07102a 50%,#03060e 100%);border:1px solid var(--border);border-radius:16px;padding:1.8rem 2.2rem 1.6rem;margin-bottom:1.4rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(59,130,246,.1) 0%,transparent 70%);}
.hero-title{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#22d3ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .25rem;}
.hero-sub{color:var(--muted);font-size:.82rem;font-family:'DM Mono',monospace;margin:0 0 1rem;}
.pills{display:flex;flex-wrap:wrap;gap:.35rem;}
.pill{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.18);color:#7eb3f8;padding:.18rem .65rem;border-radius:30px;font-size:.7rem;font-family:'DM Mono',monospace;}
.pipe{display:flex;align-items:center;flex-wrap:wrap;background:var(--card);border:1px solid var(--border);border-radius:11px;padding:.7rem 1.1rem;margin-bottom:1.3rem;}
.pipe-step{display:flex;flex-direction:column;align-items:center;flex:1;min-width:70px;padding:.25rem .4rem;text-align:center;}
.pipe-icon{font-size:1.2rem;margin-bottom:.15rem;}
.pipe-name{font-size:.72rem;font-weight:700;color:var(--text);font-family:'Syne',sans-serif;}
.pipe-sub{font-size:.62rem;color:var(--muted);font-family:'DM Mono',monospace;}
.pipe-arr{color:var(--muted);font-size:.9rem;}
.mgrid{display:grid;grid-template-columns:repeat(6,1fr);gap:.7rem;margin-bottom:1.3rem;}
.mcard{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:.9rem 1rem;position:relative;overflow:hidden;}
.mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.mcard.b::before{background:linear-gradient(90deg,#3b82f6,#06b6d4);}
.mcard.v::before{background:linear-gradient(90deg,#7c3aed,#ec4899);}
.mcard.g::before{background:linear-gradient(90deg,#10b981,#06b6d4);}
.mcard.a::before{background:linear-gradient(90deg,#f59e0b,#f97316);}
.mcard.r::before{background:linear-gradient(90deg,#f43f5e,#f97316);}
.mcard.c::before{background:linear-gradient(90deg,#06b6d4,#7c3aed);}
.mlabel{color:var(--muted);font-size:.62rem;text-transform:uppercase;letter-spacing:1px;font-family:'DM Mono',monospace;}
.mval{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;color:var(--text);line-height:1.1;}
.micon{position:absolute;top:.8rem;right:.8rem;font-size:1.2rem;opacity:.2;}
.scard{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:1.2rem 1.5rem;margin-bottom:1.1rem;}
.stitle{font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;display:flex;align-items:center;gap:.4rem;margin-bottom:.85rem;}
.stitle.b{color:#60a5fa;}.stitle.v{color:#a78bfa;}.stitle.g{color:#34d399;}.stitle.a{color:#fbbf24;}.stitle.c{color:#22d3ee;}.stitle.r{color:#fb7185;}
.summary{background:linear-gradient(135deg,rgba(59,130,246,.05),rgba(124,58,237,.03));border:1px solid rgba(59,130,246,.13);border-radius:9px;padding:.9rem 1.2rem;font-size:.9rem;line-height:1.75;color:#b0c4dd;font-style:italic;}
.etag{display:inline-flex;align-items:center;gap:.3rem;padding:.22rem .65rem;border-radius:6px;font-size:.75rem;font-family:'DM Mono',monospace;margin:.18rem;}
.PER{background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.28);color:#c4b5fd;}
.ORG{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.28);color:#93c5fd;}
.LOC{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.28);color:#6ee7b7;}
.MISC{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.28);color:#fcd34d;}
.DATE{background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.28);color:#fda4af;}
.EDEF{background:rgba(74,90,122,.1);border:1px solid rgba(74,90,122,.28);color:#94a3b8;}
.aitem{display:flex;align-items:flex-start;gap:.65rem;padding:.65rem .85rem;border-radius:8px;background:var(--card2);border:1px solid var(--border);margin-bottom:.35rem;}
.anum{width:19px;height:19px;border-radius:50%;flex-shrink:0;margin-top:2px;background:linear-gradient(135deg,#3b82f6,#7c3aed);display:flex;align-items:center;justify-content:center;font-size:.62rem;font-weight:700;color:#fff;}
.atext{font-size:.83rem;color:#c8d4e8;line-height:1.5;}
.iitem{display:flex;justify-content:space-between;align-items:center;padding:.45rem .75rem;border-radius:7px;background:var(--card2);border:1px solid var(--border);margin-bottom:.3rem;}
.ilabel{color:#c8d4e8;font-size:.78rem;font-family:'DM Mono',monospace;}
.ibadge{background:rgba(59,130,246,.13);color:#93c5fd;padding:.08rem .5rem;border-radius:18px;font-size:.7rem;font-weight:600;}
.cbar{margin-bottom:.4rem;}
.cbar-top{display:flex;justify-content:space-between;font-size:.72rem;margin-bottom:.18rem;}
.cbar-name{color:#c8d4e8;font-family:'DM Mono',monospace;}
.cbar-val{color:var(--muted);}
.cbar-track{background:var(--card2);border-radius:3px;height:4px;overflow:hidden;}
.cbar-fill{height:100%;border-radius:3px;}
.tbox{background:var(--card2);border:1px solid var(--border);border-radius:9px;padding:.9rem 1.2rem;font-size:.85rem;line-height:1.8;color:#c8d4e8;max-height:220px;overflow-y:auto;}
.tbox::-webkit-scrollbar{width:3px;}
.tbox::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.empty{text-align:center;padding:3rem 2rem;}
.empty-icon{font-size:2.8rem;margin-bottom:.7rem;}
.empty-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#1e2d47;margin-bottom:.35rem;}
.empty-sub{font-size:.8rem;font-family:'DM Mono',monospace;color:#172038;}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] h3{font-family:'Syne',sans-serif!important;color:#60a5fa!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:1.5px;margin-top:1.1rem;}
.stTextArea textarea{background:var(--card2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-size:.86rem!important;}
.stButton>button{background:linear-gradient(135deg,#1e3a8a,#5b21b6)!important;color:#e2e8f0!important;border:1px solid rgba(59,130,246,.18)!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.88rem!important;width:100%!important;padding:.5rem 1.4rem!important;transition:opacity .2s!important;}
.stButton>button:hover{opacity:.8!important;}
.stDownloadButton>button{background:linear-gradient(135deg,#1e3a8a,#5b21b6)!important;color:#e2e8f0!important;border:1px solid rgba(59,130,246,.18)!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:600!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border-radius:8px!important;padding:.18rem!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-family:'Syne',sans-serif!important;font-weight:600!important;font-size:.8rem!important;}
.stTabs [aria-selected="true"]{background:rgba(59,130,246,.13)!important;color:#93c5fd!important;border-radius:6px!important;}
div[data-testid="stExpander"]{background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stRadio>div{gap:.35rem!important;}
.stCheckbox span,.stRadio span{color:var(--text)!important;font-size:.83rem!important;}
p,li,label,[data-testid="stMarkdownContainer"] p{color:var(--text)!important;}
.stAlert{background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
[data-testid="stFileUploader"]{background:var(--card2)!important;border:1px dashed var(--border)!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    # Explicit imports only — avoids torchvision-dependent sub-modules
    from transformers.models.whisper.processing_whisper import WhisperProcessor
    from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
    from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
    from transformers.models.bart.modeling_bart import BartForConditionalGeneration
    from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
    from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
    from transformers.models.bert.modeling_bert import BertForTokenClassification
    from transformers.pipelines import pipeline as hf_pipeline
    from transformers.pipelines.zero_shot_classification import ZeroShotClassificationPipeline
    from sentence_transformers import SentenceTransformer

    dev = 0 if torch.cuda.is_available() else -1
    M = {}

    # 1. STT — Whisper-base
    proc  = WhisperProcessor.from_pretrained("openai/whisper-base")
    wmdl  = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    M["stt"] = hf_pipeline(
        "automatic-speech-recognition",
        model=wmdl, tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        max_new_tokens=256, chunk_length_s=30, device=dev,
    )

    # 2. Intent — BART-large-MNLI zero-shot
    M["intent"] = hf_pipeline("zero-shot-classification",
                               model="facebook/bart-large-mnli", device=dev)
    M["intent_labels"] = [
        "task assignment", "decision made", "question asked",
        "information sharing", "follow-up required", "blocker reported",
        "approval requested", "status update",
    ]

    # 3. NER — BERT CoNLL03
    M["ner"] = hf_pipeline("token-classification",
                            model="dbmdz/bert-large-cased-finetuned-conll03-english",
                            aggregation_strategy="simple", device=dev)

    # 4. Summarizer — BART-large-CNN (direct generate)
    M["summ_tok"] = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
    M["summ_mdl"] = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").eval()
    if dev == 0: M["summ_mdl"] = M["summ_mdl"].cuda()

    # 5. Actions — Flan-T5-base (direct generate)
    M["act_tok"] = T5TokenizerFast.from_pretrained("google/flan-t5-base")
    M["act_mdl"] = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").eval()
    if dev == 0: M["act_mdl"] = M["act_mdl"].cuda()

    # 6. Embeddings — MiniLM
    M["emb"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return M


def transcribe(audio_bytes, stt):
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        p = f.name
    try:
        arr, sr = sf.read(p)
        return stt({"array": arr.astype(np.float32), "sampling_rate": sr})["text"].strip()
    finally:
        os.unlink(p)


def run_intent(sents, pipe, labels, n):
    out = []
    for s in sents[:n]:
        r = pipe(s, candidate_labels=labels, truncation=True)
        out.append({"sentence": s, "intent": r["labels"][0], "confidence": round(r["scores"][0], 4)})
    return out


def run_ner(sents, pipe, n):
    out = []
    for s in sents[:n]:
        for e in pipe(s):
            out.append({"entity": e["entity_group"], "word": e["word"],
                        "score": round(e["score"], 4), "sentence": s[:80]})
    return out


def run_summarize(text, tok, mdl):
    inp = tok(text[:1024], return_tensors="pt", truncation=True, max_length=512)
    if next(mdl.parameters()).is_cuda:
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        ids = mdl.generate(**inp, max_new_tokens=130, min_new_tokens=30, num_beams=4, early_stopping=True)
    return tok.decode(ids[0], skip_special_tokens=True)


def run_actions(text, tok, mdl):
    prompt = f"Extract action items from this meeting. List each on a new line with owner and deadline.\n\nTranscript:\n{text[:700]}\n\nAction Items:"
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    if next(mdl.parameters()).is_cuda:
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        ids = mdl.generate(**inp, max_new_tokens=200)
    raw = tok.decode(ids[0], skip_special_tokens=True)
    return [l.strip().lstrip("-•*0123456789. ") for l in raw.split("\n") if len(l.strip()) > 8][:10]


def run_embeddings(sents, mdl):
    if not sents:
        return np.array([])
    return mdl.encode(sents[:10], normalize_embeddings=True)


def cbar(label, val, color="#3b82f6"):
    p = int(val * 100)
    return (f'<div class="cbar"><div class="cbar-top">'
            f'<span class="cbar-name">{label}</span><span class="cbar-val">{p}%</span></div>'
            f'<div class="cbar-track"><div class="cbar-fill" style="width:{p}%;'
            f'background:linear-gradient(90deg,{color},#7c3aed);"></div></div></div>')


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


def main():
    st.markdown("""<div class="hero">
        <h1 class="hero-title">🎙️ NLU Meeting Intelligence</h1>
        <p class="hero-sub">Audio → Transcript → Intent · NER · Summary · Actions · Embeddings</p>
        <div class="pills">
            <span class="pill">🎙️ Whisper STT</span>
            <span class="pill">🎯 BART Zero-Shot Intent</span>
            <span class="pill">🏷️ BERT NER</span>
            <span class="pill">📋 BART Summary</span>
            <span class="pill">✅ Flan-T5 Actions</span>
            <span class="pill">🔗 MiniLM Embeddings</span>
        </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="pipe">
        <div class="pipe-step"><div class="pipe-icon">🎙️</div><div class="pipe-name">Whisper</div><div class="pipe-sub">STT</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">🎯</div><div class="pipe-name">BART</div><div class="pipe-sub">Intent</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">🏷️</div><div class="pipe-name">BERT</div><div class="pipe-sub">NER</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">📋</div><div class="pipe-name">BART</div><div class="pipe-sub">Summary</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">✅</div><div class="pipe-name">Flan-T5</div><div class="pipe-sub">Actions</div></div>
        <span class="pipe-arr">→</span>
        <div class="pipe-step"><div class="pipe-icon">🔗</div><div class="pipe-name">MiniLM</div><div class="pipe-sub">Embeddings</div></div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Input Mode")
        mode = st.radio("Mode", ["📝 Text", "🎵 Audio"], label_visibility="collapsed")
        st.markdown("### 🔧 Settings")
        max_ic  = st.slider("Intent sentences",  5, 20, 12)
        max_ner = st.slider("NER sentences",      5, 15, 10)
        st.markdown("### 📊 Display")
        show_sim    = st.checkbox("Similarity pairs",  True)
        show_timing = st.checkbox("Processing times",  True)
        show_json   = st.checkbox("Raw JSON",          False)
        st.markdown("---")
        st.markdown('<div style="color:#1a2a40;font-size:.68rem;font-family:\'DM Mono\',monospace;line-height:2;">openai/whisper-base<br>facebook/bart-large-mnli<br>dbmdz/bert-cased-conll03<br>facebook/bart-large-cnn<br>google/flan-t5-base<br>all-MiniLM-L6-v2</div>', unsafe_allow_html=True)

    transcript = ""
    audio_bytes = None

    if "📝 Text" in mode:
        transcript = st.text_area("", value=SAMPLE, height=175, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1: sample_btn = st.button("📋 Load Sample")
        with c2: run_btn    = st.button("🚀 Analyze Meeting")
        if sample_btn: transcript = SAMPLE
    else:
        af = st.file_uploader("Upload audio (WAV / MP3 / M4A / OGG)",
                               type=["wav","mp3","m4a","ogg"], label_visibility="collapsed")
        run_btn    = st.button("🎙️ Transcribe & Analyze")
        sample_btn = False
        if af: audio_bytes = af.read()

    if run_btn and (transcript.strip() or audio_bytes):
        with st.spinner("Loading models — first run takes a few minutes…"):
            try:
                M = load_models()
            except Exception as e:
                st.error(f"Model load failed: {e}"); st.stop()

        if audio_bytes:
            with st.spinner("Transcribing audio…"):
                try:
                    transcript = transcribe(audio_bytes, M["stt"])
                except Exception as e:
                    st.error(f"Transcription failed: {e}"); st.stop()

        if not transcript.strip():
            st.warning("Nothing to analyze."); st.stop()

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if len(s.strip()) > 5]
        R   = {"transcript": transcript, "sentences": sents, "timing": {}}
        bar = st.progress(0, text="Running pipeline…")

        t = time.time(); R["intents"]  = run_intent(sents, M["intent"], M["intent_labels"], max_ic);  R["timing"]["Intent"]     = round(time.time()-t, 2); bar.progress(20)
        t = time.time(); R["entities"] = run_ner(sents, M["ner"], max_ner);                            R["timing"]["NER"]        = round(time.time()-t, 2); bar.progress(40)
        t = time.time(); R["summary"]  = run_summarize(transcript, M["summ_tok"], M["summ_mdl"]);      R["timing"]["Summary"]    = round(time.time()-t, 2); bar.progress(60)
        t = time.time(); R["actions"]  = run_actions(transcript, M["act_tok"], M["act_mdl"]);          R["timing"]["Actions"]    = round(time.time()-t, 2); bar.progress(80)
        t = time.time()
        emb = run_embeddings(sents, M["emb"])
        R["embeddings"] = emb
        R["similarity"] = (emb @ emb.T) if len(emb) > 1 else np.array([])
        R["timing"]["Embeddings"] = round(time.time()-t, 2); bar.progress(100)
        time.sleep(.15); bar.empty()
        st.session_state["R"] = R

    if "R" not in st.session_state:
        st.markdown('<div class="empty"><div class="empty-icon">🎙️</div><div class="empty-title">Ready to Analyze</div><div class="empty-sub">Paste a transcript or upload audio · then click Analyze</div></div>', unsafe_allow_html=True)
        return

    R     = st.session_state["R"]
    sents = R["sentences"]
    n_w   = len(R["transcript"].split())
    n_i   = len(set(x["intent"] for x in R["intents"]))
    n_e   = len(R["entities"])
    n_a   = len(R["actions"])
    n_s   = len(sents)
    n_d   = R["embeddings"].shape[1] if len(R["embeddings"]) > 0 else 0

    st.markdown(f"""<div class="mgrid">
        <div class="mcard b"><div class="micon">💬</div><div class="mlabel">Words</div><div class="mval">{n_w:,}</div></div>
        <div class="mcard v"><div class="micon">🎯</div><div class="mlabel">Intents</div><div class="mval">{n_i}</div></div>
        <div class="mcard g"><div class="micon">🏷️</div><div class="mlabel">Entities</div><div class="mval">{n_e}</div></div>
        <div class="mcard a"><div class="micon">✅</div><div class="mlabel">Actions</div><div class="mval">{n_a}</div></div>
        <div class="mcard r"><div class="micon">📝</div><div class="mlabel">Sentences</div><div class="mval">{n_s}</div></div>
        <div class="mcard c"><div class="micon">🔢</div><div class="mlabel">Embed Dims</div><div class="mval">{n_d}</div></div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(["📋 Summary", "🎯 Intents", "🏷️ Entities", "✅ Actions", "🔗 Embeddings", "📄 Transcript"])

    with t1:
        st.markdown('<div class="scard"><div class="stitle b">📋 Meeting Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary">{R.get("summary","—")}</div></div>', unsafe_allow_html=True)
        if show_timing:
            st.markdown('<div class="scard"><div class="stitle c">⏱️ Processing Times</div>', unsafe_allow_html=True)
            total = sum(R["timing"].values())
            for k, v in R["timing"].items():
                st.markdown(cbar(f"{k} ({v}s)", v / max(total, .001)), unsafe_allow_html=True)
            st.markdown(f'<p style="color:var(--muted);font-size:.72rem;font-family:\'DM Mono\',monospace;margin-top:.3rem;">Total: {total:.2f}s</p></div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="scard"><div class="stitle v">🎯 Intent Distribution</div>', unsafe_allow_html=True)
        intents = R.get("intents", [])
        if intents:
            counts = Counter(x["intent"] for x in intents)
            tot = len(intents)
            for lbl, cnt in counts.most_common():
                st.markdown(f'<div class="iitem"><span class="ilabel">{lbl}</span><span class="ibadge">{cnt}/{tot}</span></div>', unsafe_allow_html=True)
                st.markdown(cbar(lbl, cnt / tot, "#7c3aed"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("Per-sentence details"):
            for x in intents[:15]:
                c = "#10b981" if x["confidence"] > .8 else "#f59e0b" if x["confidence"] > .6 else "#f43f5e"
                st.markdown(f'<div style="padding:.4rem .75rem;border-left:3px solid {c};margin-bottom:.3rem;background:var(--card2);border-radius:0 6px 6px 0;"><div style="font-size:.7rem;color:var(--muted);font-family:\'DM Mono\',monospace;">{x["intent"]} · {int(x["confidence"]*100)}%</div><div style="font-size:.82rem;color:#c8d4e8;">{x["sentence"][:120]}</div></div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="scard"><div class="stitle g">🏷️ Named Entities</div>', unsafe_allow_html=True)
        ents = R.get("entities", [])
        if ents:
            groups = {}
            for e in ents:
                groups.setdefault(e["entity"], []).append(e)
            for etype, lst in sorted(groups.items()):
                cls  = etype if etype in ["PER","ORG","LOC","MISC","DATE"] else "EDEF"
                tags = "".join(f'<span class="etag {cls}"><b>{etype}</b> {e["word"]} <small>({e["score"]:.2f})</small></span>' for e in lst[:12])
                st.markdown(f'<div style="margin-bottom:.8rem;"><div style="font-size:.65rem;color:var(--muted);font-family:\'DM Mono\',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:.25rem;">{etype} ({len(lst)})</div>{tags}</div>', unsafe_allow_html=True)
        else:
            st.info("No entities detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t4:
        st.markdown('<div class="scard"><div class="stitle a">✅ Action Items</div>', unsafe_allow_html=True)
        actions = R.get("actions", [])
        if actions:
            for i, a in enumerate(actions, 1):
                st.markdown(f'<div class="aitem"><div class="anum">{i}</div><div class="atext">{a}</div></div>', unsafe_allow_html=True)
            st.download_button("⬇️ Export Actions", "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1)), "action_items.txt", "text/plain")
        else:
            st.info("No action items found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t5:
        st.markdown('<div class="scard"><div class="stitle c">🔗 Sentence Embeddings</div>', unsafe_allow_html=True)
        emb = R.get("embeddings", np.array([]))
        if len(emb) > 0:
            sh = emb.shape
            st.markdown(f"""<div style="display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:.9rem;">
            <div style="background:rgba(6,182,212,.07);border:1px solid rgba(6,182,212,.18);border-radius:7px;padding:.45rem .9rem;"><div style="color:var(--muted);font-size:.62rem;font-family:'DM Mono',monospace;">SENTENCES</div><div style="color:#22d3ee;font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;">{sh[0]}</div></div>
            <div style="background:rgba(124,58,237,.07);border:1px solid rgba(124,58,237,.18);border-radius:7px;padding:.45rem .9rem;"><div style="color:var(--muted);font-size:.62rem;font-family:'DM Mono',monospace;">DIMENSIONS</div><div style="color:#a78bfa;font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;">{sh[1]}</div></div>
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.18);border-radius:7px;padding:.45rem .9rem;"><div style="color:var(--muted);font-size:.62rem;font-family:'DM Mono',monospace;">TOTAL VALUES</div><div style="color:#34d399;font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;">{sh[0]*sh[1]:,}</div></div>
            </div>""", unsafe_allow_html=True)
            prev = emb[:min(5, sh[0]), :20]; vmin, vmax = prev.min(), prev.max()
            cells = ""
            for row in prev:
                for val in row:
                    n = (val - vmin) / max(vmax - vmin, 1e-8)
                    bg = f"rgba(59,{int(130+n*52)},{int(246-n*34)},{.08+n*.2})"
                    cells += f'<div style="height:28px;border-radius:3px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:.58rem;font-family:\'DM Mono\',monospace;color:rgba(255,255,255,.3);">{val:.2f}</div>'
            st.markdown(f'<div style="display:grid;grid-template-columns:repeat(20,1fr);gap:2px;">{cells}</div>', unsafe_allow_html=True)
            if show_sim and len(emb) > 1:
                sim = R.get("similarity", np.array([]))
                if len(sim) > 1:
                    st.markdown('<div class="stitle c" style="margin-top:1rem;font-size:.73rem;">🔁 Top Similar Pairs</div>', unsafe_allow_html=True)
                    n2    = min(len(sents), len(sim))
                    pairs = sorted([(sim[i,j],i,j) for i in range(n2) for j in range(i+1,n2)], reverse=True)
                    for score, i, j in pairs[:4]:
                        col = "#10b981" if score > .8 else "#f59e0b" if score > .6 else "#64748b"
                        s1  = sents[i][:65] + "…" if len(sents[i]) > 65 else sents[i]
                        s2  = sents[j][:65] + "…" if len(sents[j]) > 65 else sents[j]
                        st.markdown(f'<div style="padding:.6rem .85rem;background:var(--card2);border:1px solid var(--border);border-radius:8px;margin-bottom:.35rem;"><div style="display:flex;justify-content:space-between;margin-bottom:.3rem;"><span style="color:var(--muted);font-size:.65rem;font-family:\'DM Mono\',monospace;">COSINE SIMILARITY</span><span style="color:{col};font-family:\'Syne\',sans-serif;font-size:.92rem;font-weight:800;">{score:.3f}</span></div><div style="font-size:.78rem;color:#94a3b8;">"{s1}"</div><div style="font-size:.7rem;color:var(--muted);margin:.18rem 0;">↔</div><div style="font-size:.78rem;color:#94a3b8;">"{s2}"</div></div>', unsafe_allow_html=True)
        else:
            st.info("No embeddings computed.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t6:
        st.markdown('<div class="scard"><div class="stitle r">📄 Full Transcript</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tbox">{R["transcript"].replace(chr(10),"<br>")}</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Transcript TXT", R["transcript"], "transcript.txt", "text/plain")
        with c2:
            if show_json:
                safe = {k: v for k, v in R.items() if k not in ["embeddings","similarity"]}
                st.download_button("⬇️ Full JSON", json.dumps(safe, indent=2, ensure_ascii=False), "analysis.json", "application/json")

    if show_json:
        with st.expander("Raw JSON"):
            safe = {k: v for k, v in R.items() if k not in ["embeddings","similarity"]}
            st.json(safe)


if __name__ == "__main__":
    main()
