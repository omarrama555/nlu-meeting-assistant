# ============================================================
#  NLP Intelligence Platform  —  app.py
#  Streamlit Cloud deployable — zero local model files
# ============================================================

import streamlit as st
import time, json, os, io, math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import importlib

st.set_page_config(
    page_title="NLP Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080c12;--s1:#0d1117;--s2:#111827;--s3:#1a2233;
  --b1:#1f2937;--b2:#374151;
  --blue:#3b82f6;--blue-d:rgba(59,130,246,0.12);--blue-g:rgba(59,130,246,0.25);
  --green:#10b981;--green-d:rgba(16,185,129,0.12);
  --amber:#f59e0b;--amber-d:rgba(245,158,11,0.12);
  --red:#ef4444;--red-d:rgba(239,68,68,0.12);
  --violet:#8b5cf6;--violet-d:rgba(139,92,246,0.12);
  --cyan:#06b6d4;--cyan-d:rgba(6,182,212,0.12);
  --t1:#f1f5f9;--t2:#94a3b8;--t3:#4b5563;--t4:#374151;
  --fs:'DM Sans',sans-serif;--fm:'DM Mono',monospace;
  --r:10px;--rs:6px;--rl:14px;
  --sh:0 1px 3px rgba(0,0,0,.4),0 4px 16px rgba(0,0,0,.3);
}
html,body,[class*="css"]{font-family:var(--fs)!important;background:var(--bg)!important;color:var(--t1)!important;font-size:14px;line-height:1.6}
.stApp{background:var(--bg)!important}
.main .block-container{padding:2rem 2.5rem 4rem!important;max-width:1400px}
[data-testid="stSidebar"]{background:var(--s1)!important;border-right:1px solid var(--b1)!important;padding-top:0!important}
[data-testid="stSidebar"]>div:first-child{padding-top:0!important}
[data-testid="stSidebar"] *{color:var(--t1)!important}
[data-testid="stSidebarNav"]{display:none!important}
textarea,input[type="text"],input[type="number"],.stTextArea textarea,.stTextInput input{background:var(--s2)!important;color:var(--t1)!important;border:1px solid var(--b1)!important;border-radius:var(--rs)!important;font-family:var(--fm)!important;font-size:13px!important;transition:border-color .15s}
textarea:focus,input:focus{border-color:var(--blue)!important;box-shadow:0 0 0 3px var(--blue-d)!important;outline:none!important}
label,.stTextArea label,.stTextInput label,.stSelectbox label,.stSlider label,.stFileUploader label{color:var(--t2)!important;font-size:12px!important;font-weight:500!important;text-transform:uppercase!important;letter-spacing:.06em!important;margin-bottom:6px!important}
.stButton>button{background:var(--blue)!important;color:#fff!important;border:none!important;border-radius:var(--rs)!important;font-family:var(--fs)!important;font-weight:600!important;font-size:13px!important;letter-spacing:.02em!important;padding:.55rem 1.4rem!important;cursor:pointer!important;transition:all .15s ease!important;box-shadow:0 1px 4px rgba(59,130,246,.3)!important}
.stButton>button:hover{background:#2563eb!important;transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(59,130,246,.4)!important}
.stDownloadButton>button{background:transparent!important;color:var(--blue)!important;border:1px solid var(--b2)!important;border-radius:var(--rs)!important;font-size:12px!important;font-weight:600!important;padding:.4rem 1rem!important}
.stDownloadButton>button:hover{border-color:var(--blue)!important;background:var(--blue-d)!important}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--b1)!important;gap:0!important;padding:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--t3)!important;border:none!important;border-bottom:2px solid transparent!important;border-radius:0!important;font-family:var(--fs)!important;font-weight:600!important;font-size:13px!important;letter-spacing:.03em!important;padding:.7rem 1.2rem!important;margin:0!important;transition:all .15s!important}
.stTabs [data-baseweb="tab"]:hover{color:var(--t1)!important}
.stTabs [aria-selected="true"]{color:var(--blue)!important;border-bottom-color:var(--blue)!important;background:var(--blue-d)!important}
.stTabs [data-baseweb="tab-panel"]{padding:1.5rem 0!important}
[data-testid="stExpander"]{background:var(--s2)!important;border:1px solid var(--b1)!important;border-radius:var(--r)!important;margin-bottom:.75rem;overflow:hidden}
[data-testid="stExpander"] summary{color:var(--t1)!important;font-weight:600!important;font-size:13px!important;padding:.85rem 1rem!important}
[data-testid="stExpander"] summary:hover{background:var(--s3)!important}
[data-testid="stExpander"]>div>div{padding:0 1rem 1rem!important}
[data-baseweb="select"]>div{background:var(--s2)!important;border-color:var(--b1)!important;border-radius:var(--rs)!important;color:var(--t1)!important}
[data-baseweb="popover"]{background:var(--s2)!important;border:1px solid var(--b1)!important}
[data-baseweb="menu"]{background:var(--s2)!important}
[role="option"]{color:var(--t1)!important}
[role="option"]:hover{background:var(--s3)!important}
[data-testid="metric-container"]{background:var(--s2);border:1px solid var(--b1);border-radius:var(--r);padding:1.1rem 1.25rem!important;box-shadow:var(--sh)}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{color:var(--t2)!important;font-size:11px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--t1)!important;font-size:1.6rem!important;font-weight:700!important;letter-spacing:-.02em!important}
[data-testid="stFileUploader"]{background:var(--s2)!important;border:1px dashed var(--b2)!important;border-radius:var(--r)!important}
.stAlert{border-radius:var(--r)!important;border:none!important;font-size:13px!important}
.stProgress>div>div{background:var(--b1)!important;border-radius:4px!important}
.stProgress>div>div>div{background:var(--blue)!important;border-radius:4px!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--b2);border-radius:4px}
[data-testid="stRadio"] label{color:var(--t1)!important;font-size:13px!important;text-transform:none!important;letter-spacing:0!important;font-weight:400!important}

.card{background:var(--s2);border:1px solid var(--b1);border-radius:var(--r);padding:1.25rem 1.5rem;box-shadow:var(--sh);margin-bottom:1rem}
.card-b{border-left:3px solid var(--blue)}
.card-g{border-left:3px solid var(--green)}
.card-a{border-left:3px solid var(--amber)}
.card-v{border-left:3px solid var(--violet)}
.card-c{border-left:3px solid var(--cyan)}
.card-r{border-left:3px solid var(--red)}

.ptitle{font-size:1.7rem;font-weight:700;color:var(--t1);letter-spacing:-.025em;line-height:1.2}
.psub{font-size:13px;color:var(--t2);margin-top:4px;line-height:1.5}
.pheader{display:flex;align-items:flex-start;justify-content:space-between;padding-bottom:1.5rem;border-bottom:1px solid var(--b1);margin-bottom:2rem}

.slbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--t3);margin-bottom:.75rem}

.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:99px;font-family:var(--fm);font-size:11px;font-weight:500;letter-spacing:.02em}
.bb{background:var(--blue-d);color:var(--blue);border:1px solid rgba(59,130,246,.2)}
.bg{background:var(--green-d);color:var(--green);border:1px solid rgba(16,185,129,.2)}
.ba{background:var(--amber-d);color:var(--amber);border:1px solid rgba(245,158,11,.2)}
.bv{background:var(--violet-d);color:var(--violet);border:1px solid rgba(139,92,246,.2)}
.bc{background:var(--cyan-d);color:var(--cyan);border:1px solid rgba(6,182,212,.2)}
.br{background:var(--red-d);color:var(--red);border:1px solid rgba(239,68,68,.2)}
.bgr{background:var(--s3);color:var(--t2);border:1px solid var(--b2)}

.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.dg{background:var(--green);box-shadow:0 0 6px var(--green)}
.da{background:var(--amber)}
.dr{background:var(--red)}
.db{background:var(--blue);box-shadow:0 0 6px var(--blue)}

.sc{background:var(--s2);border:1px solid var(--b1);border-radius:var(--r);padding:1.1rem 1.25rem;position:relative;overflow:hidden}
.slb{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--t3);margin-bottom:6px}
.sv{font-size:1.65rem;font-weight:700;color:var(--t1);letter-spacing:-.03em;line-height:1}
.ss{font-size:11px;color:var(--t3);margin-top:5px}
.sa{position:absolute;top:0;right:0;width:3px;height:100%;border-radius:0 var(--r) var(--r) 0}

.ent{display:inline;padding:1px 6px 2px;border-radius:4px;font-size:13px;line-height:2;cursor:default}
.ePER{background:rgba(59,130,246,.18);color:#93c5fd;border-bottom:2px solid #3b82f6}
.eORG{background:rgba(16,185,129,.18);color:#6ee7b7;border-bottom:2px solid #10b981}
.eLOC{background:rgba(245,158,11,.18);color:#fcd34d;border-bottom:2px solid #f59e0b}
.eMISC{background:rgba(139,92,246,.18);color:#c4b5fd;border-bottom:2px solid #8b5cf6}
.eDEF{background:rgba(148,163,184,.12);color:#94a3b8;border-bottom:2px solid #475569}

.mono{background:var(--bg);border:1px solid var(--b1);border-radius:var(--rs);padding:1rem 1.1rem;font-family:var(--fm);font-size:12.5px;color:var(--t2);white-space:pre-wrap;word-break:break-word;max-height:320px;overflow-y:auto;line-height:1.7}

.pstep{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--s2);border:1px solid var(--b1);border-radius:var(--rs);margin-bottom:4px}
.pstep.done{border-color:var(--green);background:var(--green-d)}
.pstep.active{border-color:var(--blue);background:var(--blue-d)}
.psn{width:24px;height:24px;border-radius:50%;background:var(--s3);border:1px solid var(--b2);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;font-family:var(--fm);color:var(--t2);flex-shrink:0}
.pstep.done .psn{background:var(--green-d);color:var(--green);border-color:var(--green)}
.pstep.active .psn{background:var(--blue-d);color:var(--blue);border-color:var(--blue)}
.psl{font-size:13px;font-weight:600;color:var(--t1)}
.pss{font-size:11px;color:var(--t3);font-family:var(--fm)}
.pst{margin-left:auto;font-family:var(--fm);font-size:11px;color:var(--t3)}

.sep{border:none;border-top:1px solid var(--b1);margin:1.5rem 0}

.cw{margin:5px 0}
.cl{display:flex;justify-content:space-between;margin-bottom:3px}
.cl span:first-child{font-size:12px;color:var(--t2)}
.cl span:last-child{font-size:11px;font-family:var(--fm);color:var(--t3)}
.ct{height:5px;background:var(--b1);border-radius:4px;overflow:hidden}
.cf{height:5px;border-radius:4px;transition:width .4s ease}

.sbbrand{padding:1.5rem 1.25rem 1rem;border-bottom:1px solid var(--b1);margin-bottom:.5rem}
.sbbname{font-size:15px;font-weight:700;color:var(--t1);letter-spacing:-.01em}
.sbbtag{font-size:10px;font-weight:500;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin-top:2px}
.sbns{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--t4);padding:1rem 1.25rem .35rem}
.sbni{display:flex;align-items:center;gap:10px;padding:8px 1.25rem;cursor:pointer;font-size:13px;font-weight:500;color:var(--t2);transition:all .12s;border-left:2px solid transparent;margin:1px 0}
.sbni:hover{background:var(--s2);color:var(--t1)}
.sbni.act{background:var(--blue-d);color:var(--blue);border-left-color:var(--blue);font-weight:600}

.ttag{display:inline-flex;align-items:center;gap:4px;background:var(--s3);border:1px solid var(--b2);color:var(--t3);font-family:var(--fm);font-size:11px;padding:2px 8px;border-radius:4px}

.rb{background:var(--s1);border:1px solid var(--b1);border-radius:var(--r);padding:1.25rem}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ── Session state defaults ──────────────────────────────────
_D = {
    "page": "Home",
    "transcript": "",
    "transcript_chunks": [],
    "transcript_time": 0.0,
    "ner_results": [],
    "ner_text": "",
    "summary": "",
    "summary_original": "",
    "summary_time": 0.0,
    "intents": [],
    "intent_text": "",
    "actions_output": "",
    "pipe_state": {},
    "inference_log": [],
    "models_loaded": set(),
    "session_start": datetime.now().isoformat(),
}
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Device ──────────────────────────────────────────────────
@st.cache_resource
def detect_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = detect_device()

# ── Model loaders ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_whisper():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-small",
                    device=0 if DEVICE == "cuda" else -1, return_timestamps=True)

@st.cache_resource(show_spinner=False)
def get_intent_clf():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                    device=0 if DEVICE == "cuda" else -1)

@st.cache_resource(show_spinner=False)
def get_ner():
    from transformers import pipeline
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple",
                    device=0 if DEVICE == "cuda" else -1)

@st.cache_resource(show_spinner=False)
def get_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn",
                    device=0 if DEVICE == "cuda" else -1)

@st.cache_resource(show_spinner=False)
def get_flan():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base",
                    device=0 if DEVICE == "cuda" else -1)

@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Helpers ─────────────────────────────────────────────────
def timed(fn, *a, **kw):
    t = time.time(); r = fn(*a, **kw); return r, round(time.time()-t, 2)

def log_inf(model, secs, tokens=0):
    st.session_state["inference_log"].append(
        {"model": model, "time": secs, "tokens": tokens, "ts": datetime.now().strftime("%H:%M:%S")})
    st.session_state["models_loaded"].add(model)

def badge(txt, c="b"):
    cls = {"b":"bb","g":"bg","a":"ba","v":"bv","c":"bc","r":"br","gr":"bgr"}.get(c,"bgr")
    return f'<span class="badge {cls}">{txt}</span>'

def ttag(s):
    return f'<span class="ttag">{s}s</span>'

def conf_bar(label, score, c="blue"):
    p = int(score*100)
    return f"""<div class="cw"><div class="cl"><span>{label}</span><span>{p}%</span></div>
    <div class="ct"><div class="cf" style="width:{p}%;background:var(--{c})"></div></div></div>"""

def page_header(title, subtitle, badge_html=""):
    st.markdown(f"""<div class="pheader">
      <div><div class="ptitle">{title}</div><div class="psub">{subtitle}</div></div>
      <div>{badge_html}</div></div>""", unsafe_allow_html=True)

def dl_json(data, fname):
    st.download_button("Export JSON", json.dumps(data, indent=2, ensure_ascii=False),
                       file_name=fname, mime="application/json")

def dp(fig, h=320):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Mono", font_color="#94a3b8",
        margin=dict(l=0,r=0,t=24,b=0), height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
        xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937", tickfont_color="#4b5563"),
        yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937", tickfont_color="#4b5563"))
    return fig

# ── Sidebar nav ─────────────────────────────────────────────
NAV = {
    "Overview":        [("Home","Dashboard")],
    "Pipeline Stages": [
        ("Speech To Text","Whisper STT"),
        ("Intent Classification","BART Zero-Shot"),
        ("Named Entity Recognition","BERT NER"),
        ("Summarization","BART CNN"),
        ("Action Extraction","Flan-T5"),
        ("Embeddings","MiniLM"),
    ],
    "Integrated":  [("Full Pipeline","End-to-End")],
    "Intelligence":[("Analytics","Metrics"),("System Monitor","Env & Resources")],
}

with st.sidebar:
    st.markdown("""<div class="sbbrand">
      <div class="sbbname">NLP Platform</div>
      <div class="sbbtag">Intelligence Suite v2.0</div></div>""", unsafe_allow_html=True)

    for section, items in NAV.items():
        st.markdown(f'<div class="sbns">{section}</div>', unsafe_allow_html=True)
        for pk, ps in items:
            is_active = st.session_state["page"] == pk
            if st.button(pk, key=f"nav_{pk}", use_container_width=True):
                st.session_state["page"] = pk
                st.rerun()

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    dev_html = (badge('<span class="dot dg"></span> GPU Active', "g")
                if DEVICE == "cuda" else badge('<span class="dot da"></span> CPU Mode', "a"))
    st.markdown(f"<div style='padding:0 1.25rem .5rem'>{dev_html}</div>", unsafe_allow_html=True)
    n = len(st.session_state["models_loaded"])
    st.markdown(f"<div style='padding:0 1.25rem 1rem;font-size:11px;color:var(--t3)'>Cached: <b style='color:var(--t2)'>{n}/6</b></div>", unsafe_allow_html=True)

PAGE = st.session_state["page"]


# ════════════════════════════════════════════════════════════
#  HOME DASHBOARD
# ════════════════════════════════════════════════════════════
if PAGE == "Home":
    page_header("NLP Intelligence Platform",
        "A modular AI pipeline combining six transformer models into one unified NLU workflow.",
        badge("v2.0","b"))

    logs = st.session_state["inference_log"]
    avg_t = round(sum(l["time"] for l in logs)/max(len(logs),1),2) if logs else 0.0
    loaded = st.session_state["models_loaded"]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def scard(col, lbl, val, sub, ac):
        with col:
            st.markdown(f"""<div class="sc">
              <div class="sa" style="background:var(--{ac})"></div>
              <div class="slb">{lbl}</div><div class="sv">{val}</div>
              <div class="ss">{sub}</div></div>""", unsafe_allow_html=True)
    scard(c1,"Models Loaded",f"{len(loaded)}/6","Cached in session","blue")
    scard(c2,"Avg Inference",f"{avg_t}s","Per model call","green")
    scard(c3,"Device",DEVICE.upper(),"Torch backend","amber" if DEVICE=="cpu" else "green")
    scard(c4,"Total Calls",str(len(logs)),"This session","violet")
    scard(c5,"Pipeline Runs",str(len([v for v in st.session_state["pipe_state"].values() if v])),"Full runs","cyan")
    scard(c6,"Runtime","Active","Streamlit Cloud","green")

    st.markdown("<br>", unsafe_allow_html=True)
    left,right = st.columns([3,2])

    with left:
        st.markdown('<div class="slbl">Pipeline Architecture</div>', unsafe_allow_html=True)
        steps = [
            ("1","Audio Input","WAV / MP3 / M4A upload"),
            ("2","Whisper STT","openai/whisper-small"),
            ("3","Intent Classification","facebook/bart-large-mnli"),
            ("4","Named Entity Recognition","dslim/bert-base-NER"),
            ("5","Summarization","facebook/bart-large-cnn"),
            ("6","Action Extraction","google/flan-t5-base"),
            ("7","Embeddings & Retrieval","all-MiniLM-L6-v2"),
        ]
        for num,lbl,sub in steps:
            is_done = any(sub in m or lbl.split()[0].lower() in m.lower() for m in loaded)
            cls="pstep done" if is_done else "pstep"
            dot='<span class="dot dg"></span>' if is_done else '<span class="dot da"></span>'
            st.markdown(f"""<div class="{cls}">
              <div class="psn">{num}</div>
              <div><div class="psl">{dot} {lbl}</div><div class="pss">{sub}</div></div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="slbl">Model Registry</div>', unsafe_allow_html=True)
        models_info=[
            ("Whisper","openai/whisper-small","STT","b"),
            ("BART","facebook/bart-large-mnli","Intent","g"),
            ("BERT","dslim/bert-base-NER","NER","a"),
            ("BART-CNN","facebook/bart-large-cnn","Summ","v"),
            ("Flan-T5","google/flan-t5-base","Action","c"),
            ("MiniLM","all-MiniLM-L6-v2","Embed","r"),
        ]
        for short,full,task,color in models_info:
            is_loaded=any(full in m or short in m for m in loaded)
            sb=badge("Cached","g") if is_loaded else badge("On-demand","gr")
            st.markdown(f"""<div class="card" style="padding:.8rem 1rem;margin-bottom:.5rem">
              <div style="display:flex;align-items:center;justify-content:space-between">
                <div>{badge(task,color)} <span style="font-size:12px;font-weight:600;color:var(--t1)">{short}</span></div>
                {sb}</div>
              <div style="font-size:11px;color:var(--t3);font-family:var(--fm);margin-top:4px">{full}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1,t2=st.tabs(["Quick Start Guide","About This Platform"])
    with t1:
        cc=st.columns(3)
        guides=[
            ("Step 1: Upload Audio","Go to Speech To Text. Upload a WAV or MP3 file. Whisper will auto-transcribe it.","b"),
            ("Step 2: Run Pipeline","Use Full Pipeline to chain all 6 models in sequence on your text.","g"),
            ("Step 3: Export Results","Download JSON or CSV from any individual stage page.","a"),
        ]
        for col,(title,body,color) in zip(cc,guides):
            with col:
                st.markdown(f"""<div class="card card-{color}">
                  <div style="font-size:13px;font-weight:700;color:var(--t1);margin-bottom:6px">{title}</div>
                  <div style="font-size:12px;color:var(--t2);line-height:1.6">{body}</div></div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""<div class="card"><div style="font-size:13px;color:var(--t2);line-height:1.8">
          This platform combines six state-of-the-art Hugging Face transformer models into a unified NLP pipeline.
          All models download automatically on first use and are cached for the remainder of the session.
          No local model files required. Deployable to Streamlit Cloud with a single click.
          Automatic GPU acceleration when available, CPU fallback otherwise.</div></div>""", unsafe_allow_html=True)
        with st.expander("All Features"):
            feats=["Audio transcription with language detection (Whisper)","Zero-shot intent classification with custom labels (BART)",
                   "Named entity recognition with color-coded highlighting (BERT NER)","Long-form and meeting summarization with analytics (BART-CNN)",
                   "Action item extraction and workflow generation (Flan-T5)","Semantic similarity, retrieval, and embedding visualization (MiniLM)",
                   "Full integrated end-to-end pipeline execution","Session analytics and inference timing dashboard","System monitor with CPU, RAM, and environment details"]
            for f in feats:
                st.markdown(f'<div style="font-size:12px;color:var(--t2);padding:4px 0;border-bottom:1px solid var(--b1)">• {f}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SPEECH TO TEXT
# ════════════════════════════════════════════════════════════
elif PAGE == "Speech To Text":
    page_header("Speech To Text","Converts spoken audio into text using OpenAI Whisper STT.",badge("openai/whisper-small","b"))
    st.markdown("""<div class="card card-b" style="margin-bottom:1.5rem"><div style="font-size:12px;color:var(--t2);line-height:1.7">
      This pipeline uses OpenAI Whisper to perform automatic speech recognition. Audio is resampled to 16kHz mono
      and processed by a 244M parameter encoder-decoder transformer. Language is detected automatically.
      Use <b>translate</b> mode to convert non-English audio to English text.</div></div>""", unsafe_allow_html=True)

    left,right = st.columns([1,1],gap="large")
    with left:
        st.markdown('<div class="slbl">Audio Input</div>', unsafe_allow_html=True)
        afile = st.file_uploader("Upload audio file",type=["mp3","wav","m4a","ogg","flac"],key="stt_up")
        if afile:
            abytes = afile.read()
            st.audio(abytes)
            sz = round(len(abytes)/1024,1)
            st.markdown(f"""<div class="card" style="padding:.75rem 1rem;margin-top:.75rem">
              <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
                <div><div class="slb">Filename</div><div style="font-size:12px;color:var(--t2);font-family:var(--fm)">{afile.name}</div></div>
                <div><div class="slb">Size</div><div style="font-size:12px;color:var(--t2);font-family:var(--fm)">{sz} KB</div></div>
                <div><div class="slb">Type</div><div style="font-size:12px;color:var(--t2);font-family:var(--fm)">{afile.type}</div></div>
              </div></div>""", unsafe_allow_html=True)
        task_mode = st.radio("Task mode",["transcribe","translate"],horizontal=True)
        run_btn = st.button("Run Transcription", disabled=afile is None)
        if run_btn and afile:
            tmp=f"/tmp/stt_{int(time.time())}.audio"
            with open(tmp,"wb") as f: f.write(abytes)
            with st.spinner("Loading Whisper model..."):
                asr=get_whisper()
            prog=st.progress(0,"Transcribing...")
            t0=time.time()
            res=asr(tmp,generate_kwargs={"task":task_mode})
            elapsed=round(time.time()-t0,2)
            prog.progress(1.0,"Complete")
            st.session_state["transcript"]=res["text"]
            st.session_state["transcript_chunks"]=res.get("chunks",[])
            st.session_state["transcript_time"]=elapsed
            log_inf("Whisper STT",elapsed,len(res["text"].split()))
            st.rerun()

    with right:
        st.markdown('<div class="slbl">Transcription Output</div>', unsafe_allow_html=True)
        if st.session_state["transcript"]:
            txt=st.session_state["transcript"]
            elapsed=st.session_state["transcript_time"]
            chunks=st.session_state["transcript_chunks"]
            st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:.75rem">
              {badge('<span class="dot dg"></span> Complete',"g")} {ttag(elapsed)}
              {badge(f'{len(txt.split())} words',"gr")}</div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="mono">{txt}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            e1,e2 = st.columns(2)
            with e1: st.download_button("Export TXT",txt,"transcript.txt","text/plain")
            with e2: dl_json({"transcript":txt,"chunks":chunks,"inference_time":elapsed},"transcript.json")
            if chunks:
                st.markdown('<div class="slbl" style="margin-top:1rem">Timestamped Segments</div>', unsafe_allow_html=True)
                rows=[{"start":c.get("timestamp",[None,None])[0],"end":c.get("timestamp",[None,None])[1],"text":c.get("text","")} for c in chunks]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        else:
            st.markdown("""<div class="rb" style="display:flex;align-items:center;justify-content:center;min-height:200px">
              <div style="text-align:center"><div style="font-size:13px;color:var(--t3)">Upload audio and run transcription</div>
              <div style="font-size:11px;color:var(--t4);margin-top:4px">Results appear here</div></div></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION
# ════════════════════════════════════════════════════════════
elif PAGE == "Intent Classification":
    page_header("Intent Classification","Zero-shot intent detection without task-specific fine-tuning.",badge("facebook/bart-large-mnli","g"))
    st.markdown("""<div class="card card-g" style="margin-bottom:1.5rem"><div style="font-size:12px;color:var(--t2);line-height:1.7">
      BART trained on Multi-Genre NLI computes entailment scores between the input text and each candidate label.
      No training data required. Supports single-class, multi-label, and fully custom label sets.</div></div>""", unsafe_allow_html=True)

    DEFAULT_LABELS=["schedule meeting","send email","create task","approve document",
                    "request update","cancel appointment","provide feedback","escalate issue",
                    "assign responsibility","set deadline"]
    tabs=st.tabs(["Intent Prediction","Multi-Intent Ranking","Custom Label Classification"])

    with tabs[0]:
        tin=st.text_area("Input text",value=st.session_state.get("intent_text",""),
                          placeholder="Enter a sentence to classify...",height=100,key="ii1")
        if st.button("Predict Intent",key="bi1"):
            if tin.strip():
                with st.spinner("Loading BART classifier..."):
                    clf=get_intent_clf()
                res,elapsed=timed(clf,tin,candidate_labels=DEFAULT_LABELS[:5],multi_label=False)
                st.session_state["intents"]=list(zip(res["labels"],res["scores"]))
                st.session_state["intent_text"]=tin
                log_inf("BART Intent",elapsed,len(tin.split()))
                tl,ts=res["labels"][0],res["scores"][0]
                st.markdown("<br>",unsafe_allow_html=True)
                c1,c2,c3=st.columns(3)
                c1.metric("Top Intent",tl)
                c2.metric("Confidence",f"{round(ts*100,1)}%")
                c3.metric("Inference Time",f"{elapsed}s")
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown('<div class="slbl">Confidence Breakdown</div>',unsafe_allow_html=True)
                for lbl,score in zip(res["labels"],res["scores"]):
                    st.markdown(conf_bar(lbl,score,"blue"),unsafe_allow_html=True)
            else:
                st.warning("Enter text to classify.")

    with tabs[1]:
        tin2=st.text_area("Input text",value=st.session_state.get("intent_text",""),
                           placeholder="Enter a sentence...",height=100,key="ii2")
        topk=st.slider("Top intents",2,10,6,key="topk")
        if st.button("Rank Intents",key="bi2"):
            if tin2.strip():
                with st.spinner("Loading BART classifier..."):
                    clf=get_intent_clf()
                res,elapsed=timed(clf,tin2,candidate_labels=DEFAULT_LABELS,multi_label=True)
                log_inf("BART Intent",elapsed,len(tin2.split()))
                df=pd.DataFrame({"Intent":res["labels"][:topk],"Score":[round(s,4) for s in res["scores"][:topk]],
                                  "Pct":[round(s*100,1) for s in res["scores"][:topk]]})
                st.markdown(f'<div style="margin-bottom:.75rem">{ttag(elapsed)}</div>',unsafe_allow_html=True)
                fig=go.Figure(go.Bar(x=df["Pct"],y=df["Intent"],orientation="h",
                    marker_color="#3b82f6",marker_line_width=0,
                    text=[f"{v}%" for v in df["Pct"]],textposition="outside",
                    textfont=dict(size=10,color="#94a3b8")))
                dp(fig,280); fig.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=11)))
                st.plotly_chart(fig,use_container_width=True)
                st.dataframe(df[["Intent","Score","Pct"]],use_container_width=True,hide_index=True)
            else:
                st.warning("Enter text first.")

    with tabs[2]:
        tin3=st.text_area("Input text",placeholder="Describe what you want to do...",height=100,key="ii3")
        craw=st.text_input("Custom labels (comma-separated)",
            value="approve budget, reject proposal, request clarification, defer decision, escalate to manager",key="cl")
        if st.button("Classify",key="bi3"):
            lbls=[l.strip() for l in craw.split(",") if l.strip()]
            if tin3.strip() and lbls:
                with st.spinner("Classifying..."):
                    clf=get_intent_clf()
                res,elapsed=timed(clf,tin3,candidate_labels=lbls,multi_label=True)
                log_inf("BART Intent",elapsed)
                df=pd.DataFrame({"Label":res["labels"],"Score":[round(s,4) for s in res["scores"]]})
                c1,c2=st.columns([2,1])
                with c1:
                    fig=px.bar(df,x="Label",y="Score",color="Score",
                               color_continuous_scale=["#1f2937","#3b82f6"],template="plotly_dark")
                    dp(fig,260); fig.update_coloraxes(showscale=False)
                    st.plotly_chart(fig,use_container_width=True)
                with c2:
                    st.dataframe(df,use_container_width=True,hide_index=True)
                    dl_json({"text":tin3,"labels":res["labels"],"scores":res["scores"]},"intent_results.json")
            else:
                st.warning("Provide text and at least one label.")
