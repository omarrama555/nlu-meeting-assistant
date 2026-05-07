# ============================================================
#  Meeting NLU Intelligence Platform — app.py
#  Streamlit Cloud deployable — robust lazy loading
# ============================================================

import streamlit as st
import time, json, os, io, re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Meeting NLU Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Typography & Theme ──────────────────────────────────────
THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#05080f;
  --s1:#080d18;
  --s2:#0d1424;
  --s3:#121a2e;
  --b1:#1a2440;
  --b2:#243258;

  --blue:#4f8ef7;
  --blue-d:rgba(79,142,247,0.10);
  --blue-g:rgba(79,142,247,0.20);
  --blue-b:rgba(79,142,247,0.06);

  --teal:#00d4aa;
  --teal-d:rgba(0,212,170,0.10);
  --amber:#ffb347;
  --amber-d:rgba(255,179,71,0.10);
  --red:#ff5e7d;
  --red-d:rgba(255,94,125,0.10);
  --violet:#a78bfa;
  --violet-d:rgba(167,139,250,0.10);
  --rose:#f472b6;
  --rose-d:rgba(244,114,182,0.10);

  --t1:#e8edf8;
  --t2:#8899bb;
  --t3:#445577;
  --t4:#2a3654;

  --fs:'Inter',sans-serif;
  --fd:'Syne',sans-serif;
  --fm:'JetBrains Mono',monospace;

  --r:12px;--rs:8px;--rl:16px;
  --sh:0 2px 8px rgba(0,0,0,.5),0 8px 32px rgba(0,0,0,.3);
  --glow:0 0 20px rgba(79,142,247,0.15);
}

html,body,[class*="css"]{
  font-family:var(--fs)!important;
  background:var(--bg)!important;
  color:var(--t1)!important;
  font-size:14px;
  line-height:1.6;
}

.stApp{background:var(--bg)!important}
.main .block-container{padding:2rem 2.5rem 4rem!important;max-width:1440px}

/* Sidebar */
[data-testid="stSidebar"]{
  background:var(--s1)!important;
  border-right:1px solid var(--b1)!important;
  padding-top:0!important;
}
[data-testid="stSidebar"]>div:first-child{padding-top:0!important}
[data-testid="stSidebar"] *{color:var(--t1)!important}
[data-testid="stSidebarNav"]{display:none!important}

/* Inputs */
textarea,input[type="text"],input[type="number"],
.stTextArea textarea,.stTextInput input{
  background:var(--s2)!important;
  color:var(--t1)!important;
  border:1px solid var(--b1)!important;
  border-radius:var(--rs)!important;
  font-family:var(--fm)!important;
  font-size:13px!important;
  transition:border-color .2s,box-shadow .2s;
}
textarea:focus,input:focus{
  border-color:var(--blue)!important;
  box-shadow:0 0 0 3px var(--blue-d)!important;
  outline:none!important;
}

/* Labels */
label,.stTextArea label,.stTextInput label,
.stSelectbox label,.stSlider label,.stFileUploader label{
  color:var(--t2)!important;
  font-size:11px!important;
  font-weight:600!important;
  text-transform:uppercase!important;
  letter-spacing:.08em!important;
  margin-bottom:6px!important;
  font-family:var(--fd)!important;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,var(--blue),#3b6fd4)!important;
  color:#fff!important;border:none!important;
  border-radius:var(--rs)!important;
  font-family:var(--fd)!important;
  font-weight:700!important;font-size:13px!important;
  letter-spacing:.04em!important;
  padding:.6rem 1.6rem!important;
  cursor:pointer!important;
  transition:all .2s ease!important;
  box-shadow:0 2px 12px rgba(79,142,247,.3)!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 6px 24px rgba(79,142,247,.5)!important;
}
.stButton>button:active{transform:translateY(0)!important}

.stDownloadButton>button{
  background:transparent!important;
  color:var(--blue)!important;
  border:1px solid var(--b2)!important;
  border-radius:var(--rs)!important;
  font-size:12px!important;font-weight:600!important;
  padding:.4rem 1rem!important;
  transition:all .15s!important;
}
.stDownloadButton>button:hover{
  border-color:var(--blue)!important;
  background:var(--blue-d)!important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  background:transparent!important;
  border-bottom:1px solid var(--b1)!important;
  gap:0!important;padding:0!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;
  color:var(--t3)!important;
  border:none!important;
  border-bottom:2px solid transparent!important;
  border-radius:0!important;
  font-family:var(--fd)!important;
  font-weight:700!important;font-size:12px!important;
  letter-spacing:.05em!important;
  padding:.75rem 1.3rem!important;
  margin:0!important;
  transition:all .15s!important;
  text-transform:uppercase!important;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--t1)!important}
.stTabs [aria-selected="true"]{
  color:var(--blue)!important;
  border-bottom-color:var(--blue)!important;
  background:var(--blue-b)!important;
}
.stTabs [data-baseweb="tab-panel"]{padding:1.75rem 0!important}

/* Expanders */
[data-testid="stExpander"]{
  background:var(--s2)!important;
  border:1px solid var(--b1)!important;
  border-radius:var(--r)!important;
  margin-bottom:.75rem;overflow:hidden;
}
[data-testid="stExpander"] summary{
  color:var(--t1)!important;
  font-weight:700!important;font-size:13px!important;
  padding:.9rem 1.1rem!important;
  font-family:var(--fd)!important;
}
[data-testid="stExpander"] summary:hover{background:var(--s3)!important}
[data-testid="stExpander"]>div>div{padding:0 1.1rem 1.1rem!important}

/* Selects */
[data-baseweb="select"]>div{
  background:var(--s2)!important;
  border-color:var(--b1)!important;
  border-radius:var(--rs)!important;
  color:var(--t1)!important;
}
[data-baseweb="popover"]{background:var(--s2)!important;border:1px solid var(--b1)!important}
[data-baseweb="menu"]{background:var(--s2)!important}
[role="option"]{color:var(--t1)!important}
[role="option"]:hover{background:var(--s3)!important}

/* Metrics */
[data-testid="metric-container"]{
  background:var(--s2);
  border:1px solid var(--b1);
  border-radius:var(--r);
  padding:1.1rem 1.3rem!important;
  box-shadow:var(--sh);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{
  color:var(--t2)!important;font-size:10px!important;
  font-weight:700!important;text-transform:uppercase!important;
  letter-spacing:.1em!important;font-family:var(--fd)!important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color:var(--t1)!important;font-size:1.7rem!important;
  font-weight:800!important;letter-spacing:-.03em!important;
  font-family:var(--fd)!important;
}

/* File uploader */
[data-testid="stFileUploader"]{
  background:var(--s2)!important;
  border:1px dashed var(--b2)!important;
  border-radius:var(--r)!important;
  transition:border-color .2s!important;
}
[data-testid="stFileUploader"]:hover{border-color:var(--blue)!important}

/* Alerts */
.stAlert{border-radius:var(--r)!important;border:none!important;font-size:13px!important}

/* Progress */
.stProgress>div>div{background:var(--b1)!important;border-radius:4px!important}
.stProgress>div>div>div{
  background:linear-gradient(90deg,var(--blue),var(--teal))!important;
  border-radius:4px!important;
}

/* Scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--b2);border-radius:4px}

/* Radio */
[data-testid="stRadio"] label{
  color:var(--t1)!important;font-size:13px!important;
  text-transform:none!important;letter-spacing:0!important;font-weight:400!important;
}

/* Dataframe */
[data-testid="stDataFrame"]{
  border-radius:var(--r)!important;
  overflow:hidden!important;
  border:1px solid var(--b1)!important;
}

/* ── Custom Components ── */

.card{
  background:var(--s2);border:1px solid var(--b1);
  border-radius:var(--r);padding:1.3rem 1.5rem;
  box-shadow:var(--sh);margin-bottom:1rem;
}
.card-b{border-left:3px solid var(--blue)}
.card-t{border-left:3px solid var(--teal)}
.card-a{border-left:3px solid var(--amber)}
.card-v{border-left:3px solid var(--violet)}
.card-r{border-left:3px solid var(--red)}

/* Page header */
.pheader{
  display:flex;align-items:flex-start;
  justify-content:space-between;
  padding-bottom:1.75rem;
  border-bottom:1px solid var(--b1);
  margin-bottom:2rem;
}
.ptitle{
  font-size:1.9rem;font-weight:800;
  color:var(--t1);letter-spacing:-.03em;
  line-height:1.15;font-family:var(--fd);
}
.psub{font-size:13px;color:var(--t2);margin-top:6px;line-height:1.6}

/* Section label */
.slbl{
  font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.12em;
  color:var(--t3);margin-bottom:.85rem;
  font-family:var(--fd);
}

/* Badges */
.badge{
  display:inline-flex;align-items:center;gap:5px;
  padding:3px 10px;border-radius:99px;
  font-family:var(--fm);font-size:11px;
  font-weight:500;letter-spacing:.02em;
}
.bb{background:var(--blue-d);color:var(--blue);border:1px solid rgba(79,142,247,.2)}
.bt{background:var(--teal-d);color:var(--teal);border:1px solid rgba(0,212,170,.2)}
.ba{background:var(--amber-d);color:var(--amber);border:1px solid rgba(255,179,71,.2)}
.bv{background:var(--violet-d);color:var(--violet);border:1px solid rgba(167,139,250,.2)}
.br{background:var(--red-d);color:var(--red);border:1px solid rgba(255,94,125,.2)}
.bgr{background:var(--s3);color:var(--t2);border:1px solid var(--b2)}

/* Dots */
.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.dg{background:var(--teal);box-shadow:0 0 6px var(--teal)}
.da{background:var(--amber)}
.dr{background:var(--red)}
.db{background:var(--blue);box-shadow:0 0 8px var(--blue)}

/* Stat card */
.sc{
  background:var(--s2);border:1px solid var(--b1);
  border-radius:var(--r);padding:1.1rem 1.3rem;
  position:relative;overflow:hidden;
  transition:border-color .2s,box-shadow .2s;
}
.sc:hover{border-color:var(--b2);box-shadow:var(--glow)}
.sa{position:absolute;top:0;right:0;width:3px;height:100%;border-radius:0 var(--r) var(--r) 0}
.slb{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--t3);margin-bottom:8px;font-family:var(--fd)}
.sv{font-size:1.65rem;font-weight:800;color:var(--t1);letter-spacing:-.04em;line-height:1;font-family:var(--fd)}
.ss{font-size:11px;color:var(--t3);margin-top:6px}

/* NER highlighting */
.ent{display:inline;padding:1px 6px 2px;border-radius:5px;font-size:13px;line-height:2.2;cursor:default}
.ePER{background:rgba(79,142,247,.18);color:#93c5fd;border-bottom:2px solid var(--blue)}
.eORG{background:rgba(0,212,170,.18);color:#5eead4;border-bottom:2px solid var(--teal)}
.eLOC{background:rgba(255,179,71,.18);color:#fcd34d;border-bottom:2px solid var(--amber)}
.eMISC{background:rgba(167,139,250,.18);color:#c4b5fd;border-bottom:2px solid var(--violet)}
.eDEF{background:rgba(148,163,184,.10);color:#94a3b8;border-bottom:2px solid #475569}

/* Mono box */
.mono{
  background:var(--bg);border:1px solid var(--b1);
  border-radius:var(--rs);padding:1rem 1.2rem;
  font-family:var(--fm);font-size:12.5px;
  color:var(--t2);white-space:pre-wrap;
  word-break:break-word;max-height:340px;
  overflow-y:auto;line-height:1.75;
}

/* Pipeline steps */
.pstep{
  display:flex;align-items:center;gap:12px;
  padding:10px 14px;background:var(--s2);
  border:1px solid var(--b1);border-radius:var(--rs);
  margin-bottom:5px;transition:border-color .15s;
}
.pstep.done{border-color:var(--teal);background:var(--teal-d)}
.pstep.active{border-color:var(--blue);background:var(--blue-d)}
.psn{
  width:26px;height:26px;border-radius:50%;
  background:var(--s3);border:1px solid var(--b2);
  display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;font-family:var(--fm);
  color:var(--t2);flex-shrink:0;
}
.pstep.done .psn{background:var(--teal-d);color:var(--teal);border-color:var(--teal)}
.pstep.active .psn{background:var(--blue-d);color:var(--blue);border-color:var(--blue)}
.psl{font-size:13px;font-weight:700;color:var(--t1);font-family:var(--fd)}
.pss{font-size:11px;color:var(--t3);font-family:var(--fm)}

/* Confidence bar */
.cw{margin:6px 0}
.cl{display:flex;justify-content:space-between;margin-bottom:4px}
.cl span:first-child{font-size:12px;color:var(--t2)}
.cl span:last-child{font-size:11px;font-family:var(--fm);color:var(--t3)}
.ct{height:5px;background:var(--b1);border-radius:4px;overflow:hidden}
.cf{height:5px;border-radius:4px;transition:width .5s ease}

/* Sidebar brand */
.sbbrand{
  padding:1.75rem 1.25rem 1rem;
  border-bottom:1px solid var(--b1);margin-bottom:.5rem;
}
.sbbname{
  font-size:16px;font-weight:800;color:var(--t1);
  letter-spacing:-.02em;font-family:var(--fd);
}
.sbbtag{
  font-size:10px;font-weight:600;color:var(--t3);
  text-transform:uppercase;letter-spacing:.12em;margin-top:3px;
}
.sbns{
  font-size:9px;font-weight:700;text-transform:uppercase;
  letter-spacing:.14em;color:var(--t4);
  padding:1rem 1.25rem .4rem;
  font-family:var(--fd);
}
.sbni{
  display:flex;align-items:center;gap:10px;
  padding:9px 1.25rem;cursor:pointer;
  font-size:13px;font-weight:600;color:var(--t2);
  transition:all .12s;border-left:2px solid transparent;
  margin:1px 0;font-family:var(--fs);
}
.sbni:hover{background:var(--s2);color:var(--t1)}
.sbni.act{background:var(--blue-d);color:var(--blue);border-left-color:var(--blue);font-weight:700}

/* Time tag */
.ttag{
  display:inline-flex;align-items:center;gap:4px;
  background:var(--s3);border:1px solid var(--b2);
  color:var(--t3);font-family:var(--fm);font-size:11px;
  padding:2px 8px;border-radius:4px;
}

/* Result box */
.rb{
  background:var(--s1);border:1px solid var(--b1);
  border-radius:var(--r);padding:1.5rem;
  min-height:200px;display:flex;
  align-items:center;justify-content:center;
}

/* Sep */
.sep{border:none;border-top:1px solid var(--b1);margin:1.5rem 0}

/* Action card */
.act-item{
  background:var(--s3);border-radius:var(--rs);
  padding:.75rem 1rem;margin-bottom:.5rem;
  border-left:3px solid var(--teal);
  font-size:13px;color:var(--t2);
  font-family:var(--fs);line-height:1.6;
}

/* Meeting overview card */
.meet-role{
  display:inline-block;padding:2px 8px;
  border-radius:4px;font-size:11px;
  font-weight:700;font-family:var(--fd);
  letter-spacing:.04em;margin-right:6px;
}
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
        st.session_state[k] = v if not isinstance(v, set) else set()

# ── Device detection ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def detect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = detect_device()

# ── Model loaders with error handling ───────────────────────
@st.cache_resource(show_spinner=False)
def get_whisper():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if DEVICE == "cuda" else -1,
        return_timestamps=True,
    )

@st.cache_resource(show_spinner=False)
def get_intent_clf():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def get_ner():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def get_summarizer():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def get_flan():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if DEVICE == "cuda" else -1,
    )

@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Helpers ─────────────────────────────────────────────────
def timed(fn, *a, **kw):
    t = time.time()
    r = fn(*a, **kw)
    return r, round(time.time() - t, 2)

def log_inf(model, secs, tokens=0):
    st.session_state["inference_log"].append({
        "model": model, "time": secs,
        "tokens": tokens, "ts": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state["models_loaded"].add(model)

def badge(txt, c="b"):
    cls = {"b":"bb","t":"bt","a":"ba","v":"bv","r":"br","gr":"bgr"}.get(c,"bgr")
    return f'<span class="badge {cls}">{txt}</span>'

def ttag(s):
    return f'<span class="ttag">⏱ {s}s</span>'

def conf_bar(label, score, color="blue"):
    p = int(score * 100)
    return f"""<div class="cw">
      <div class="cl"><span>{label}</span><span>{p}%</span></div>
      <div class="ct"><div class="cf" style="width:{p}%;background:var(--{color})"></div></div>
    </div>"""

def page_header(title, subtitle, badge_html=""):
    st.markdown(f"""<div class="pheader">
      <div>
        <div class="ptitle">{title}</div>
        <div class="psub">{subtitle}</div>
      </div>
      <div style="flex-shrink:0">{badge_html}</div>
    </div>""", unsafe_allow_html=True)

def dl_json(data, fname):
    st.download_button(
        "⬇ Export JSON",
        json.dumps(data, indent=2, ensure_ascii=False),
        file_name=fname, mime="application/json"
    )

def make_plotly_dark(fig, h=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="JetBrains Mono",
        font_color="#8899bb",
        margin=dict(l=0, r=0, t=24, b=0),
        height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#8899bb"),
        xaxis=dict(gridcolor="#1a2440", zerolinecolor="#1a2440", tickfont_color="#445577"),
        yaxis=dict(gridcolor="#1a2440", zerolinecolor="#1a2440", tickfont_color="#445577"),
    )
    return fig

def safe_run(fn, *args, err_msg="Model error", **kwargs):
    """Run a function safely and show a nice error on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.error(f"⚠️ {err_msg}: {str(e)[:200]}")
        return None

# ── Sidebar Navigation ───────────────────────────────────────
NAV = {
    "Overview":        [("Home", "🏠")],
    "Pipeline Stages": [
        ("Speech To Text", "🎙"),
        ("Intent Classification", "🎯"),
        ("Named Entity Recognition", "🏷"),
        ("Summarization", "📝"),
        ("Action Extraction", "✅"),
        ("Embeddings", "🔵"),
    ],
    "Integrated":   [("Full Pipeline", "⚡")],
    "Intelligence": [("Analytics", "📊"), ("System Monitor", "🖥")],
}

with st.sidebar:
    st.markdown("""<div class="sbbrand">
      <div class="sbbname">Meeting NLU</div>
      <div class="sbbtag">Intelligence Platform · v2.0</div>
    </div>""", unsafe_allow_html=True)

    for section, items in NAV.items():
        st.markdown(f'<div class="sbns">{section}</div>', unsafe_allow_html=True)
        for pk, icon in items:
            is_active = st.session_state["page"] == pk
            if st.button(f"{icon}  {pk}", key=f"nav_{pk}", use_container_width=True):
                st.session_state["page"] = pk
                st.rerun()

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    dev_badge = (badge('<span class="dot dg"></span> GPU Active', "t")
                 if DEVICE == "cuda"
                 else badge('<span class="dot da"></span> CPU Mode', "a"))
    st.markdown(f"<div style='padding:0 1.25rem .5rem'>{dev_badge}</div>", unsafe_allow_html=True)
    n = len(st.session_state["models_loaded"])
    st.markdown(
        f"<div style='padding:0 1.25rem 1rem;font-size:11px;color:var(--t3)'>"
        f"Models Cached: <b style='color:var(--t2)'>{n}/6</b></div>",
        unsafe_allow_html=True
    )

PAGE = st.session_state["page"]


# ════════════════════════════════════════════════════════════
#  HOME
# ════════════════════════════════════════════════════════════
if PAGE == "Home":
    logs = st.session_state["inference_log"]
    loaded = st.session_state["models_loaded"]
    avg_t = round(sum(l["time"] for l in logs) / max(len(logs), 1), 2) if logs else 0.0

    st.markdown("""<div style="padding:2.5rem 0 2rem">
      <div style="font-size:2.4rem;font-weight:800;font-family:'Syne',sans-serif;
                  letter-spacing:-.04em;color:#e8edf8;line-height:1.1;margin-bottom:.6rem">
        Meeting NLU<br>
        <span style="background:linear-gradient(135deg,#4f8ef7,#00d4aa);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
          Intelligence Platform
        </span>
      </div>
      <div style="font-size:14px;color:#8899bb;max-width:560px;line-height:1.7">
        An end-to-end NLP pipeline that turns raw meeting audio and text into structured insights —
        transcription, intent detection, named entities, summaries, and action items.
      </div>
    </div>""", unsafe_allow_html=True)

    # Stat row
    def scard(col, lbl, val, sub, ac):
        with col:
            st.markdown(f"""<div class="sc">
              <div class="sa" style="background:var(--{ac})"></div>
              <div class="slb">{lbl}</div>
              <div class="sv">{val}</div>
              <div class="ss">{sub}</div>
            </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    scard(c1, "Models", f"{len(loaded)}/6", "Cached this session", "blue")
    scard(c2, "Avg Time", f"{avg_t}s", "Per inference call", "teal")
    scard(c3, "Device", DEVICE.upper(), "Torch backend", "amber" if DEVICE=="cpu" else "teal")
    scard(c4, "Calls", str(len(logs)), "Total this session", "violet")
    scard(c5, "Pipeline Runs", str(len([v for v in st.session_state["pipe_state"].values() if v])), "Full runs", "blue")
    scard(c6, "Status", "Live", "Streamlit Cloud", "teal")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="slbl">NLU Pipeline — 7 Stages</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Audio Input",              "WAV / MP3 / M4A upload",       "🎵"),
            ("2", "Speech-to-Text",           "openai/whisper-small",         "🎙"),
            ("3", "Intent Classification",    "facebook/bart-large-mnli",     "🎯"),
            ("4", "Named Entity Recognition", "dslim/bert-base-NER",          "🏷"),
            ("5", "Summarization",            "facebook/bart-large-cnn",      "📝"),
            ("6", "Action Extraction",        "google/flan-t5-base",          "✅"),
            ("7", "Embeddings & Retrieval",   "all-MiniLM-L6-v2",            "🔵"),
        ]
        for num, lbl, sub, icon in steps:
            is_done = any(sub in m or lbl.split()[0].lower() in m.lower() for m in loaded)
            cls = "pstep done" if is_done else "pstep"
            dot = '<span class="dot dg"></span>' if is_done else '<span class="dot da"></span>'
            st.markdown(f"""<div class="{cls}">
              <div class="psn">{num}</div>
              <div style="flex:1">
                <div class="psl">{dot} {icon} {lbl}</div>
                <div class="pss">{sub}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="slbl">Model Registry</div>', unsafe_allow_html=True)
        models_info = [
            ("Whisper",   "openai/whisper-small",        "STT",    "b"),
            ("BART-MNLI", "facebook/bart-large-mnli",    "Intent", "t"),
            ("BERT-NER",  "dslim/bert-base-NER",         "NER",    "a"),
            ("BART-CNN",  "facebook/bart-large-cnn",     "Summ",   "v"),
            ("Flan-T5",   "google/flan-t5-base",         "Action", "r"),
            ("MiniLM",    "all-MiniLM-L6-v2",            "Embed",  "b"),
        ]
        for short, full, task, color in models_info:
            is_loaded = any(full in m or short in m for m in loaded)
            sb = badge("✓ Cached", "t") if is_loaded else badge("On-demand", "gr")
            st.markdown(f"""<div class="card" style="padding:.85rem 1rem;margin-bottom:.5rem">
              <div style="display:flex;align-items:center;justify-content:space-between">
                <div style="display:flex;align-items:center;gap:8px">
                  {badge(task, color)}
                  <span style="font-size:12px;font-weight:700;color:var(--t1);font-family:'Syne',sans-serif">{short}</span>
                </div>
                {sb}
              </div>
              <div style="font-size:11px;color:var(--t3);font-family:'JetBrains Mono',monospace;margin-top:5px">{full}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards instead of "guide"
    st.markdown('<div class="slbl">Platform Capabilities</div>', unsafe_allow_html=True)
    caps = [
        ("🎙 Speech-to-Text", "Whisper-powered ASR supports WAV, MP3, M4A. Detects language automatically and outputs timestamped segments.", "b"),
        ("🎯 Zero-Shot Intent", "BART-MNLI classifies any text into custom intents without retraining. Perfect for dynamic meeting contexts.", "t"),
        ("🏷 Named Entity Recognition", "BERT NER highlights people, organizations, and locations with color-coded visual output.", "a"),
        ("📝 Smart Summarization", "BART-CNN condenses long transcripts into concise, extractive summaries with compression metrics.", "v"),
        ("✅ Action Item Extraction", "Flan-T5 reads meeting text and generates structured action items with assignees and deadlines.", "r"),
        ("🔵 Semantic Embeddings", "MiniLM encodes sentences into 384-dim vectors for similarity search and clustering visualization.", "b"),
        ("⚡ Full Pipeline", "Chain all 6 models end-to-end on a meeting transcript. Export the full structured JSON report.", "t"),
        ("📊 Analytics", "Real-time inference timing, model usage history, and session-level performance metrics.", "a"),
    ]
    cols = st.columns(4)
    for i, (title, body, color) in enumerate(caps):
        with cols[i % 4]:
            st.markdown(f"""<div class="card card-{color}" style="height:140px">
              <div style="font-size:13px;font-weight:700;color:var(--t1);
                          font-family:'Syne',sans-serif;margin-bottom:6px">{title}</div>
              <div style="font-size:12px;color:var(--t2);line-height:1.6">{body}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SPEECH TO TEXT
# ════════════════════════════════════════════════════════════
elif PAGE == "Speech To Text":
    page_header(
        "Speech To Text",
        "Convert meeting audio to text with automatic language detection and timestamps.",
        badge("openai/whisper-small", "b")
    )
    st.markdown("""<div class="card card-b" style="margin-bottom:1.5rem">
      <div style="font-size:12px;color:var(--t2);line-height:1.8">
        OpenAI Whisper (244M params) performs automatic speech recognition with chunk-level timestamps.
        Audio is resampled to 16kHz mono internally. <b style="color:var(--t1)">Translate mode</b> converts any language to English.
        Supported formats: <b style="color:var(--t1)">MP3, WAV, M4A, OGG, FLAC</b>.
      </div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="slbl">Audio Input</div>', unsafe_allow_html=True)
        afile = st.file_uploader("Upload audio file", type=["mp3","wav","m4a","ogg","flac"], key="stt_up")
        if afile:
            abytes = afile.read()
            st.audio(abytes)
            sz = round(len(abytes)/1024, 1)
            st.markdown(f"""<div class="card" style="padding:.8rem 1rem;margin-top:.75rem">
              <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
                <div><div class="slb">File</div>
                  <div style="font-size:12px;color:var(--t2);font-family:'JetBrains Mono',monospace">{afile.name}</div></div>
                <div><div class="slb">Size</div>
                  <div style="font-size:12px;color:var(--t2);font-family:'JetBrains Mono',monospace">{sz} KB</div></div>
                <div><div class="slb">Type</div>
                  <div style="font-size:12px;color:var(--t2);font-family:'JetBrains Mono',monospace">{afile.type or 'audio'}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        task_mode = st.radio("Task mode", ["transcribe", "translate"], horizontal=True)
        run_btn = st.button("▶ Run Transcription", disabled=afile is None)

        if run_btn and afile:
            tmp = f"/tmp/stt_{int(time.time())}.audio"
            with open(tmp, "wb") as f:
                f.write(abytes)
            with st.spinner("Loading Whisper... (first load takes ~30s)"):
                asr = safe_run(get_whisper, err_msg="Whisper load failed")
            if asr:
                prog = st.progress(0, "Transcribing audio...")
                t0 = time.time()
                res = safe_run(asr, tmp, generate_kwargs={"task": task_mode}, err_msg="Transcription failed")
                if res:
                    elapsed = round(time.time() - t0, 2)
                    prog.progress(1.0, "Complete ✓")
                    st.session_state["transcript"] = res["text"]
                    st.session_state["transcript_chunks"] = res.get("chunks", [])
                    st.session_state["transcript_time"] = elapsed
                    log_inf("Whisper STT", elapsed, len(res["text"].split()))
                    st.rerun()

    with right:
        st.markdown('<div class="slbl">Transcription Output</div>', unsafe_allow_html=True)
        if st.session_state["transcript"]:
            txt = st.session_state["transcript"]
            elapsed = st.session_state["transcript_time"]
            chunks = st.session_state["transcript_chunks"]
            st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:.75rem">
              {badge('<span class="dot dg"></span> Complete', "t")}
              {ttag(elapsed)}
              {badge(f'{len(txt.split())} words', "gr")}
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="mono">{txt}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            with e1: st.download_button("⬇ Export TXT", txt, "transcript.txt", "text/plain")
            with e2: dl_json({"transcript": txt, "chunks": chunks, "inference_time": elapsed}, "transcript.json")
            if chunks:
                st.markdown('<div class="slbl" style="margin-top:1rem">Timestamped Segments</div>', unsafe_allow_html=True)
                rows = [{"Start": c.get("timestamp", [None,None])[0],
                         "End":   c.get("timestamp", [None,None])[1],
                         "Text":  c.get("text", "")} for c in chunks]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("""<div class="rb">
              <div style="text-align:center">
                <div style="font-size:2rem;margin-bottom:.5rem">🎙</div>
                <div style="font-size:13px;color:var(--t3)">Upload audio and run transcription</div>
                <div style="font-size:11px;color:var(--t4);margin-top:4px">Results appear here</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION
# ════════════════════════════════════════════════════════════
elif PAGE == "Intent Classification":
    page_header(
        "Intent Classification",
        "Zero-shot intent detection — no fine-tuning required, fully custom label sets.",
        badge("facebook/bart-large-mnli", "t")
    )
    st.markdown("""<div class="card card-t" style="margin-bottom:1.5rem">
      <div style="font-size:12px;color:var(--t2);line-height:1.8">
        BART-MNLI computes entailment scores between input text and candidate labels.
        Works with any label set. <b style="color:var(--t1)">Multi-label mode</b> returns scores for all candidates simultaneously.
      </div>
    </div>""", unsafe_allow_html=True)

    DEFAULT_LABELS = [
        "schedule meeting","send email","create task","approve document",
        "request update","cancel appointment","provide feedback",
        "escalate issue","assign responsibility","set deadline"
    ]
    tabs = st.tabs(["Intent Prediction", "Multi-Intent Ranking", "Custom Label Classification"])

    with tabs[0]:
        tin = st.text_area("Input text", value=st.session_state.get("intent_text",""),
                           placeholder="Enter meeting sentence to classify...", height=100, key="ii1")
        if st.button("▶ Predict Intent", key="bi1"):
            if tin.strip():
                with st.spinner("Loading BART classifier..."):
                    clf = safe_run(get_intent_clf, err_msg="BART load failed")
                if clf:
                    res, elapsed = timed(clf, tin, candidate_labels=DEFAULT_LABELS[:5], multi_label=False)
                    st.session_state["intents"] = list(zip(res["labels"], res["scores"]))
                    st.session_state["intent_text"] = tin
                    log_inf("BART Intent", elapsed, len(tin.split()))
                    tl, ts = res["labels"][0], res["scores"][0]
                    st.markdown("<br>", unsafe_allow_html=True)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Top Intent", tl)
                    c2.metric("Confidence", f"{round(ts*100,1)}%")
                    c3.metric("Inference Time", f"{elapsed}s")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="slbl">Confidence Breakdown</div>', unsafe_allow_html=True)
                    for lbl, score in zip(res["labels"], res["scores"]):
                        st.markdown(conf_bar(lbl, score, "blue"), unsafe_allow_html=True)
            else:
                st.warning("Enter text to classify.")

    with tabs[1]:
        tin2 = st.text_area("Input text", value=st.session_state.get("intent_text",""),
                            placeholder="Enter meeting sentence...", height=100, key="ii2")
        topk = st.slider("Top intents to show", 2, 10, 6, key="topk")
        if st.button("▶ Rank Intents", key="bi2"):
            if tin2.strip():
                with st.spinner("Ranking intents..."):
                    clf = safe_run(get_intent_clf, err_msg="BART load failed")
                if clf:
                    res, elapsed = timed(clf, tin2, candidate_labels=DEFAULT_LABELS, multi_label=True)
                    log_inf("BART Intent", elapsed, len(tin2.split()))
                    df = pd.DataFrame({
                        "Intent": res["labels"][:topk],
                        "Score": [round(s,4) for s in res["scores"][:topk]],
                        "Pct": [round(s*100,1) for s in res["scores"][:topk]]
                    })
                    st.markdown(f'<div style="margin-bottom:.75rem">{ttag(elapsed)}</div>', unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=df["Pct"], y=df["Intent"], orientation="h",
                        marker_color="#4f8ef7", marker_line_width=0,
                        text=[f"{v}%" for v in df["Pct"]], textposition="outside",
                        textfont=dict(size=10, color="#8899bb")
                    ))
                    make_plotly_dark(fig, 280)
                    fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=11)))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df[["Intent","Score","Pct"]], use_container_width=True, hide_index=True)
            else:
                st.warning("Enter text first.")

    with tabs[2]:
        tin3 = st.text_area("Input text", placeholder="Describe the meeting action...", height=100, key="ii3")
        craw = st.text_input("Custom labels (comma-separated)",
            value="approve budget, reject proposal, request clarification, defer decision, escalate to manager", key="cl")
        if st.button("▶ Classify", key="bi3"):
            lbls = [l.strip() for l in craw.split(",") if l.strip()]
            if tin3.strip() and lbls:
                with st.spinner("Classifying..."):
                    clf = safe_run(get_intent_clf, err_msg="BART load failed")
                if clf:
                    res, elapsed = timed(clf, tin3, candidate_labels=lbls, multi_label=True)
                    log_inf("BART Intent", elapsed)
                    df = pd.DataFrame({
                        "Label": res["labels"],
                        "Score": [round(s,4) for s in res["scores"]]
                    })
                    c1, c2 = st.columns([2,1])
                    with c1:
                        fig = px.bar(df, x="Label", y="Score",
                                     color="Score", color_continuous_scale=["#121a2e","#4f8ef7"],
                                     template="plotly_dark")
                        make_plotly_dark(fig, 260)
                        fig.update_coloraxes(showscale=False)
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        dl_json({"text":tin3,"labels":res["labels"],"scores":res["scores"]}, "intent_results.json")
            else:
                st.warning("Provide text and at least one label.")


# ════════════════════════════════════════════════════════════
#  NAMED ENTITY RECOGNITION
# ════════════════════════════════════════════════════════════
elif PAGE == "Named Entity Recognition":
    page_header(
        "Named Entity Recognition",
        "Highlight people, organizations, and locations in meeting transcripts.",
        badge("dslim/bert-base-NER", "a")
    )
    st.markdown("""<div class="card card-a" style="margin-bottom:1.5rem">
      <div style="font-size:12px;color:var(--t2);line-height:1.8">
        BERT fine-tuned on CoNLL-2003 NER. Entity types:
        <span style="color:#93c5fd">PER</span> (persons),
        <span style="color:#5eead4">ORG</span> (organizations),
        <span style="color:#fcd34d">LOC</span> (locations),
        <span style="color:#c4b5fd">MISC</span> (miscellaneous).
        Aggregation strategy: <b style="color:var(--t1)">simple</b> (merges word-piece tokens).
      </div>
    </div>""", unsafe_allow_html=True)

    DEMO_TEXT = """Alice scheduled a meeting with the engineering team at Google headquarters in Mountain View.
John from Microsoft will present the quarterly roadmap. Sarah and Mike will join remotely from London.
The board meeting with Amazon representatives is set for next Thursday in New York."""

    tabs = st.tabs(["Highlight Entities", "Entity Table", "Batch Analysis"])

    with tabs[0]:
        txt_in = st.text_area("Input text", value=st.session_state.get("ner_text", DEMO_TEXT), height=130, key="ner1")
        if st.button("▶ Run NER", key="bn1"):
            if txt_in.strip():
                with st.spinner("Running NER..."):
                    ner = safe_run(get_ner, err_msg="NER model failed")
                if ner:
                    ents, elapsed = timed(ner, txt_in)
                    st.session_state["ner_results"] = ents
                    st.session_state["ner_text"] = txt_in
                    log_inf("BERT NER", elapsed, len(txt_in.split()))
                    # Build highlighted HTML
                    out = txt_in
                    for ent in sorted(ents, key=lambda x: x["start"], reverse=True):
                        css = {"PER":"ePER","ORG":"eORG","LOC":"eLOC","MISC":"eMISC"}.get(ent["entity_group"],"eDEF")
                        span = (f'<span class="ent {css}" title="{ent["entity_group"]} '
                                f'({round(ent["score"]*100,1)}%)">{ent["word"]}</span>')
                        out = out[:ent["start"]] + span + out[ent["end"]:]
                    st.markdown(f'<div style="line-height:2.5;font-size:14px;padding:1rem;background:var(--s2);border:1px solid var(--b1);border-radius:var(--r)">{out}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Legend
                    st.markdown("""<div style="display:flex;gap:1rem;flex-wrap:wrap">
                      <span class="badge bb">PER — Person</span>
                      <span class="badge bt">ORG — Organization</span>
                      <span class="badge ba">LOC — Location</span>
                      <span class="badge bv">MISC — Miscellaneous</span>
                    </div>""", unsafe_allow_html=True)
                    st.markdown(f'<div style="margin-top:.75rem">{ttag(elapsed)} {badge(f"{len(ents)} entities","gr")}</div>', unsafe_allow_html=True)
            else:
                st.warning("Enter text first.")

    with tabs[1]:
        if st.session_state["ner_results"]:
            ents = st.session_state["ner_results"]
            df = pd.DataFrame([{
                "Entity": e["word"],
                "Type": e["entity_group"],
                "Confidence": f"{round(e['score']*100,1)}%",
                "Start": e["start"],
                "End": e["end"]
            } for e in ents])
            st.dataframe(df, use_container_width=True, hide_index=True)
            dl_json(ents, "ner_results.json")
            # Type distribution
            type_counts = df["Type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig = px.pie(type_counts, names="Type", values="Count",
                         color_discrete_sequence=["#4f8ef7","#00d4aa","#ffb347","#a78bfa"],
                         template="plotly_dark")
            make_plotly_dark(fig, 250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run NER analysis first in the 'Highlight Entities' tab.")

    with tabs[2]:
        sents_raw = st.text_area("Multiple sentences (one per line)", height=150, key="ner_batch",
                                  placeholder="Alice called John at Microsoft.\nSarah flew to London yesterday.")
        if st.button("▶ Analyze All", key="bn2"):
            sents = [s.strip() for s in sents_raw.strip().split("\n") if s.strip()]
            if sents:
                with st.spinner("Running batch NER..."):
                    ner = safe_run(get_ner, err_msg="NER model failed")
                if ner:
                    all_ents = []
                    t0 = time.time()
                    for s in sents:
                        for e in ner(s):
                            all_ents.append({"sentence": s[:60]+"…" if len(s)>60 else s,
                                              "entity": e["word"], "type": e["entity_group"],
                                              "score": round(e["score"]*100,1)})
                    elapsed = round(time.time()-t0,2)
                    log_inf("BERT NER", elapsed)
                    st.markdown(f'{ttag(elapsed)} {badge(f"{len(all_ents)} entities found","t")}', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(all_ents), use_container_width=True, hide_index=True)
            else:
                st.warning("Enter sentences first.")


# ════════════════════════════════════════════════════════════
#  SUMMARIZATION
# ════════════════════════════════════════════════════════════
elif PAGE == "Summarization":
    page_header(
        "Summarization",
        "Condense long meeting transcripts into concise, extractive summaries.",
        badge("facebook/bart-large-cnn", "v")
    )
    st.markdown("""<div class="card card-v" style="margin-bottom:1.5rem">
      <div style="font-size:12px;color:var(--t2);line-height:1.8">
        BART-CNN trained on CNN/DailyMail — extractive + abstractive summarization.
        Best for transcripts between <b style="color:var(--t1)">50–1500 words</b>.
        Adjust length controls below to tune output verbosity.
      </div>
    </div>""", unsafe_allow_html=True)

    auto_fill = st.session_state.get("transcript", "")
    DEMO = """Alice: Good morning everyone. Let's get started with our weekly product sync.
John, can you give us the engineering update?
John: Sure. We completed the user authentication module this week. The database migration to PostgreSQL is 90% done, we expect to finish by Thursday.
Sarah: That's great news. On the design side, we finalized the new dashboard mockups. I'll share them with the team by end of day today.
Alice: Perfect. What about the security audit? Has that been addressed?
John: Mike is leading that. Mike, do you want to update the team?
Mike: Yes, we identified three medium-priority vulnerabilities. I'll send a detailed report to all stakeholders by tomorrow morning.
Alice: Thanks Mike. We need to schedule a follow-up meeting with the security team next week.
Sarah: Absolutely. I'll send out the calendar invite today.
Alice: One more thing — we need to finalize the Q4 budget proposal. John, please coordinate with finance to get the numbers ready before the board meeting on December 5th.
John: Got it. I'll set up a meeting with the CFO this week."""

    txt_in = st.text_area("Meeting transcript", value=auto_fill or DEMO, height=200, key="sum_in")
    c1, c2 = st.columns(2)
    with c1: max_len = st.slider("Max summary length (tokens)", 50, 300, 130, key="sl_max")
    with c2: min_len = st.slider("Min summary length (tokens)", 20, 100, 40, key="sl_min")

    if st.button("▶ Summarize", key="bsum"):
        if txt_in.strip():
            with st.spinner("Loading BART-CNN summarizer..."):
                summ = safe_run(get_summarizer, err_msg="Summarizer load failed")
            if summ:
                inp = txt_in[:1024]
                res, elapsed = timed(summ, inp, max_length=max_len, min_length=min_len, do_sample=False)
                summary = res[0]["summary_text"]
                st.session_state["summary"] = summary
                st.session_state["summary_original"] = txt_in
                st.session_state["summary_time"] = elapsed
                log_inf("BART-CNN Summ", elapsed, len(txt_in.split()))

        else:
            st.warning("Enter transcript text first.")

    if st.session_state["summary"]:
        s = st.session_state["summary"]
        orig = st.session_state["summary_original"]
        elapsed = st.session_state["summary_time"]
        ratio = round((1 - len(s.split())/max(len(orig.split()),1))*100, 1)
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Original Words", len(orig.split()))
        m2.metric("Summary Words", len(s.split()))
        m3.metric("Compression", f"{ratio}%")
        m4.metric("Inference Time", f"{elapsed}s")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="slbl">Summary Output</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card card-v"><div style="font-size:14px;color:var(--t1);line-height:1.8">{s}</div></div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1: st.download_button("⬇ Export TXT", s, "summary.txt", "text/plain")
        with e2: dl_json({"summary":s,"original_words":len(orig.split()),"summary_words":len(s.split()),"compression":f"{ratio}%","time":elapsed}, "summary.json")


# ════════════════════════════════════════════════════════════
#  ACTION EXTRACTION
# ════════════════════════════════════════════════════════════
elif PAGE == "Action Extraction":
    page_header(
        "Action Item Extraction",
        "Extract structured action items, owners, and deadlines from meeting text.",
        badge("google/flan-t5-base", "r")
    )
    st.markdown("""<div class="card card-r" style="margin-bottom:1.5rem">
      <div style="font-size:12px;color:var(--t2);line-height:1.8">
        Flan-T5 (instruction-tuned T5) extracts actionable tasks from meeting transcripts.
        The model is prompted to output a structured list of action items with owners and deadlines.
        Best results with transcripts ≥ 3 sentences.
      </div>
    </div>""", unsafe_allow_html=True)

    auto_fill = st.session_state.get("transcript", "")
    DEMO = """John: We completed the user authentication module. The database migration is 90% done, we expect to finish by Thursday.
Sarah: I'll share the new dashboard mockups with the team by end of day today.
Mike: I'll send a detailed security vulnerability report to all stakeholders by tomorrow morning.
Alice: We need to schedule a follow-up meeting with the security team next week. Sarah, can you set that up?
John: I'll set up a meeting with the CFO this week to finalize the Q4 budget before the board meeting on December 5th."""

    tabs = st.tabs(["Extract Actions", "Custom Prompt", "Action Items from Transcript"])

    with tabs[0]:
        txt_in = st.text_area("Meeting text", value=auto_fill or DEMO, height=180, key="act1")
        if st.button("▶ Extract Action Items", key="bact1"):
            if txt_in.strip():
                prompt = f"Extract all action items, assign owners, and list deadlines from this meeting:\n\n{txt_in[:900]}\n\nAction items:"
                with st.spinner("Running Flan-T5..."):
                    flan = safe_run(get_flan, err_msg="Flan-T5 load failed")
                if flan:
                    res, elapsed = timed(flan, prompt, max_new_tokens=250, do_sample=False)
                    output = res[0]["generated_text"]
                    st.session_state["actions_output"] = output
                    log_inf("Flan-T5 Action", elapsed, len(txt_in.split()))
                    st.markdown(f'<div style="margin-bottom:.75rem">{ttag(elapsed)}</div>', unsafe_allow_html=True)
                    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
                    for line in lines:
                        st.markdown(f'<div class="act-item">✅ {line}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    dl_json({"input": txt_in, "action_items": lines, "raw": output, "time": elapsed}, "actions.json")
            else:
                st.warning("Enter meeting text first.")

    with tabs[1]:
        custom_prompt = st.text_area("Custom prompt (use {text} as placeholder)",
            value="List all decisions made and who is responsible for each from this meeting:\n\n{text}\n\nDecisions and owners:",
            height=120, key="act_prompt")
        txt_c = st.text_area("Meeting text", value=auto_fill or DEMO, height=150, key="act2")
        if st.button("▶ Run Custom Prompt", key="bact2"):
            if txt_c.strip() and "{text}" in custom_prompt:
                full_prompt = custom_prompt.replace("{text}", txt_c[:800])
                with st.spinner("Running Flan-T5..."):
                    flan = safe_run(get_flan, err_msg="Flan-T5 load failed")
                if flan:
                    res, elapsed = timed(flan, full_prompt, max_new_tokens=250, do_sample=False)
                    output = res[0]["generated_text"]
                    log_inf("Flan-T5 Action", elapsed)
                    st.markdown(f'<div style="margin-bottom:.75rem">{ttag(elapsed)}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="mono">{output}</div>', unsafe_allow_html=True)
            elif "{text}" not in custom_prompt:
                st.warning("Prompt must contain {text} placeholder.")
            else:
                st.warning("Enter meeting text.")

    with tabs[2]:
        st.markdown("""<div class="card" style="margin-bottom:1rem">
          <div style="font-size:12px;color:var(--t2);line-height:1.7">
            Paste a full meeting transcript. The model will split by speaker turns and extract
            action items per speaker.
          </div>
        </div>""", unsafe_allow_html=True)
        transcript_in = st.text_area("Full transcript (Speaker: text format)", height=200, key="act3",
            placeholder="Alice: We need to...\nJohn: I'll handle...")
        if st.button("▶ Analyze by Speaker", key="bact3"):
            if transcript_in.strip():
                speakers = {}
                for line in transcript_in.strip().split("\n"):
                    if ":" in line:
                        sp, text = line.split(":", 1)
                        speakers.setdefault(sp.strip(), []).append(text.strip())
                if speakers:
                    with st.spinner("Extracting per-speaker actions..."):
                        flan = safe_run(get_flan, err_msg="Flan-T5 load failed")
                    if flan:
                        t0 = time.time()
                        for sp, lines in speakers.items():
                            combined = " ".join(lines)
                            prompt = f"What action items did {sp} commit to in this meeting text: {combined[:400]}\nAction items:"
                            res = flan(prompt, max_new_tokens=150, do_sample=False)
                            output = res[0]["generated_text"]
                            st.markdown(f"""<div class="card" style="margin-bottom:.5rem">
                              <div class="meet-role" style="background:var(--blue-d);color:var(--blue)">{sp}</div>
                              <div style="font-size:13px;color:var(--t2);margin-top:.5rem;line-height:1.7">{output}</div>
                            </div>""", unsafe_allow_html=True)
                        elapsed = round(time.time()-t0, 2)
                        log_inf("Flan-T5 Action", elapsed)
                        st.markdown(f'{ttag(elapsed)}', unsafe_allow_html=True)
                else:
                    st.warning("No speaker turns detected. Format as 'Speaker: text'.")
            else:
                st.warning("Enter transcript first.")


# ════════════════════════════════════════════════════════════
#  EMBEDDINGS
# ════════════════════════════════════════════════════════════
elif PAGE == "Embeddings":
    page_header(
        "Semantic Embeddings",
        "Sentence-level semantic vectors for similarity search and clustering.",
        badge("all-MiniLM-L6-v2", "b")
    )
    st.markdown("""<div class="card card-b" style="margin-bottom:1.5rem">
      <div style="font-size:12px;color:var(--t2);line-height:1.8">
        MiniLM encodes sentences into <b style="color:var(--t1)">384-dimensional</b> L2-normalized vectors.
        Cosine similarity measures semantic relatedness. Use for meeting search, deduplication, and topic clustering.
      </div>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["Similarity Search", "Embedding Visualization", "Semantic Clustering"])

    with tabs[0]:
        query = st.text_input("Search query", placeholder="Who owns the security audit?", key="emb_q")
        corpus_raw = st.text_area("Corpus sentences (one per line)", height=150, key="emb_c",
            value="""Alice will schedule the security team follow-up meeting next week.
John is responsible for the database migration completion by Thursday.
Sarah will share the dashboard mockups with the team today.
Mike will send the vulnerability report to all stakeholders tomorrow.
John will set up a CFO meeting to finalize the Q4 budget before December 5th.
The engineering team completed the user authentication module this week.""")
        if st.button("▶ Search", key="bemb1"):
            if query.strip() and corpus_raw.strip():
                sentences = [s.strip() for s in corpus_raw.strip().split("\n") if s.strip()]
                with st.spinner("Computing embeddings..."):
                    emb = safe_run(get_embedder, err_msg="Embedder load failed")
                if emb:
                    t0 = time.time()
                    q_vec = emb.encode([query], normalize_embeddings=True)
                    s_vecs = emb.encode(sentences, normalize_embeddings=True)
                    scores = (s_vecs @ q_vec.T).flatten().tolist()
                    elapsed = round(time.time()-t0, 2)
                    log_inf("MiniLM Embed", elapsed, len(sentences))
                    df = pd.DataFrame({"Sentence": sentences, "Score": [round(s,4) for s in scores]})
                    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
                    st.markdown(f'<div style="margin-bottom:.75rem">{ttag(elapsed)}</div>', unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=df["Score"], y=df["Sentence"].str[:60]+"…",
                        orientation="h", marker_color="#4f8ef7", marker_line_width=0,
                        text=[f"{v:.3f}" for v in df["Score"]], textposition="outside",
                        textfont=dict(size=10, color="#8899bb")
                    ))
                    make_plotly_dark(fig, 280)
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Enter query and corpus.")

    with tabs[1]:
        viz_text = st.text_area("Sentences to visualize (one per line)", height=150, key="emb_v",
            placeholder="Enter 5-20 meeting sentences...")
        if st.button("▶ Visualize Embeddings", key="bemb2"):
            sentences = [s.strip() for s in viz_text.strip().split("\n") if s.strip()]
            if len(sentences) >= 3:
                with st.spinner("Computing embeddings..."):
                    emb = safe_run(get_embedder, err_msg="Embedder load failed")
                if emb:
                    t0 = time.time()
                    vecs = emb.encode(sentences, normalize_embeddings=True)
                    elapsed = round(time.time()-t0, 2)
                    log_inf("MiniLM Embed", elapsed)
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(vecs)
                    df = pd.DataFrame({"x": coords[:,0], "y": coords[:,1],
                                       "text": [s[:50]+"…" if len(s)>50 else s for s in sentences]})
                    fig = px.scatter(df, x="x", y="y", text="text", template="plotly_dark")
                    fig.update_traces(marker=dict(size=12, color="#4f8ef7", opacity=.85),
                                      textfont=dict(size=10, color="#8899bb"),
                                      textposition="top center")
                    make_plotly_dark(fig, 380)
                    st.markdown(f'<div style="margin-bottom:.5rem">{ttag(elapsed)} {badge("PCA 2D projection","gr")}</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Enter at least 3 sentences.")

    with tabs[2]:
        cluster_text = st.text_area("Meeting sentences to cluster (one per line)", height=150, key="emb_cl",
            placeholder="Enter 6+ meeting sentences for clustering...")
        n_clusters = st.slider("Number of clusters", 2, 8, 3, key="n_cl")
        if st.button("▶ Cluster", key="bemb3"):
            sentences = [s.strip() for s in cluster_text.strip().split("\n") if s.strip()]
            if len(sentences) >= n_clusters:
                with st.spinner("Clustering..."):
                    emb = safe_run(get_embedder, err_msg="Embedder load failed")
                if emb:
                    from sklearn.cluster import KMeans
                    from sklearn.decomposition import PCA
                    t0 = time.time()
                    vecs = emb.encode(sentences, normalize_embeddings=True)
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                    labels = km.fit_predict(vecs)
                    elapsed = round(time.time()-t0, 2)
                    log_inf("MiniLM Embed", elapsed)
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(vecs)
                    df = pd.DataFrame({"x": coords[:,0], "y": coords[:,1],
                                       "Cluster": [f"Cluster {l+1}" for l in labels],
                                       "text": [s[:50]+"…" if len(s)>50 else s for s in sentences]})
                    fig = px.scatter(df, x="x", y="y", color="Cluster", text="text",
                                     color_discrete_sequence=["#4f8ef7","#00d4aa","#ffb347","#a78bfa","#ff5e7d","#f472b6","#34d399","#fb923c"],
                                     template="plotly_dark")
                    fig.update_traces(marker_size=12, textfont=dict(size=10), textposition="top center")
                    make_plotly_dark(fig, 380)
                    st.plotly_chart(fig, use_container_width=True)
                    for cl in sorted(df["Cluster"].unique()):
                        group = df[df["Cluster"]==cl]["text"].tolist()
                        st.markdown(f'<div class="slbl">{cl}</div>', unsafe_allow_html=True)
                        for s in group:
                            st.markdown(f'<div class="act-item">{s}</div>', unsafe_allow_html=True)
            else:
                st.warning(f"Need at least {n_clusters} sentences.")


# ════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ════════════════════════════════════════════════════════════
elif PAGE == "Full Pipeline":
    page_header(
        "Full Meeting Pipeline",
        "End-to-end NLU: run all 6 models on a meeting transcript in one click.",
        badge("⚡ End-to-End", "t")
    )

    DEMO = """Alice: Good morning everyone. Let's get started with our weekly product sync. John, can you give us the engineering update?
John: Sure. We completed the user authentication module this week. The database migration to PostgreSQL is 90% done, we expect to finish by Thursday.
Sarah: That's great news. On the design side, we finalized the new dashboard mockups. I'll share them with the team by end of day today.
Alice: Perfect. What about the security audit? Has that been addressed?
Mike: Yes, we identified three medium-priority vulnerabilities. I'll send a detailed report to all stakeholders by tomorrow morning.
Alice: We need to schedule a follow-up meeting with the security team next week. Sarah, can you set that up?
Sarah: Absolutely. I'll send out the calendar invite today.
Alice: John, please coordinate with finance to get the Q4 budget numbers ready before the board meeting on December 5th.
John: Got it. I'll set up a meeting with the CFO this week."""

    auto_transcript = st.session_state.get("transcript", "")
    txt = st.text_area("Meeting transcript", value=auto_transcript or DEMO, height=220, key="pipe_in")

    steps_to_run = st.multiselect(
        "Pipeline stages to run",
        ["Intent Classification", "NER", "Summarization", "Action Extraction", "Embeddings"],
        default=["Intent Classification", "NER", "Summarization", "Action Extraction"],
        key="pipe_stages"
    )

    if st.button("▶ Run Full Pipeline", key="bpipe"):
        if txt.strip() and steps_to_run:
            results = {"transcript": txt, "stages": {}}
            total_t = 0

            # Intent
            if "Intent Classification" in steps_to_run:
                with st.spinner("Running Intent Classification..."):
                    clf = safe_run(get_intent_clf, err_msg="BART failed")
                if clf:
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=\n)", txt) if len(s.strip())>5]
                    intents = []
                    t0 = time.time()
                    DEFAULT_LABELS = ["schedule meeting","send email","create task","approve document",
                                       "request update","cancel appointment","provide feedback","escalate issue"]
                    for sent in sentences[:10]:
                        r = clf(sent, candidate_labels=DEFAULT_LABELS[:5], multi_label=False)
                        intents.append({"sentence": sent, "intent": r["labels"][0], "score": round(r["scores"][0]*100,1)})
                    elapsed = round(time.time()-t0, 2)
                    total_t += elapsed
                    results["stages"]["intent"] = intents
                    log_inf("BART Intent", elapsed)
                    st.session_state["pipe_state"]["intent"] = True

            # NER
            if "NER" in steps_to_run:
                with st.spinner("Running NER..."):
                    ner = safe_run(get_ner, err_msg="NER failed")
                if ner:
                    ents, elapsed = timed(ner, txt[:800])
                    total_t += elapsed
                    results["stages"]["ner"] = [{"entity": e["word"], "type": e["entity_group"], "score": round(e["score"],4)} for e in ents]
                    log_inf("BERT NER", elapsed)
                    st.session_state["pipe_state"]["ner"] = True

            # Summarization
            if "Summarization" in steps_to_run:
                with st.spinner("Running Summarization..."):
                    summ = safe_run(get_summarizer, err_msg="BART-CNN failed")
                if summ:
                    res, elapsed = timed(summ, txt[:1024], max_length=130, min_length=30, do_sample=False)
                    total_t += elapsed
                    results["stages"]["summary"] = res[0]["summary_text"]
                    st.session_state["summary"] = res[0]["summary_text"]
                    log_inf("BART-CNN Summ", elapsed)
                    st.session_state["pipe_state"]["summary"] = True

            # Actions
            if "Action Extraction" in steps_to_run:
                with st.spinner("Running Action Extraction..."):
                    flan = safe_run(get_flan, err_msg="Flan-T5 failed")
                if flan:
                    prompt = f"Extract all action items with owners and deadlines from this meeting:\n\n{txt[:800]}\n\nAction items:"
                    res, elapsed = timed(flan, prompt, max_new_tokens=200, do_sample=False)
                    total_t += elapsed
                    output = res[0]["generated_text"]
                    results["stages"]["actions"] = output
                    st.session_state["actions_output"] = output
                    log_inf("Flan-T5 Action", elapsed)
                    st.session_state["pipe_state"]["actions"] = True

            # Embeddings
            if "Embeddings" in steps_to_run:
                with st.spinner("Computing Embeddings..."):
                    emb_model = safe_run(get_embedder, err_msg="MiniLM failed")
                if emb_model:
                    sentences = [s.strip() for s in txt.split("\n") if len(s.strip())>5]
                    t0 = time.time()
                    vecs = emb_model.encode(sentences[:15], normalize_embeddings=True)
                    elapsed = round(time.time()-t0, 2)
                    total_t += elapsed
                    results["stages"]["embeddings"] = {"shape": f"{len(vecs)}x{len(vecs[0])}", "sentences": sentences[:15]}
                    log_inf("MiniLM Embed", elapsed)
                    st.session_state["pipe_state"]["embeddings"] = True

            st.session_state["pipe_state"]["_results"] = results
            st.session_state["pipe_state"]["_total_t"] = total_t
            st.rerun()

    # Show results
    if st.session_state["pipe_state"].get("_results"):
        results = st.session_state["pipe_state"]["_results"]
        total_t = st.session_state["pipe_state"].get("_total_t", 0)
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem">
          {badge("✓ Pipeline Complete","t")} {ttag(total_t)}
          {badge(f"{len(results['stages'])} stages","gr")}
        </div>""", unsafe_allow_html=True)

        if "summary" in results["stages"]:
            st.markdown('<div class="slbl">📝 Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card card-v"><div style="font-size:14px;color:var(--t1);line-height:1.8">{results["stages"]["summary"]}</div></div>', unsafe_allow_html=True)

        if "actions" in results["stages"]:
            st.markdown('<div class="slbl">✅ Action Items</div>', unsafe_allow_html=True)
            lines = [l.strip() for l in results["stages"]["actions"].split("\n") if l.strip()]
            for line in lines:
                st.markdown(f'<div class="act-item">{line}</div>', unsafe_allow_html=True)

        if "intent" in results["stages"]:
            st.markdown('<div class="slbl">🎯 Intent per Sentence</div>', unsafe_allow_html=True)
            df_i = pd.DataFrame(results["stages"]["intent"])
            st.dataframe(df_i, use_container_width=True, hide_index=True)

        if "ner" in results["stages"]:
            st.markdown('<div class="slbl">🏷 Named Entities</div>', unsafe_allow_html=True)
            df_e = pd.DataFrame(results["stages"]["ner"])
            st.dataframe(df_e, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        dl_json(results, "full_pipeline_results.json")


# ════════════════════════════════════════════════════════════
#  ANALYTICS
# ════════════════════════════════════════════════════════════
elif PAGE == "Analytics":
    page_header("Session Analytics", "Inference timing, model usage, and session performance.", badge("📊 Live", "t"))

    logs = st.session_state["inference_log"]
    loaded = st.session_state["models_loaded"]

    if not logs:
        st.info("No inference calls yet. Run any model to see analytics here.")
    else:
        avg_t = round(sum(l["time"] for l in logs) / len(logs), 2)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Calls", len(logs))
        c2.metric("Models Active", len(loaded))
        c3.metric("Avg Inference", f"{avg_t}s")
        c4.metric("Total Tokens", sum(l.get("tokens",0) for l in logs))

        st.markdown("<br>", unsafe_allow_html=True)
        df_log = pd.DataFrame(logs)

        l, r = st.columns(2)
        with l:
            st.markdown('<div class="slbl">Inference Time per Call</div>', unsafe_allow_html=True)
            fig = px.line(df_log, y="time", x=df_log.index, color="model",
                          color_discrete_sequence=["#4f8ef7","#00d4aa","#ffb347","#a78bfa","#ff5e7d","#f472b6"],
                          template="plotly_dark", markers=True)
            make_plotly_dark(fig, 250)
            st.plotly_chart(fig, use_container_width=True)

        with r:
            st.markdown('<div class="slbl">Calls per Model</div>', unsafe_allow_html=True)
            model_counts = df_log["model"].value_counts().reset_index()
            model_counts.columns = ["Model","Count"]
            fig2 = px.bar(model_counts, x="Model", y="Count",
                          color="Count", color_continuous_scale=["#121a2e","#4f8ef7"],
                          template="plotly_dark")
            make_plotly_dark(fig2, 250)
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="slbl">Inference Log</div>', unsafe_allow_html=True)
        st.dataframe(df_log[["ts","model","time","tokens"]], use_container_width=True, hide_index=True)
        dl_json(logs, "inference_log.json")


# ════════════════════════════════════════════════════════════
#  SYSTEM MONITOR
# ════════════════════════════════════════════════════════════
elif PAGE == "System Monitor":
    page_header("System Monitor", "Environment, hardware, and runtime diagnostics.", badge("🖥 Live", "a"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="slbl">Runtime Environment</div>', unsafe_allow_html=True)
        import sys, platform
        env_data = {
            "Python": sys.version.split()[0],
            "Platform": platform.system(),
            "Architecture": platform.machine(),
            "Compute": DEVICE.upper(),
        }
        try:
            import torch
            env_data["PyTorch"] = torch.__version__
            env_data["CUDA Available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                env_data["GPU"] = torch.cuda.get_device_name(0)
        except Exception:
            env_data["PyTorch"] = "Not loaded"
        try:
            import transformers
            env_data["Transformers"] = transformers.__version__
        except Exception:
            pass
        try:
            import streamlit as _st
            env_data["Streamlit"] = _st.__version__
        except Exception:
            pass

        for k, v in env_data.items():
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
              padding:.5rem 0;border-bottom:1px solid var(--b1)">
              <span style="font-size:12px;color:var(--t3);font-family:'Syne',sans-serif;font-weight:600">{k}</span>
              <span style="font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--t2)">{v}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="slbl">Memory & Resources</div>', unsafe_allow_html=True)
        try:
            import psutil
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.5)
            st.markdown(f"""<div class="card">
              <div style="display:flex;justify-content:space-between;margin-bottom:1rem">
                <div><div class="slb">RAM Used</div>
                  <div style="font-size:1.4rem;font-weight:800;font-family:'Syne',sans-serif">{round(ram.used/1e9,1)} GB</div>
                  <div style="font-size:11px;color:var(--t3)">{ram.percent}% of {round(ram.total/1e9,1)} GB</div>
                </div>
                <div><div class="slb">CPU Usage</div>
                  <div style="font-size:1.4rem;font-weight:800;font-family:'Syne',sans-serif">{cpu}%</div>
                </div>
              </div>
              <div class="ct" style="height:8px;margin-top:.5rem">
                <div class="cf" style="width:{ram.percent}%;background:{'var(--red)' if ram.percent>85 else 'var(--blue)'}"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        except ImportError:
            st.info("Install `psutil` to see memory stats.")
        except Exception as e:
            st.warning(f"Resource monitor unavailable: {e}")

        st.markdown('<div class="slbl" style="margin-top:1.5rem">Session Info</div>', unsafe_allow_html=True)
        start = st.session_state["session_start"]
        try:
            start_dt = datetime.fromisoformat(start)
            duration = datetime.now() - start_dt
            mins = int(duration.total_seconds() // 60)
            secs = int(duration.total_seconds() % 60)
            dur_str = f"{mins}m {secs}s"
        except Exception:
            dur_str = "—"
        st.markdown(f"""<div class="card">
          <div style="font-size:12px;color:var(--t3);margin-bottom:4px">Session Started</div>
          <div style="font-size:13px;font-family:'JetBrains Mono',monospace;color:var(--t2)">{start}</div>
          <div style="font-size:12px;color:var(--t3);margin-top:.75rem;margin-bottom:4px">Duration</div>
          <div style="font-size:13px;font-family:'JetBrains Mono',monospace;color:var(--t2)">{dur_str}</div>
        </div>""", unsafe_allow_html=True)
