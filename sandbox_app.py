"""
AI Recruiter Dashboard
Redrob Intelligent Candidate Discovery & Ranking Hackathon

Frontend-only redesign. All ranking / scoring / honeypot-detection logic
lives untouched in src/data_loader.py (score_candidate_for_jd) and is
never modified here.
"""

import csv
import io
import json
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import score_candidate_for_jd  # noqa: E402  (backend untouched)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Redrob AI Recruiter",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

JOB_ROLE = "Senior AI Engineer"
JOB_EXPERIENCE = "5–9 Years"
JOB_LOCATION = "India (Pune / Noida / Gurugram preferred)"
JOB_SKILLS = [
    "Python", "Machine Learning", "Deep Learning", "LLMs",
    "LangChain", "RAG", "FAISS", "Pinecone", "HuggingFace", "Vector Databases",
]
RANKING_ENGINE = "Deterministic Rule-Based Weighted Scoring"
COMPUTE_MODE = "CPU Only · No GPU · No Hosted LLM"


# ============================================================================
# MARKDOWN / HTML HELPER
#
# IMPORTANT: Streamlit's markdown renderer treats a BLANK LINE inside a raw
# HTML block as the end of that block. Any content after the blank line
# then gets parsed as plain Markdown/text instead of staying inside the
# HTML tag it was written in. This is what caused the entire <style> block
# in inject_css() to spill out onto the page as literal CSS text: the CSS
# has blank lines between rule groups for readability, and each one broke
# the <style>...</style> block apart. Blank lines carry no meaning in HTML
# or CSS, so html_block() strips them out unconditionally before rendering.
#
# It also strips common leading whitespace first (st.markdown's parser
# otherwise treats a consistently-indented block as a markdown "indented
# code block" and prints it verbatim instead of parsing it as HTML), and
# always passes unsafe_allow_html=True explicitly (never rely on st.write()
# for HTML content — it does not reliably honor that flag).
#
# CONTAINER AWARENESS: Streamlit only routes content into the sidebar (or
# a column) when the call happens inside that container's `with` block, or
# when you call the container-specific method directly (st.sidebar.xxx).
# Every place in this file that needs to render into the sidebar therefore
# runs inside a single `with st.sidebar:` block in render_sidebar().
# ============================================================================

def html_block(text: str) -> None:
    dedented = textwrap.dedent(text).strip()
    no_blank_lines = "\n".join(line for line in dedented.splitlines() if line.strip() != "")
    st.markdown(no_blank_lines, unsafe_allow_html=True)


# ============================================================================
# THEME
# ============================================================================

def get_theme_vars(dark: bool) -> str:
    if dark:
        return """
        :root{
            --ink-900:#f1f5f9;
            --ink-700:#cbd5e1;
            --ink-500:#94a3b8;
            --line:#26304a;
            --surface:#131a2c;
            --surface-alt:#0e1424;
            --canvas:#0a0f1c;
            --brand:#6d7bf5;
            --brand-dark:#8b96f7;
            --brand-soft:rgba(109,123,245,.16);
            --teal:#2dd4bf;
            --amber:#f2b33d;
            --rose:#f47186;
            --gold:#e8bf6a;
            --shadow:0 10px 30px -16px rgba(0,0,0,.6);
        }
        """
    return """
        :root{
            --ink-900:#0f172a;
            --ink-700:#334155;
            --ink-500:#64748b;
            --line:#e6e9f2;
            --surface:#ffffff;
            --surface-alt:#f8f9fd;
            --canvas:#f2f4fa;
            --brand:#4046d6;
            --brand-dark:#2f34ad;
            --brand-soft:#eef0fd;
            --teal:#0e9f8e;
            --amber:#c47f0a;
            --rose:#d64550;
            --gold:#b9862d;
            --shadow:0 10px 28px -18px rgba(15,23,42,.22);
        }
        """


def inject_css(dark: bool) -> None:
    html_block(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>

        {get_theme_vars(dark)}

        html, body, [class*="css"], .stMarkdown, .stText, p, span, div {{
            font-family:'Inter', -apple-system, sans-serif;
        }}
        h1, h2, h3, h4 {{ font-family:'Inter', -apple-system, sans-serif; letter-spacing:-.015em; }}
        code, .mono {{ font-family:'JetBrains Mono', monospace; }}

        .stApp{{ background:var(--canvas); color:var(--ink-900); transition:background .2s ease; }}
        #MainMenu, footer {{ visibility:hidden; }}

        /* ---------------- Header / toolbar ----------------
           Streamlit renders a fixed top header bar (the strip that holds
           the "Deploy" button / hamburger menu) with its own opaque white
           background by default. It sits above .block-container and does
           NOT pick up .stApp's background, so on a dark canvas it shows up
           as a stray white bar above the hero section. Force it (and the
           toolbar container inside it) to match the current canvas color
           instead of leaving it hard-coded white. */
        header[data-testid="stHeader"]{{
            background:var(--canvas) !important;
            background-color:var(--canvas) !important;
        }}
        div[data-testid="stToolbar"]{{
            background:transparent !important;
        }}

        .block-container{{
            padding-top:1.2rem;
            padding-bottom:3rem;
            max-width:1500px;
        }}

        /* ---------------- Sidebar ---------------- */
        section[data-testid="stSidebar"]{{
            background:linear-gradient(180deg,#0b1120 0%,#0f1830 100%);
            border-right:1px solid rgba(255,255,255,.06);
        }}
        section[data-testid="stSidebar"] *{{ color:#e6e9f5 !important; }}
        section[data-testid="stSidebar"] hr{{ border-color:rgba(255,255,255,.08); }}

        .sb-brand{{
            display:flex; align-items:center; gap:11px;
            padding:2px 0 20px 0;
        }}
        .sb-brand .mark{{
            width:40px; height:40px; border-radius:12px;
            background:linear-gradient(135deg,var(--brand) 0%,#9b6df0 100%);
            display:flex; align-items:center; justify-content:center;
            font-size:19px; box-shadow:0 6px 16px -6px rgba(109,123,245,.6);
        }}
        .sb-brand .name{{ font-weight:800; font-size:17px; line-height:1.15; }}
        .sb-brand .sub{{ font-size:10.5px; color:#7d88a8 !important; letter-spacing:.09em; font-weight:600; }}

        .sb-section-label{{
            font-size:10.5px; text-transform:uppercase; letter-spacing:.09em;
            color:#5c6785 !important; font-weight:700; margin:20px 0 9px 0;
        }}
        .sb-chip{{
            display:inline-block; padding:3.5px 11px; border-radius:999px;
            font-size:11.5px; font-weight:700; margin-bottom:6px; letter-spacing:.01em;
        }}
        .sb-chip.ok{{ background:rgba(45,212,191,.16); color:#5eead4 !important; }}
        .sb-chip.info{{ background:rgba(109,123,245,.22); color:#b4bdfb !important; }}
        .sb-kv{{ font-size:12.5px; color:#aab3cf !important; margin:3px 0; line-height:1.5; }}
        .sb-kv b{{ color:#eef0fa !important; }}

        /* ---------------- Pill nav (styled radio) ---------------- */
        section[data-testid="stSidebar"] div[role="radiogroup"]{{
            display:flex; flex-direction:column; gap:4px; margin-top:2px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label{{
            background:transparent;
            border:1px solid transparent;
            border-radius:11px;
            padding:9px 12px !important;
            margin:0 !important;
            cursor:pointer;
            transition:background .12s ease, border-color .12s ease;
            display:flex; align-items:center; gap:8px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{{
            background:rgba(255,255,255,.05);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){{
            background:linear-gradient(120deg,var(--brand) 0%,#8b6df0 100%);
            border-color:transparent;
            box-shadow:0 6px 16px -8px rgba(109,123,245,.7);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{{
            color:#ffffff !important; font-weight:700 !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{{
            display:none;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label p{{
            font-size:14px !important; font-weight:600 !important; margin:0 !important;
        }}

        /* ---------------- Top bar ---------------- */
        .topbar{{ display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }}
        .topbar .crumbs{{ font-size:12.5px; color:var(--ink-500); font-weight:600; }}

        /* ---------------- Hero ---------------- */
        .hero{{
            position:relative; overflow:hidden;
            background:linear-gradient(120deg,#4046d6 0%,#6440d0 55%,#8b3fc9 100%);
            border-radius:24px;
            padding:36px 40px;
            color:white;
            margin-bottom:22px;
            box-shadow:0 20px 44px -20px rgba(64,70,214,.5);
        }}
        .hero::after{{
            content:""; position:absolute; top:-40%; right:-8%; width:320px; height:320px;
            background:radial-gradient(circle,rgba(255,255,255,.18) 0%,rgba(255,255,255,0) 70%);
            border-radius:50%;
        }}
        .hero h1{{ margin:0; font-size:31px; font-weight:900; letter-spacing:-.02em; position:relative; }}
        .hero p{{ margin:9px 0 0 0; opacity:.92; font-size:14.5px; max-width:640px; position:relative; }}
        .hero .tag{{
            display:inline-block; background:rgba(255,255,255,.16);
            backdrop-filter:blur(6px);
            padding:5px 13px; border-radius:999px; font-size:12px;
            font-weight:700; margin-top:15px; letter-spacing:.02em; position:relative;
        }}

        /* ---------------- Cards ---------------- */
        .card{{
            background:var(--surface);
            border-radius:18px;
            padding:22px 24px;
            border:1px solid var(--line);
            box-shadow:var(--shadow);
            margin-bottom:18px;
            transition:box-shadow .15s ease, border-color .15s ease;
        }}
        .card:hover{{ border-color:var(--brand); }}
        .card h3{{ margin-top:0; font-size:16px; font-weight:800; }}
        .card-title-row{{ display:flex; align-items:center; justify-content:space-between; }}

        .section-title{{
            font-size:21px; font-weight:900; margin:6px 0 4px 0; letter-spacing:-.015em;
        }}
        .section-sub{{ color:var(--ink-500); font-size:13.5px; margin-bottom:14px; }}

        /* ---------------- KPI cards ---------------- */
        .kpi{{
            background:var(--surface);
            border-radius:16px;
            padding:18px 20px;
            border:1px solid var(--line);
            box-shadow:var(--shadow);
            height:100%;
            transition:transform .12s ease, border-color .12s ease;
        }}
        .kpi:hover{{ transform:translateY(-2px); border-color:var(--brand); }}
        .kpi.gold{{ background:linear-gradient(160deg,var(--surface) 60%,var(--brand-soft) 140%); }}
        .kpi .icon{{
            width:34px; height:34px; border-radius:10px; background:var(--brand-soft);
            display:flex; align-items:center; justify-content:center; font-size:16px;
        }}
        .kpi .val{{ font-size:25px; font-weight:900; margin-top:10px; letter-spacing:-.02em; }}
        .kpi .lbl{{ font-size:12px; color:var(--ink-500); font-weight:600; margin-top:2px; }}

        /* ---------------- Job overview rows ---------------- */
        .jd-row{{ display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px dashed var(--line); font-size:14px; }}
        .jd-row:last-child{{ border-bottom:none; }}
        .jd-row .k{{ color:var(--ink-500); font-weight:600; }}
        .jd-row .v{{ font-weight:700; text-align:right; max-width:60%; }}
        .skill-pill{{
            display:inline-block; background:var(--brand-soft); color:var(--brand-dark);
            border:1px solid var(--line); padding:3.5px 11px; border-radius:999px;
            font-size:11.5px; font-weight:700; margin:3px 4px 0 0;
        }}

        /* ---------------- Status badges ---------------- */
        .badge{{
            display:inline-block; padding:4px 11px; border-radius:999px;
            font-size:12px; font-weight:800; white-space:nowrap; letter-spacing:.01em;
        }}
        .badge-excellent{{ background:rgba(14,159,110,.15); color:#0e9f6e; }}
        .badge-strong{{ background:rgba(64,70,214,.15); color:var(--brand-dark); }}
        .badge-good{{ background:rgba(196,127,10,.15); color:var(--amber); }}
        .badge-average{{ background:rgba(234,88,12,.15); color:#ea580c; }}
        .badge-weak{{ background:rgba(214,69,80,.15); color:var(--rose); }}

        /* ---------------- Buttons ---------------- */
        .stButton button{{
            border-radius:12px; height:46px; font-weight:700;
            border:1px solid var(--line); background:var(--surface); color:var(--ink-900);
            transition:transform .1s ease;
        }}
        .stButton button:hover{{ transform:translateY(-1px); border-color:var(--brand); }}
        .stButton button[kind="primary"]{{
            background:linear-gradient(120deg,var(--brand),#8b3fc9);
            border:none; color:#fff;
            box-shadow:0 10px 24px -12px rgba(109,70,214,.6);
        }}

        /* ---------------- Download button ----------------
           st.download_button renders as data-testid="stDownloadButton",
           which is a DIFFERENT component from st.button's "stButton" — so
           none of the .stButton rules above ever reached it. Left
           unstyled, it kept Streamlit's default white background with
           light text, which is invisible against a dark canvas. Mirror
           the .stButton styling here explicitly, including its inner <p>
           label (Streamlit wraps the button text in a <p> tag). */
        [data-testid="stDownloadButton"] button{{
            border-radius:12px; height:46px; font-weight:700;
            border:1px solid var(--line); background:var(--surface); color:var(--ink-900) !important;
            transition:transform .1s ease;
        }}
        [data-testid="stDownloadButton"] button:hover{{
            transform:translateY(-1px); border-color:var(--brand);
        }}
        [data-testid="stDownloadButton"] button p{{
            color:var(--ink-900) !important;
        }}

        div[data-testid="stMetric"]{{
            background:var(--surface); border-radius:14px; padding:16px;
            border:1px solid var(--line); box-shadow:var(--shadow);
        }}
        div[data-testid="stMetricLabel"] {{ color:var(--ink-500) !important; }}

        [data-testid="stDataFrame"]{{ border-radius:14px; overflow:hidden; border:1px solid var(--line); }}

        .stProgress > div > div{{ border-radius:10px; background:linear-gradient(90deg,var(--brand),#8b3fc9); }}

        .empty-state{{
            text-align:center; padding:60px 20px; color:var(--ink-500);
            background:var(--surface); border:1px dashed var(--line); border-radius:18px;
        }}
        .empty-state .big{{ font-size:40px; margin-bottom:10px; }}

        .pill-row{{ display:flex; flex-wrap:wrap; gap:6px; }}
        .concern-ok{{ color:var(--teal); font-weight:700; }}

        .ranked-table{{ width:100%; border-collapse:collapse; font-size:13.5px; }}
        .ranked-table th{{
            text-align:left; padding:11px 12px; background:var(--surface-alt);
            border-bottom:1px solid var(--line); color:var(--ink-500);
            font-size:11px; text-transform:uppercase; letter-spacing:.05em; font-weight:700;
        }}
        .ranked-table td{{ padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:middle; color:var(--ink-900); }}
        .ranked-table tr:hover td{{ background:var(--surface-alt); }}

        input, textarea, select {{ color:var(--ink-900) !important; }}
        [data-baseweb="select"] > div {{ background:var(--surface) !important; border-color:var(--line) !important; }}

        /* ---------------- File uploader ----------------
           Streamlit's file uploader ships its own light-theme colors on
           the dropzone, icon, helper text, and "Browse files" button that
           don't follow our CSS variables by default — they'd disappear
           against a dark canvas. Force every part of it explicitly. */
        [data-testid="stFileUploader"] section {{
            background:var(--surface-alt) !important;
            border:1px dashed var(--line) !important;
            border-radius:12px !important;
        }}
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] section span,
        [data-testid="stFileUploader"] section small,
        [data-testid="stFileUploader"] section div {{
            color:var(--ink-900) !important;
        }}
        [data-testid="stFileUploader"] section svg {{
            fill:var(--ink-500) !important;
        }}
        [data-testid="stFileUploader"] button {{
            background:var(--surface) !important;
            color:var(--ink-900) !important;
            border:1px solid var(--line) !important;
        }}

        /* ---------------- Sidebar buttons ----------------
           The sidebar background is always dark navy regardless of the
           app-wide theme toggle, so its buttons need their own fixed
           light-on-dark styling instead of following --surface (which
           turns white in light mode and would be unreadable here). */
        section[data-testid="stSidebar"] .stButton button {{
            background:rgba(255,255,255,.06) !important;
            border:1px solid rgba(255,255,255,.16) !important;
            color:#e6e9f5 !important;
        }}
        section[data-testid="stSidebar"] .stButton button:hover {{
            background:rgba(255,255,255,.12) !important;
            border-color:var(--brand) !important;
        }}
        section[data-testid="stSidebar"] .stButton button:disabled {{
            opacity:.45 !important;
        }}

        /* ---------------- "Use Sample Data" toggle ----------------
           Scoped to .block-container (main content) so the sidebar's own
           Dark Mode toggle is left untouched. :has() targets the div that
           directly wraps the checkbox input, i.e. the switch track itself,
           rather than the whole label row (which also contains the text). */
        .block-container [data-testid="stToggle"] div:has(> input[type="checkbox"]) {{
            border:2px solid var(--gold) !important;
            border-radius:999px !important;
            box-shadow:0 0 0 3px rgba(184,134,45,.12);
        }}

        div[data-testid="stFileUploaderDropzone"] {{ background:var(--surface-alt) !important; border-color:var(--line) !important; }}

        /* ---------------- Text visibility in both themes ----------------
           Streamlit's own headings/labels/metrics ship with a fixed color
           that does NOT follow our --ink-900 variable, so they'd go
           near-invisible against a dark canvas (or too faint on light).
           Scoped to .block-container so the sidebar's own light-on-dark
           styling above is left untouched. */
        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4,
        .block-container h5,
        .block-container h6,
        .block-container p,
        .block-container li,
        .block-container label,
        [data-testid="stMarkdownContainer"] {{ color:var(--ink-900) !important; }}

        [data-testid="stCaptionContainer"],
        .block-container small {{ color:var(--ink-500) !important; }}

        div[data-testid="stMetricValue"] {{ color:var(--ink-900) !important; }}
        div[data-testid="stMetricDelta"] {{ color:var(--ink-500) !important; }}

        [data-testid="stDataFrame"] * {{ color:var(--ink-900) !important; }}

        .stAlert p {{ color:inherit !important; }}

        </style>
        """
    )


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

@st.cache_data(show_spinner=False)
def load_sample_candidates() -> list:
    sample_path = Path(__file__).parent / "data" / "raw" / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf8") as f:
            return json.load(f)
    return []


@st.cache_data(show_spinner=False)
def parse_uploaded_bytes(content: bytes, filename: str) -> list:
    text = content.decode("utf-8")
    if filename.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def run_ranking(candidates: list) -> tuple:
    """Score every candidate via the untouched backend and time the run."""
    start = time.perf_counter()
    scored = []
    progress = st.progress(0, text="Scoring candidates...")
    total = len(candidates)
    for i, c in enumerate(candidates):
        scored.append(score_candidate_for_jd(c))
        if total:
            progress.progress((i + 1) / total, text=f"Scoring candidates... {i + 1}/{total}")
    progress.empty()

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for i, c in enumerate(scored):
        c["rank"] = i + 1

    elapsed = time.perf_counter() - start
    return scored, elapsed


# ============================================================================
# SMALL VISUAL HELPERS
# ============================================================================

def get_status(match_pct: float) -> tuple:
    """Return (label, css_class) for a match percentage."""
    if match_pct >= 90:
        return "Excellent", "badge-excellent"
    if match_pct >= 75:
        return "Strong", "badge-strong"
    if match_pct >= 60:
        return "Good", "badge-good"
    if match_pct >= 40:
        return "Average", "badge-average"
    return "Weak", "badge-weak"


def badge_html(match_pct: float) -> str:
    label, css = get_status(match_pct)
    return f'<span class="badge {css}">{label}</span>'


def kpi_card(col, icon: str, label: str, value: str, gold: bool = False) -> None:
    with col:
        html_block(
            f"""
            <div class="kpi {'gold' if gold else ''}">
                <div class="icon">{icon}</div>
                <div class="val">{value}</div>
                <div class="lbl">{label}</div>
            </div>
            """
        )


def render_kpi_row(scored: list, runtime_seconds: float) -> None:
    average_score = sum(c["score"] for c in scored) / len(scored)
    honeypots = sum(1 for c in scored if c["score"] <= 0.01)
    top_score = max(c["score"] for c in scored)

    cols = st.columns(6, gap="medium")
    kpi_card(cols[0], "👥", "Candidates Evaluated", f"{len(scored)}")
    kpi_card(cols[1], "🏆", "Best Match", f"{top_score * 100:.1f}%", gold=True)
    kpi_card(cols[2], "📊", "Average Match", f"{average_score * 100:.1f}%")
    kpi_card(cols[3], "🚩", "Honeypots Filtered", f"{honeypots}")
    kpi_card(cols[4], "⚡", "Runtime", f"{runtime_seconds:.2f}s")
    kpi_card(cols[5], "🖥️", "Compute", "CPU Mode")


def build_table(scored: list) -> pd.DataFrame:
    rows = []
    for c in scored:
        b = c["_breakdown"]
        match_pct = round(c["score"] * 100, 1)
        label, _ = get_status(match_pct)
        rows.append(
            {
                "Rank": c["rank"],
                "Candidate ID": c["candidate_id"],
                "Match %": match_pct,
                "Status": label,
                "Experience": b["years"],
                "AI Skills": b["ai_skill_count"],
                "Location": b["location"],
                "Reasoning": c["reasoning"],
            }
        )
    return pd.DataFrame(rows)


def to_ranked_csv(scored: list) -> str:
    """Untouched CSV shape: candidate_id, rank, score, reasoning."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for c in scored:
        writer.writerow([c["candidate_id"], c["rank"], c["score"], c["reasoning"]])
    return output.getvalue()


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar() -> tuple:
    """Everything here runs inside a single `with st.sidebar:` block so plain
    st.* / html_block() calls are correctly routed into the sidebar instead
    of leaking into the main content area."""
    with st.sidebar:
        html_block(
            """
            <div class="sb-brand">
                <div class="mark">🧭</div>
                <div>
                    <div class="name">Redrob AI Recruiter</div>
                    <div class="sub">CANDIDATE INTELLIGENCE</div>
                </div>
            </div>
            """
        )

        html_block('<div class="sb-section-label">Navigation</div>')
        page = st.radio(
            "Navigation",
            ["🏠  Dashboard", "👥  Candidates", "📊  Analytics", "⚙️  Settings"],
            label_visibility="collapsed",
        )

        html_block('<div class="sb-section-label">Appearance</div>')
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))
        st.session_state["dark_mode"] = dark_mode

        html_block('<div class="sb-section-label">Job Summary</div>')
        html_block(
            f"""
            <span class="sb-chip info">{JOB_ROLE}</span><br>
            <div class="sb-kv"><b>Experience:</b> {JOB_EXPERIENCE}</div>
            <div class="sb-kv"><b>Location:</b> {JOB_LOCATION}</div>
            """
        )

        html_block('<div class="sb-section-label">Runtime</div>')
        html_block(
            """
            <span class="sb-chip ok">CPU Only</span><br>
            <div class="sb-kv">No GPU · No hosted LLM</div>
            <div class="sb-kv">Deterministic &amp; reproducible</div>
            """
        )

        html_block('<div class="sb-section-label">Explainability</div>')
        html_block('<span class="sb-chip ok">Enabled</span>')

        st.divider()
        if st.button("↺  Reset Ranking", use_container_width=True):
            for key in ("scored", "runtime", "dataset_label"):
                st.session_state.pop(key, None)
            st.rerun()

    return page, dark_mode


# ============================================================================
# HEADER + UPLOAD
# ============================================================================

def render_hero() -> None:
    html_block(
        """
        <div class="hero">
            <h1>🧭 AI Recruiter Dashboard</h1>
            <p>Explainable, deterministic candidate ranking for the Redrob Intelligent
            Candidate Discovery Challenge — built for recruiters, not just engineers.</p>
            <span class="tag">CPU-only · Deterministic · Fully explainable</span>
        </div>
        """
    )


def render_job_overview(dataset_label: str) -> None:
    left, right = st.columns([2, 1], gap="large")

    with left:
        skill_pills = "".join(f'<span class="skill-pill">{s}</span>' for s in JOB_SKILLS)
        html_block(
            f"""
            <div class="card">
                <h3>📋 Job Overview</h3>
                <div class="jd-row"><span class="k">Role</span><span class="v">{JOB_ROLE}</span></div>
                <div class="jd-row"><span class="k">Experience</span><span class="v">{JOB_EXPERIENCE}</span></div>
                <div class="jd-row"><span class="k">Ranking Engine</span><span class="v">{RANKING_ENGINE}</span></div>
                <div class="jd-row"><span class="k">Compute Mode</span><span class="v">{COMPUTE_MODE}</span></div>
                <div class="jd-row"><span class="k">Dataset</span><span class="v">{dataset_label}</span></div>
                <div style="margin-top:12px;">
                    <span style="font-size:12.5px;color:var(--ink-500);font-weight:700;">REQUIRED SKILLS</span>
                    <div class="pill-row" style="margin-top:7px;">{skill_pills}</div>
                </div>
            </div>
            """
        )

    with right:
        st.metric("Experience Band", JOB_EXPERIENCE)
        st.metric("Preferred Location", "India")
        st.metric("Ranking Engine", "Rule Based")


def render_upload_controls() -> tuple:
    html_block('<div class="section-title">Candidate Data</div>')
    html_block(
        '<div class="section-sub">Upload a candidate dataset or use the bundled sample to begin ranking.</div>'
    )
    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader("Upload Candidate Dataset", type=["json", "jsonl"])
    with sample_col:
        html_block("<div style='height:28px'></div>")
        use_sample = st.toggle("Use Sample Data")
    return uploaded_file, use_sample


# ============================================================================
# DASHBOARD PAGE
# ============================================================================

def render_dashboard_page(scored: list) -> None:
    html_block(
        """
        <div class="card">
            <h3>🏆 Top Candidate Recommendations</h3>
            <div class="section-sub" style="margin-bottom:0;">
                Highest ranked candidates against the current job description.
            </div>
        </div>
        """
    )

    table = build_table(scored).head(15).copy()
    table["Status"] = table["Match %"].apply(badge_html)

    # NOTE: st.write() does not reliably honor unsafe_allow_html, which is
    # what caused the raw <table>...</table> markup to print as plain text.
    # html_block() (st.markdown + unsafe_allow_html=True) renders it properly.
    html_block(table.to_html(escape=False, index=False, classes="ranked-table"))

    st.download_button(
        "⬇ Download Ranked CSV",
        to_ranked_csv(scored),
        "sandbox_ranked_output.csv",
        "text/csv",
        use_container_width=True,
    )


# ============================================================================
# CANDIDATES PAGE
# ============================================================================

def apply_filters(scored: list, search: str, min_score: int, min_exp: float,
                   locations: list, min_ai_skills: int, min_response: float,
                   max_notice: int) -> list:
    filtered = []
    query = search.lower().strip()
    for c in scored:
        b = c["_breakdown"]
        if c["score"] * 100 < min_score:
            continue
        if b["years"] < min_exp:
            continue
        if locations and b["location"] not in locations:
            continue
        if b["ai_skill_count"] < min_ai_skills:
            continue
        if b["response_rate"] < min_response:
            continue
        if b["notice_days"] > max_notice:
            continue
        if query:
            haystack = " ".join(
                [
                    c["candidate_id"].lower(),
                    b.get("location", "").lower(),
                    b.get("current_title", "").lower(),
                    " ".join(b.get("matched_ai_skills", [])).lower(),
                    str(b.get("years", "")),
                ]
            )
            if query not in haystack:
                continue
        filtered.append(c)
    return filtered


def render_score_breakdown(b: dict) -> None:
    items = [
        ("AI Skills", b["ai_skills_score"]),
        ("Title Match", b["title_score"]),
        ("Experience", b["exp_score"]),
        ("Availability", b["availability_score"]),
        ("Company", b["company_score"]),
        ("Location", b["location_score"]),
        ("Platform", b["platform_score"]),
        ("Notice Period", b["notice_score"]),
    ]
    for label, score in items:
        st.markdown(f"**{label}**")
        st.progress(min(max(float(score), 0.0), 1.0))
        st.caption(f"{score:.2f}")


def generate_recruiter_summary(b: dict) -> str:
    """Deterministic, template-based summary from the existing score breakdown.
    No LLM calls — purely derived from numbers already computed by the backend."""
    lines = []

    total = (
        b["ai_skills_score"] + b["title_score"] + b["exp_score"]
        + b["availability_score"] + b["company_score"] + b["location_score"]
    ) / 6

    if total >= 0.85:
        lines.append("Excellent fit for the role.")
    elif total >= 0.7:
        lines.append("Strong candidate for the role.")
    elif total >= 0.5:
        lines.append("Reasonable fit with some gaps.")
    else:
        lines.append("Limited alignment with role requirements.")

    if b["ai_skill_count"] >= 5:
        lines.append("Strong AI/ML skill coverage.")
    if b["exp_score"] >= 0.8:
        lines.append("Experience closely matches the JD's target range.")
    if b["availability_score"] >= 0.7:
        lines.append("High recruiter responsiveness.")
    if b["location_score"] >= 0.8:
        lines.append("Suitable hiring location.")
    if b.get("is_consulting_only"):
        lines.append("Career history is consulting-only — verify hands-on production exposure.")

    return " ".join(lines)


def render_candidate_profile(candidate: dict) -> None:
    b = candidate["_breakdown"]
    match_pct = candidate["score"] * 100
    label, css = get_status(match_pct)

    html_block(
        f"""
        <div class="card">
            <div class="card-title-row">
                <h3>{candidate['candidate_id']}</h3>
                <span class="badge {css}">{label}</span>
            </div>
        </div>
        """
    )

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.metric("Overall Match", f"{match_pct:.1f}%")
        st.metric("Rank", candidate["rank"])
        st.metric("Current Title", b.get("current_title", "—"))
        st.metric("Experience", f"{b['years']} yrs")
        st.metric("Location", b.get("location", "—"))
        st.metric("Recruiter Response", f"{b['response_rate']*100:.0f}%")
        st.metric("Platform Score", f"{b['platform_score']:.2f}")
        st.metric("Notice Period", f"{b['notice_days']} days")

    with right:
        st.markdown("#### Recruiter Summary")
        st.success(generate_recruiter_summary(b))

        st.markdown("#### Matched AI Skills")
        if b["matched_ai_skills"]:
            pills = "".join(f'<span class="skill-pill">{s}</span>' for s in b["matched_ai_skills"])
            html_block(f'<div class="pill-row">{pills}</div>')
        else:
            st.info("No matched AI skills.")

        st.markdown("#### Reasoning")
        st.write(candidate["reasoning"])

    st.divider()
    strength_col, concern_col = st.columns(2, gap="large")

    with strength_col:
        st.markdown("#### ✅ Strengths")
        found = False
        if b["ai_skill_count"] >= 5:
            st.success("Strong AI skill coverage")
            found = True
        if b["exp_score"] >= 0.8:
            st.success("Experience aligns well with JD")
            found = True
        if b["availability_score"] >= 0.7:
            st.success("High recruiter responsiveness")
            found = True
        if b["company_score"] >= 0.7:
            st.success("Strong company background")
            found = True
        if b["location_score"] >= 0.8:
            st.success("Preferred hiring location")
            found = True
        if not found:
            st.info("No standout strengths identified.")

    with concern_col:
        st.markdown("#### ⚠️ Potential Concerns")
        concern = False
        if b["notice_score"] < 0.6:
            concern = True
            st.warning("Long notice period")
        if b["platform_score"] < 0.5:
            concern = True
            st.warning("Low platform engagement")
        if b["availability_score"] < 0.5:
            concern = True
            st.warning("Low recruiter response")
        if b["ai_skill_count"] < 4:
            concern = True
            st.warning("Limited AI skills")
        if b.get("is_consulting_only"):
            concern = True
            st.warning("Consulting-only career background")
        if not concern:
            html_block('<span class="concern-ok">No major concerns identified</span>')

    st.divider()
    st.markdown("#### Score Breakdown")
    render_score_breakdown(b)


def render_candidates_page(scored: list) -> None:
    html_block('<div class="section-title">Candidate Explorer</div>')
    html_block(
        '<div class="section-sub">Search, filter, sort, and drill into individual candidate profiles.</div>'
    )

    with st.container(border=True):
        search = st.text_input(
            "Search",
            placeholder="Search by candidate ID, location, skill, title, or experience",
        )

        f1, f2, f3 = st.columns(3)
        min_score = f1.slider("Minimum Match %", 0, 100, 0)
        max_years = max((c["_breakdown"]["years"] for c in scored), default=20)
        min_exp = f2.slider("Minimum Experience (yrs)", 0.0, float(max(max_years, 1)), 0.0)
        min_ai_skills = f3.slider("Minimum AI Skill Count", 0, 15, 0)

        f4, f5, f6 = st.columns(3)
        all_locations = sorted({c["_breakdown"]["location"] for c in scored if c["_breakdown"]["location"]})
        locations = f4.multiselect("Location", all_locations)
        min_response = f5.slider("Minimum Recruiter Response Rate", 0.0, 1.0, 0.0, step=0.05)
        max_notice_available = max((c["_breakdown"]["notice_days"] for c in scored), default=90)
        max_notice = f6.slider(
            "Maximum Notice Period (days)", 0, int(max(max_notice_available, 1)), int(max(max_notice_available, 1))
        )

    filtered = apply_filters(scored, search, min_score, min_exp, locations, min_ai_skills, min_response, max_notice)

    if not filtered:
        html_block(
            """
            <div class="empty-state">
                <div class="big">🔍</div>
                <b>No candidates match the selected filters.</b><br>
                Try loosening a filter or clearing the search box.
            </div>
            """
        )
        return

    st.caption(f"Showing {len(filtered)} of {len(scored)} candidates")

    table = build_table(filtered)

    page_size = 15
    total_pages = max(1, (len(table) + page_size - 1) // page_size)
    page_num = st.session_state.get("candidates_page", 1)
    page_num = min(page_num, total_pages)

    nav_l, nav_c, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("⬅ Prev", disabled=page_num <= 1):
            page_num -= 1
    with nav_r:
        if st.button("Next ➡", disabled=page_num >= total_pages):
            page_num += 1
    with nav_c:
        html_block(
            f"<div style='text-align:center;padding-top:8px;color:var(--ink-500);'>Page {page_num} of {total_pages}</div>"
        )
    st.session_state["candidates_page"] = page_num

    start = (page_num - 1) * page_size
    page_table = table.iloc[start:start + page_size]

    st.dataframe(
        page_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match %": st.column_config.ProgressColumn(
                "Match %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

    st.divider()
    st.markdown("### Candidate Profile")
    ids = [c["candidate_id"] for c in filtered]
    selected_id = st.selectbox("Select a candidate to view their full profile", ids)
    candidate = next(c for c in filtered if c["candidate_id"] == selected_id)
    render_candidate_profile(candidate)


# ============================================================================
# ANALYTICS PAGE
# ============================================================================

PLOTLY_TEMPLATE_LIGHT = "plotly_white"
PLOTLY_TEMPLATE_DARK = "plotly_dark"
BRAND_COLORS = ["#4046d6", "#0e9f8e", "#c47f0a", "#d64550", "#8b3fc9", "#0891b2"]


def render_analytics_page(scored: list, dark: bool) -> None:
    template = PLOTLY_TEMPLATE_DARK if dark else PLOTLY_TEMPLATE_LIGHT
    paper_bg = "rgba(0,0,0,0)"

    html_block('<div class="section-title">Recruitment Analytics</div>')
    html_block(
        '<div class="section-sub">Distribution and composition insights across the evaluated candidate pool.</div>'
    )

    df = pd.DataFrame(
        [
            {
                "Candidate": c["candidate_id"],
                "Match %": round(c["score"] * 100, 1),
                "Status": get_status(round(c["score"] * 100, 1))[0],
                "Experience": c["_breakdown"]["years"],
                "AI Skills": c["_breakdown"]["ai_skill_count"],
                "Location": c["_breakdown"]["location"],
                "Response Rate": c["_breakdown"]["response_rate"],
                "Title": c["_breakdown"].get("current_title", "Unknown"),
                "Consulting Only": c["_breakdown"].get("is_consulting_only", False),
                "Honeypot": c["score"] <= 0.01,
            }
            for c in scored
        ]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Match", f"{df['Match %'].mean():.1f}%")
    c2.metric("Average Experience", f"{df['Experience'].mean():.1f} yrs")
    c3.metric("Average AI Skills", f"{df['AI Skills'].mean():.1f}")
    c4.metric("Honeypots", int(df["Honeypot"].sum()))

    st.divider()

    def layout(fig, height=320):
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=height,
            paper_bgcolor=paper_bg, plot_bgcolor=paper_bg,
        )
        return fig

    left, right = st.columns(2)
    with left:
        st.markdown("##### Match Score Distribution")
        fig = px.histogram(df, x="Match %", nbins=20, template=template,
                            color_discrete_sequence=[BRAND_COLORS[0]])
        st.plotly_chart(layout(fig), use_container_width=True)

    with right:
        st.markdown("##### Experience Distribution")
        fig = px.histogram(df, x="Experience", nbins=15, template=template,
                            color_discrete_sequence=[BRAND_COLORS[1]])
        st.plotly_chart(layout(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("##### AI Skill Count")
        skill_counts = df["AI Skills"].value_counts().sort_index()
        fig = px.bar(x=skill_counts.index, y=skill_counts.values, template=template,
                     labels={"x": "AI Skill Count", "y": "Candidates"},
                     color_discrete_sequence=[BRAND_COLORS[2]])
        st.plotly_chart(layout(fig), use_container_width=True)

    with right:
        st.markdown("##### Recruiter Response Rate")
        fig = px.histogram(df, x="Response Rate", nbins=15, template=template,
                            color_discrete_sequence=[BRAND_COLORS[3]])
        st.plotly_chart(layout(fig), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Top Candidate Locations")
        loc_counts = df["Location"].value_counts().head(10).sort_values()
        fig = px.bar(x=loc_counts.values, y=loc_counts.index, orientation="h",
                     template=template, labels={"x": "Candidates", "y": ""},
                     color_discrete_sequence=[BRAND_COLORS[4]])
        st.plotly_chart(layout(fig, 340), use_container_width=True)

    with right:
        st.markdown("##### Top Titles")
        title_counts = df["Title"].value_counts().head(10).sort_values()
        fig = px.bar(x=title_counts.values, y=title_counts.index, orientation="h",
                     template=template, labels={"x": "Candidates", "y": ""},
                     color_discrete_sequence=[BRAND_COLORS[5]])
        st.plotly_chart(layout(fig, 340), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Top AI Skills (Matched)")
        all_skills = []
        for c in scored:
            all_skills.extend(c["_breakdown"].get("matched_ai_skills", []))
        if all_skills:
            skill_series = pd.Series(all_skills).value_counts().head(10).sort_values()
            fig = px.bar(x=skill_series.values, y=skill_series.index, orientation="h",
                         template=template, labels={"x": "Occurrences", "y": ""},
                         color_discrete_sequence=[BRAND_COLORS[0]])
            st.plotly_chart(layout(fig, 340), use_container_width=True)
        else:
            st.info("No matched AI skills available to chart.")

    with right:
        st.markdown("##### Honeypot vs. Valid Candidates")
        honeypot_counts = df["Honeypot"].value_counts()
        labels = ["Honeypot" if v else "Valid" for v in honeypot_counts.index]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=honeypot_counts.values,
                    hole=0.55,
                    marker=dict(colors=[BRAND_COLORS[3], BRAND_COLORS[1]]),
                )
            ]
        )
        fig.update_layout(template=template)
        st.plotly_chart(layout(fig, 340), use_container_width=True)

    st.markdown("##### Experience vs. Match Score")
    fig = px.scatter(
        df, x="Experience", y="Match %", color="Status", hover_data=["Candidate", "Title"],
        template=template, color_discrete_sequence=BRAND_COLORS,
    )
    st.plotly_chart(layout(fig, 380), use_container_width=True)


# ============================================================================
# SETTINGS PAGE
# ============================================================================

def render_settings_page(dataset_label: str, runtime_seconds: float) -> None:
    html_block('<div class="section-title">Settings</div>')
    html_block(
        '<div class="section-sub">Ranking engine configuration and hackathon compliance details.</div>'
    )

    left, right = st.columns(2, gap="large")

    with left:
        html_block(
            f"""
            <div class="card">
                <h3>⚙️ Ranking Engine</h3>
                <div class="jd-row"><span class="k">Engine</span><span class="v">{RANKING_ENGINE}</span></div>
                <div class="jd-row"><span class="k">Compute Mode</span><span class="v">{COMPUTE_MODE}</span></div>
                <div class="jd-row"><span class="k">Determinism</span><span class="v">Fully Deterministic</span></div>
                <div class="jd-row"><span class="k">Explainability</span><span class="v">Per-component score breakdown</span></div>
                <div class="jd-row"><span class="k">Last Runtime</span><span class="v">{runtime_seconds:.2f}s</span></div>
                <div class="jd-row"><span class="k">Dataset</span><span class="v">{dataset_label}</span></div>
            </div>
            """
        )

    with right:
        html_block(
            """
            <div class="card">
                <h3>✅ Hackathon Compliance</h3>
                <div class="jd-row"><span class="k">CPU Only</span><span class="v">Yes</span></div>
                <div class="jd-row"><span class="k">No Hosted LLM</span><span class="v">Yes</span></div>
                <div class="jd-row"><span class="k">No GPU</span><span class="v">Yes</span></div>
                <div class="jd-row"><span class="k">Deterministic</span><span class="v">Yes</span></div>
                <div class="jd-row"><span class="k">Explainable</span><span class="v">Yes</span></div>
                <div class="jd-row"><span class="k">Reproducible</span><span class="v">Yes</span></div>
            </div>
            """
        )

    html_block(
        """
        <div class="card">
            <h3>ℹ️ About This Dashboard</h3>
            <p style="color:var(--ink-500);font-size:14px;">
            This is a frontend-only redesign of the AI Recruiter Dashboard for the Redrob
            Intelligent Candidate Discovery &amp; Ranking Hackathon. All scoring, honeypot
            detection, and ranking logic is untouched and lives in
            <code>src/data_loader.py</code> and <code>src/runner.py</code>. The recruiter
            summary shown on each candidate profile is generated deterministically from the
            existing score breakdown — no language model is called at any point.
            </p>
        </div>
        """
    )

    if st.button("🗑️ Clear Cached Data", use_container_width=False):
        st.cache_data.clear()
        st.success("Cache cleared.")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    # Render the sidebar FIRST so we know the current, up-to-date value of
    # the dark-mode toggle (Streamlit updates session_state for a widget as
    # soon as it's drawn on a rerun). CSS is then injected once, using that
    # fresh value, so the theme switch applies immediately instead of
    # lagging one click behind.
    page, dark_mode = render_sidebar()
    inject_css(dark_mode)

    render_hero()

    dataset_label = st.session_state.get("dataset_label", "No dataset loaded")

    with st.container(border=True):
        render_job_overview(dataset_label)

    uploaded_file, use_sample = render_upload_controls()

    candidates = None
    if use_sample:
        candidates = load_sample_candidates()
        if candidates:
            st.success(f"Loaded {len(candidates)} sample candidates")
            st.session_state["dataset_label"] = f"Sample Dataset ({len(candidates)} candidates)"
        else:
            st.warning("Sample dataset not found on disk.")
    elif uploaded_file is not None:
        try:
            candidates = parse_uploaded_bytes(uploaded_file.getvalue(), uploaded_file.name)
            st.success(f"Loaded {len(candidates)} uploaded candidates")
            st.session_state["dataset_label"] = f"{uploaded_file.name} ({len(candidates)} candidates)"
        except Exception as e:
            st.error(f"Could not parse uploaded file: {e}")

    if candidates:
        if len(candidates) > 100:
            st.warning("Only first 100 candidates are evaluated in sandbox mode.")
            candidates = candidates[:100]

        if st.button("🚀 Run AI Ranking", type="primary", use_container_width=True):
            with st.spinner("Ranking candidates..."):
                scored, elapsed = run_ranking(candidates)
                st.session_state["scored"] = scored
                st.session_state["runtime"] = elapsed
                st.session_state["candidates_page"] = 1

    st.divider()

    if "scored" in st.session_state:
        scored = st.session_state["scored"]
        runtime_seconds = st.session_state.get("runtime", 0.0)

        render_kpi_row(scored, runtime_seconds)
        st.divider()

        if page.strip() == "🏠  Dashboard":
            render_dashboard_page(scored)
        elif page.strip() == "👥  Candidates":
            render_candidates_page(scored)
        elif page.strip() == "📊  Analytics":
            render_analytics_page(scored, dark_mode)
        elif page.strip() == "⚙️  Settings":
            render_settings_page(dataset_label, runtime_seconds)
    else:
        html_block(
            """
            <div class="empty-state">
                <div class="big">📂</div>
                <b>Upload a candidate dataset or use the sample data to begin.</b><br>
                Then press <b>Run AI Ranking</b> to score and rank candidates.
            </div>
            """
        )

    st.divider()
    st.caption("AI Recruiter Dashboard • Redrob Intelligent Candidate Discovery Challenge")


if __name__ == "__main__":
    main()