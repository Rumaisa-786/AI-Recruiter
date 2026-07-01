"""
Sandbox app for hackathon submission requirement.
Accepts a small candidate sample, runs the ranking logic end-to-end,
and produces a downloadable ranked CSV — same scoring logic as
src/data_loader.py, no LLM calls, runs entirely on CPU.
"""

import streamlit as st
import json
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import score_candidate_for_jd

st.set_page_config(page_title="AI Candidate Ranker — Sandbox", page_icon="🎯", layout="wide")

st.title("AI candidate ranker — sandbox")
st.caption(
    "Reproducibility sandbox for the Redrob hackathon. Upload a small candidate "
    "sample (up to 100 records) and run the same scoring logic that produces "
    "the full submission. No GPU, no hosted LLM calls — pure Python over the "
    "fields already present in each record."
)

st.divider()

col_upload, col_options = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload candidate sample (.json or .jsonl)",
        type=["json", "jsonl"],
        key="sandbox_uploader",
    )

with col_options:
    st.write("")
    st.write("")
    use_sample_data = st.checkbox("Use bundled sample_candidates.json", value=False, key="use_sample_checkbox")


def load_candidates_from_upload(file):
    content = file.read().decode("utf-8")
    if file.name.endswith(".jsonl"):
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    data = json.loads(content)
    return data if isinstance(data, list) else [data]


def load_bundled_sample():
    sample_path = Path(__file__).parent / "data" / "raw" / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


candidates = None

if use_sample_data:
    candidates = load_bundled_sample()
    if candidates:
        st.info(f"Loaded {len(candidates)} candidates from the bundled sample file.")
elif uploaded_file is not None:
    try:
        candidates = load_candidates_from_upload(uploaded_file)
        st.success(f"Loaded {len(candidates)} candidates from upload.")
    except Exception as e:
        st.error(f"Could not parse file: {e}")

if candidates:
    if len(candidates) > 100:
        st.warning(f"Input has {len(candidates)} candidates — truncating to the first 100 per sandbox limits.")
        candidates = candidates[:100]

    run_clicked = st.button("Run ranking", type="primary", key="run_ranking_button")

    if run_clicked:
        with st.spinner("Scoring candidates..."):
            scored = [score_candidate_for_jd(c) for c in candidates]
            scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            for rank, c in enumerate(scored):
                c["rank"] = rank + 1

        st.session_state["scored"] = scored

    if "scored" in st.session_state:
        scored = st.session_state["scored"]

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates ranked", len(scored))
        m2.metric("Top score", f"{scored[0]['score']:.3f}")
        m3.metric("Median score", f"{scored[len(scored)//2]['score']:.3f}")
        flagged = sum(1 for c in scored if c["score"] <= 0.01)
        m4.metric("Flagged as honeypot", flagged)

        st.subheader("Ranked results")

        display_rows = [
            {
                "rank": c["rank"],
                "candidate_id": c["candidate_id"],
                "score": c["score"],
                "reasoning": c["reasoning"],
            }
            for c in scored
        ]

        st.dataframe(
            display_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                "score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
        )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for c in scored:
            writer.writerow([c["candidate_id"], c["rank"], c["score"], c["reasoning"]])

        st.download_button(
            label="Download ranked CSV",
            data=output.getvalue(),
            file_name="sandbox_ranked_output.csv",
            mime="text/csv",
            type="primary",
            key="download_csv_button",
        )
else:
    st.info("Upload a candidate file or check the bundled sample box above to begin.")

st.divider()
st.caption("No GPU and no hosted LLM API calls are used in this scoring step. See README.md for full architecture details.")