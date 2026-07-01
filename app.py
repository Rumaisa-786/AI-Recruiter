"""
Gradio sandbox app for hackathon submission requirement.
Accepts a small candidate sample, runs the ranking logic end-to-end,
and produces a downloadable ranked CSV — same scoring logic as
src/data_loader.py, no LLM calls, runs entirely on CPU.
"""

import json
import csv
import io
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import score_candidate_for_jd


def load_candidates_from_file(file_path: str):
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    data = json.loads(content)
    return data if isinstance(data, list) else [data]


def load_bundled_sample():
    sample_path = Path(__file__).parent / "data" / "raw" / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def run_ranking(file, use_bundled_sample):
    if use_bundled_sample:
        candidates = load_bundled_sample()
        if not candidates:
            return None, None, "Bundled sample file not found.", None
    elif file is not None:
        try:
            candidates = load_candidates_from_file(file)
        except Exception as e:
            return None, None, f"Could not parse file: {e}", None
    else:
        return None, None, "Upload a file or check the bundled sample box.", None

    if len(candidates) > 100:
        candidates = candidates[:100]
        truncated_note = f" (truncated from larger input to 100 per sandbox limits)"
    else:
        truncated_note = ""

    scored = [score_candidate_for_jd(c) for c in candidates]
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for rank, c in enumerate(scored):
        c["rank"] = rank + 1

    df = pd.DataFrame([
        {
            "rank": c["rank"],
            "candidate_id": c["candidate_id"],
            "score": c["score"],
            "reasoning": c["reasoning"],
        }
        for c in scored
    ])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for c in scored:
        writer.writerow([c["candidate_id"], c["rank"], c["score"], c["reasoning"]])

    csv_path = Path("sandbox_ranked_output.csv")
    csv_path.write_text(output.getvalue(), encoding="utf-8")

    flagged = sum(1 for c in scored if c["score"] <= 0.01)
    summary = (
        f"Ranked {len(scored)} candidates{truncated_note}. "
        f"Top score: {scored[0]['score']:.3f}. "
        f"Flagged as honeypot: {flagged}."
    )

    return df, str(csv_path), summary, None


with gr.Blocks(title="AI candidate ranker — sandbox") as demo:
    gr.Markdown("# AI candidate ranker — sandbox")
    gr.Markdown(
        "Reproducibility sandbox for the Redrob hackathon. Upload a small "
        "candidate sample (up to 100 records, JSON or JSONL) and run the same "
        "scoring logic that produces the full submission. No GPU, no hosted "
        "LLM calls — pure Python over the fields already present in each record."
    )

    with gr.Row():
        file_input = gr.File(label="Upload candidate sample (.json or .jsonl)", file_types=[".json", ".jsonl"])
        use_sample = gr.Checkbox(label="Use bundled sample_candidates.json instead", value=False)

    run_button = gr.Button("Run ranking", variant="primary")

    summary_box = gr.Textbox(label="Summary", interactive=False)
    results_table = gr.Dataframe(label="Ranked results", interactive=False, wrap=True)
    download_file = gr.File(label="Download ranked CSV")

    run_button.click(
        fn=run_ranking,
        inputs=[file_input, use_sample],
        outputs=[results_table, download_file, summary_box, gr.State()],
    )

    gr.Markdown(
        "No GPU and no hosted LLM API calls are used in this scoring step. "
        "See README.md for full architecture details."
    )

if __name__ == "__main__":
    demo.launch()