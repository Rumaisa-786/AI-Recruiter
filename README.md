# 🎯 AI Recruiter Dashboard
### Explainable AI Candidate Discovery & Ranking System

An AI-powered recruiter dashboard built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. The system evaluates a large pool of candidate profiles, ranks them against a target job description using an explainable scoring engine, detects inconsistent (honeypot) profiles, and presents recruiter-friendly insights through an interactive dashboard.

The solution is **fully deterministic, CPU-only, and optimized for large-scale candidate ranking**, making it compliant with the hackathon's runtime and resource constraints.

---

## ✨ Key Features

- 🎯 Explainable AI candidate ranking
- 🧠 Multi-factor deterministic scoring engine
- 🚩 Automatic honeypot profile detection
- 📊 Interactive recruiter dashboard
- 📈 Candidate analytics and score breakdowns
- 🔍 Recruiter-friendly candidate inspection
- 📥 Export ranked candidates as CSV
- ⚡ Optimized for 100,000+ candidate datasets
- 💻 CPU-only execution (no GPU required)
- 🔒 Fully reproducible results

---

# 📸 Dashboard

The application provides an intuitive recruiter interface that enables users to:

- Upload candidate datasets
- Rank candidates against the target job description
- Inspect individual candidate profiles
- View detailed scoring explanations
- Analyze candidate distributions
- Export ranked results

> *(Add dashboard screenshots here after deployment.)*

---

# 🏗️ System Architecture

```
Candidate Dataset
        │
        ▼
Data Loading
        │
        ▼
Feature Extraction
        │
        ▼
AI Skill Matching
        │
        ▼
Title Relevance
        │
        ▼
Experience Analysis
        │
        ▼
Availability Assessment
        │
        ▼
Company Quality
        │
        ▼
Location Matching
        │
        ▼
Honeypot Detection
        │
        ▼
Weighted Scoring Engine
        │
        ▼
Candidate Ranking
        │
        ▼
Dashboard & CSV Export
```

---

# 📂 Repository Structure

```
ai-recruiter/
│
├── data/
│   ├── raw/
│   │   ├── candidates.jsonl
│   │   └── sample_candidates.json
│   ├── jobs/
│   │   └── job_description.docx
│   └── candidate_schema.json
│
├── src/
│   ├── data_loader.py
│   ├── runner.py
│   ├── explainability/
│   ├── matching/
│   ├── ranking/
│   ├── scoring/
│   └── understanding/
│
├── outputs/
│
├── sandbox_app.py
├── validate_submission.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Candidate Scoring Methodology

Each candidate receives a composite score based on multiple recruiter-centric signals.

| Component | Weight | Purpose |
|-----------|--------|---------|
| AI / ML Skill Match | **40%** | Matches AI skills and semantic experience with the job requirements |
| Title Relevance | **25%** | Measures alignment between current role and target position |
| Experience Fit | **18%** | Rewards candidates within the desired experience range |
| Recruiter Availability | **12%** | Considers recruiter response rate, activity, and openness |
| Company Quality | **3%** | Rewards relevant product-company experience |
| Location Match | **2%** | Prioritizes preferred hiring locations |

Additional business rules ensure that unsuitable profiles are appropriately down-ranked.

---

# 🚩 Honeypot Detection

The dataset intentionally contains inconsistent candidate profiles.

The system detects anomalies such as:

- Expert-level skills with unrealistically low experience
- Total experience inconsistent with employment history
- Invalid platform metrics
- Role durations exceeding claimed experience

Profiles flagged as honeypots are assigned a minimal score to prevent them from appearing among the top-ranked candidates.

---

# 🔍 Explainable AI

Every ranked candidate includes a recruiter-friendly explanation describing why they received their score.

Example:

> **Senior Machine Learning Engineer** with strong AI skill coverage, relevant production experience, and high recruiter responsiveness.

Recruiters can inspect:

- Overall Match Score
- AI Skill Match
- Experience Fit
- Title Match
- Availability
- Company Quality
- Location Preference
- Platform Signals
- Final Recommendation

---

# 📊 Dashboard Features

The Streamlit dashboard includes:

### 📈 Dashboard

- KPI overview
- Candidate leaderboard
- CSV export

### 👤 Candidate Explorer

- Individual candidate inspection
- Recruiter summary
- Strengths & concerns
- Detailed score breakdown
- Matched AI skills

### 📉 Analytics

- Match score distribution
- Experience distribution
- AI skill statistics
- Recruiter response analysis
- Candidate location analysis

---

# ⚡ Performance

| Metric | Value |
|--------|-------|
| Candidate Dataset | 100,000 Profiles |
| Execution Mode | CPU Only |
| GPU Required | ❌ No |
| Hosted LLM Required | ❌ No |
| Deterministic | ✅ Yes |
| Explainable | ✅ Yes |
| Submission Format | CSV |

The solution is designed to meet the hackathon's compute and reproducibility requirements while maintaining efficient execution.

---

# 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- JSON
- CSV

---

# 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd ai-recruiter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Ranking Engine

Generate the ranked submission:

```bash
python -m src.runner
```

Validate the generated CSV:

```bash
python validate_submission.py outputs/submission.csv
```

---

# 🖥️ Launch the Dashboard

```bash
streamlit run sandbox_app.py
```

---

# 🧪 Design Evolution

During development, two approaches were explored:

### Hybrid Retrieval Prototype

- Dense retrieval
- BM25 retrieval
- Cross-encoder re-ranking
- LLM-assisted evaluation

While accurate, this approach exceeded the hackathon's runtime and compute constraints.

### Final Submission Pipeline

The final solution distills the same recruiter reasoning into a lightweight deterministic scoring engine operating entirely on structured candidate data.

Benefits include:

- CPU-only execution
- Deterministic outputs
- Explainable recommendations
- Fast processing
- Fully reproducible rankings
- Hackathon-compliant resource usage

---

# 🎯 Project Highlights

- Explainable AI candidate ranking
- Recruiter-centric dashboard
- Deterministic weighted scoring
- Automatic honeypot detection
- Interactive candidate analytics
- CSV submission generation
- Large-scale candidate processing
- Clean, modular architecture

---

# 👨‍💻 Developed For

**Redrob Intelligent Candidate Discovery & Ranking Challenge**

This project demonstrates how explainable AI can support recruiters by producing transparent, scalable, and reproducible candidate rankings while respecting practical deployment constraints.