# LLM Evaluation Analytics Dashboard

A lightweight Streamlit dashboard for exploring synthetic LLM evaluation data across model quality, hallucination patterns, languages, task categories, and annotator consistency.

## Why This Project

LLM evaluation teams need a clear view of where model quality changes and where review effort should be focused. This portfolio project demonstrates how structured AI evaluation data can be transformed into business-friendly quality signals using Python, Pandas, Plotly, and Streamlit.

## Dashboard Features

- Overview KPIs for evaluation volume, final score, accuracy, safety, and hallucination rate
- Model performance comparison
- Language-level quality and coverage analysis
- Category-level risk and performance signals
- Annotator workload and score consistency monitoring
- Shared date, model, language, category, and prompt-type filters

All records are synthetic. The project uses no external APIs, databases, authentication, or paid services.

## Project Structure

```text
llm-evaluation-analytics-dashboard/
├── app.py
├── generate_data.py
├── PRD.md
├── data/
│   └── llm_evaluation_data.csv
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

Open `http://localhost:8501` in a browser.

## Dataset

`generate_data.py` creates 1,500 reproducible evaluation records with:

- Four models: Gemini, GPT, Claude, and Llama
- English, Korean, and Japanese evaluations
- Five evaluation categories and four prompt types
- Helpfulness, accuracy, safety, hallucination, and final-score metrics
- Twelve synthetic annotators

The generator includes modest model, language, category, and prompt difficulty effects so the dashboard reveals meaningful patterns rather than uniformly random results.

## Portfolio Positioning

> Built an LLM evaluation analytics dashboard using Python, Pandas, Plotly, and Streamlit to analyze model quality trends, hallucination patterns, language-level performance, and annotator consistency across synthetic AI evaluation data.

## Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
