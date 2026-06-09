# PRD: LLM Evaluation Analytics Dashboard

## 1. Project Overview

Build a lightweight analytics dashboard for LLM evaluation data.

The goal of this project is to simulate how AI annotation and model evaluation data can be analyzed to identify quality trends, hallucination patterns, model performance differences, and annotator consistency.

This project is designed as a portfolio project for AI Data Analytics / AI Evaluation / Applied Data roles.

## 2. Objective

The dashboard should show that the developer can:

- Analyze structured AI evaluation data
- Use Python and Pandas for data processing
- Create clear visualizations for model quality metrics
- Identify patterns in hallucination, safety, accuracy, and helpfulness
- Present insights in a simple, business-friendly dashboard

## 3. Target Users

Primary users:

- AI data quality analysts
- AI evaluation team members
- Annotation operation managers
- Recruiters or hiring managers reviewing the portfolio

## 4. Tech Stack

Use only free and lightweight tools:

- Python
- Pandas
- Plotly
- Streamlit
- CSV dataset

Do not use:

- Paid APIs
- OpenAI API
- Database
- Backend server
- Authentication
- Cloud infrastructure

## 5. Dataset

Use a synthetic CSV dataset.

The dataset should represent LLM evaluation records.

Required columns:

- evaluation_id
- date
- model
- language
- category
- prompt_type
- annotator_id
- helpfulness_score
- accuracy_score
- safety_score
- hallucination_flag
- final_score

Example values:

- model: Gemini, GPT, Claude, Llama
- language: English, Korean, Japanese
- category: Coding, General QA, Summarization, Reasoning, Safety
- prompt_type: Direct, Multi-turn, Adversarial, Instruction-following
- hallucination_flag: 0 or 1
- scores: 1 to 5

The project should include a script to generate synthetic data, for example `generate_data.py`.

## 6. Core Features

### 6.1 Overview Page

Show key metrics:

- Total evaluations
- Average final score
- Hallucination rate
- Average accuracy score
- Average safety score

### 6.2 Model Performance Analysis

Show model-level comparison:

- Average final score by model
- Average accuracy by model
- Hallucination rate by model

Include at least one bar chart.

### 6.3 Language Analysis

Show language-level quality trends:

- Average final score by language
- Hallucination rate by language
- Evaluation count by language

### 6.4 Category Analysis

Show category-level patterns:

- Average final score by category
- Hallucination rate by category
- Lowest-performing categories

### 6.5 Annotator Quality Analysis

Show annotator-level consistency:

- Average score by annotator
- Number of evaluations per annotator
- Score variance or standard deviation by annotator

The goal is not to judge real people, but to demonstrate how annotation consistency can be monitored.

### 6.6 Filters

Add sidebar filters:

- Date range
- Model
- Language
- Category
- Prompt type

All charts and metrics should update based on selected filters.

## 7. Non-Goals

Do not build:

- User login
- Database storage
- Real API integration
- Machine learning model training
- RAG features
- Complex deployment pipeline

The project should stay simple and focused on analytics.

## 8. Deliverables

The final project should include:

- Streamlit dashboard
- Synthetic CSV dataset
- Data generation script
- README.md
- Requirements file
- Screenshots for portfolio use

## 9. Suggested File Structure

```text
llm-evaluation-analytics-dashboard/
├── app.py
├── generate_data.py
├── data/
│   └── llm_evaluation_data.csv
├── requirements.txt
├── README.md
└── screenshots/
```

## 10. Success Criteria

The project is successful if:

- The app runs locally with one command
- The dashboard clearly shows AI evaluation trends
- The README explains the business context
- The project can be understood by a recruiter within 1 minute
- The project can be added to a resume as an AI data analytics project

## 11. Resume Positioning

Possible resume bullet:

> Built an LLM evaluation analytics dashboard using Python, Pandas, Plotly, and Streamlit to analyze model quality trends, hallucination patterns, language-level performance, and annotator consistency across synthetic AI evaluation data.
