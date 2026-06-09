"""Streamlit dashboard for exploring synthetic LLM evaluation results."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "llm_evaluation_data.csv"
CHART_COLORS = ["#2563EB", "#0F766E", "#7C3AED", "#EA580C", "#DC2626"]


st.set_page_config(
    page_title="LLM Evaluation Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return dataframe


def apply_chart_style(figure, percent_axis: bool = False):
    figure.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    figure.update_traces(marker_line_width=0)
    if percent_axis:
        figure.update_yaxes(tickformat=".0%")
    return figure


def multiselect_filter(label: str, values: pd.Series) -> list[str]:
    options = sorted(values.unique())
    return st.sidebar.multiselect(label, options, default=options)


data = load_data()

st.title("LLM Evaluation Analytics Dashboard")
st.caption(
    "Explore synthetic model evaluation results across quality, hallucination, "
    "language, category, and annotator consistency."
)

st.sidebar.header("Filters")
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(data["date"].min().date(), data["date"].max().date()),
    min_value=data["date"].min().date(),
    max_value=data["date"].max().date(),
)
selected_models = multiselect_filter("Model", data["model"])
selected_languages = multiselect_filter("Language", data["language"])
selected_categories = multiselect_filter("Category", data["category"])
selected_prompt_types = multiselect_filter("Prompt type", data["prompt_type"])

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates[0]

filtered = data[
    data["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    & data["model"].isin(selected_models)
    & data["language"].isin(selected_languages)
    & data["category"].isin(selected_categories)
    & data["prompt_type"].isin(selected_prompt_types)
].copy()

st.sidebar.caption(f"{len(filtered):,} of {len(data):,} evaluations shown")

if filtered.empty:
    st.warning("No evaluations match the selected filters. Expand the filters to continue.")
    st.stop()

st.subheader("Overview")
metric_columns = st.columns(5)
metric_columns[0].metric("Total evaluations", f"{len(filtered):,}")
metric_columns[1].metric("Average final score", f"{filtered['final_score'].mean():.2f} / 5")
metric_columns[2].metric("Hallucination rate", f"{filtered['hallucination_flag'].mean():.1%}")
metric_columns[3].metric("Average accuracy", f"{filtered['accuracy_score'].mean():.2f} / 5")
metric_columns[4].metric("Average safety", f"{filtered['safety_score'].mean():.2f} / 5")

st.divider()
st.subheader("Model Performance")
model_summary = (
    filtered.groupby("model", as_index=False)
    .agg(
        average_final_score=("final_score", "mean"),
        average_accuracy=("accuracy_score", "mean"),
        hallucination_rate=("hallucination_flag", "mean"),
        evaluations=("evaluation_id", "count"),
    )
    .sort_values("average_final_score", ascending=False)
)

model_left, model_right = st.columns(2)
with model_left:
    model_score_chart = px.bar(
        model_summary,
        x="model",
        y=["average_final_score", "average_accuracy"],
        barmode="group",
        title="Average quality scores by model",
        labels={"value": "Average score (1-5)", "model": "Model", "variable": "Metric"},
        color_discrete_sequence=CHART_COLORS,
    )
    st.plotly_chart(apply_chart_style(model_score_chart), width="stretch")

with model_right:
    model_hallucination_chart = px.bar(
        model_summary.sort_values("hallucination_rate"),
        x="model",
        y="hallucination_rate",
        title="Hallucination rate by model",
        labels={"hallucination_rate": "Hallucination rate", "model": "Model"},
        color="hallucination_rate",
        color_continuous_scale="Blues",
    )
    model_hallucination_chart.update_layout(coloraxis_showscale=False)
    st.plotly_chart(
        apply_chart_style(model_hallucination_chart, percent_axis=True),
        width="stretch",
    )

st.divider()
st.subheader("Language Analysis")
language_summary = (
    filtered.groupby("language", as_index=False)
    .agg(
        average_final_score=("final_score", "mean"),
        hallucination_rate=("hallucination_flag", "mean"),
        evaluations=("evaluation_id", "count"),
    )
    .sort_values("average_final_score", ascending=False)
)

language_columns = st.columns(3)
language_score_chart = px.bar(
    language_summary,
    x="language",
    y="average_final_score",
    title="Average final score",
    labels={"average_final_score": "Average score (1-5)", "language": "Language"},
    color_discrete_sequence=[CHART_COLORS[0]],
)
language_columns[0].plotly_chart(
    apply_chart_style(language_score_chart), width="stretch"
)

language_hallucination_chart = px.bar(
    language_summary,
    x="language",
    y="hallucination_rate",
    title="Hallucination rate",
    labels={"hallucination_rate": "Hallucination rate", "language": "Language"},
    color_discrete_sequence=[CHART_COLORS[4]],
)
language_columns[1].plotly_chart(
    apply_chart_style(language_hallucination_chart, percent_axis=True),
    width="stretch",
)

language_count_chart = px.bar(
    language_summary,
    x="language",
    y="evaluations",
    title="Evaluation volume",
    labels={"evaluations": "Evaluations", "language": "Language"},
    color_discrete_sequence=[CHART_COLORS[1]],
)
language_columns[2].plotly_chart(
    apply_chart_style(language_count_chart), width="stretch"
)

st.divider()
st.subheader("Category Analysis")
category_summary = (
    filtered.groupby("category", as_index=False)
    .agg(
        average_final_score=("final_score", "mean"),
        hallucination_rate=("hallucination_flag", "mean"),
        evaluations=("evaluation_id", "count"),
    )
    .sort_values("average_final_score")
)

category_left, category_right = st.columns([2, 1])
with category_left:
    category_chart = px.bar(
        category_summary,
        x="average_final_score",
        y="category",
        orientation="h",
        title="Average final score by category",
        labels={"average_final_score": "Average score (1-5)", "category": "Category"},
        color="hallucination_rate",
        color_continuous_scale="RdYlBu_r",
        hover_data={"hallucination_rate": ":.1%", "evaluations": True},
    )
    category_chart.update_layout(coloraxis_colorbar_title="Hallucination")
    st.plotly_chart(apply_chart_style(category_chart), width="stretch")

with category_right:
    lowest_category = category_summary.iloc[0]
    highest_hallucination = category_summary.loc[
        category_summary["hallucination_rate"].idxmax()
    ]
    st.markdown("#### Key signals")
    st.metric(
        "Lowest-performing category",
        lowest_category["category"],
        f"{lowest_category['average_final_score']:.2f} average score",
        delta_color="off",
    )
    st.metric(
        "Highest hallucination category",
        highest_hallucination["category"],
        f"{highest_hallucination['hallucination_rate']:.1%} rate",
        delta_color="off",
    )
    st.caption(
        "Use these signals to prioritize evaluation coverage and targeted quality reviews."
    )

st.divider()
st.subheader("Annotator Quality Analysis")
st.caption(
    "Synthetic annotator statistics demonstrate workload and consistency monitoring; "
    "they are not performance judgments about real people."
)
annotator_summary = (
    filtered.groupby("annotator_id", as_index=False)
    .agg(
        average_final_score=("final_score", "mean"),
        evaluations=("evaluation_id", "count"),
        score_std_dev=("final_score", "std"),
    )
    .fillna({"score_std_dev": 0})
)

annotator_chart = px.scatter(
    annotator_summary,
    x="evaluations",
    y="average_final_score",
    size="score_std_dev",
    color="score_std_dev",
    text="annotator_id",
    title="Annotator workload and score consistency",
    labels={
        "evaluations": "Number of evaluations",
        "average_final_score": "Average final score",
        "score_std_dev": "Score std. dev.",
    },
    color_continuous_scale="Viridis",
    size_max=28,
)
annotator_chart.update_traces(textposition="top center")
st.plotly_chart(apply_chart_style(annotator_chart), width="stretch")

display_annotators = annotator_summary.sort_values(
    ["score_std_dev", "evaluations"], ascending=[True, False]
).rename(
    columns={
        "annotator_id": "Annotator",
        "average_final_score": "Average score",
        "evaluations": "Evaluations",
        "score_std_dev": "Score std. dev.",
    }
)
st.dataframe(
    display_annotators,
    width="stretch",
    hide_index=True,
    column_config={
        "Average score": st.column_config.NumberColumn(format="%.2f"),
        "Score std. dev.": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.caption(
    "Data is synthetic and generated locally. No external APIs, user data, or paid services are used."
)
