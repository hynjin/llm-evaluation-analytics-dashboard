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


def generate_key_insights(dataframe: pd.DataFrame) -> list[str]:
    """Create concise business findings from the currently filtered data."""
    insights = []

    model_performance = (
        dataframe.groupby("model")
        .agg(
            average_score=("final_score", "mean"),
            hallucination_rate=("hallucination_flag", "mean"),
        )
        .sort_values("average_score", ascending=False)
    )
    best_model = model_performance.iloc[0]
    insights.append(
        f"**Top model:** {model_performance.index[0]} leads with a "
        f"{best_model['average_score']:.2f}/5 average final score and a "
        f"{best_model['hallucination_rate']:.1%} hallucination rate."
    )

    if len(model_performance) > 1:
        worst_model = model_performance.iloc[-1]
        score_gap = best_model["average_score"] - worst_model["average_score"]
        insights.append(
            f"**Model improvement opportunity:** {model_performance.index[-1]} has the "
            f"lowest average score at {worst_model['average_score']:.2f}/5, trailing "
            f"the leading model by {score_gap:.2f} points."
        )
    else:
        insights.append(
            f"**Model scope:** {model_performance.index[0]} is the only model in the "
            "current view, so peer performance comparisons are unavailable."
        )

    category_risk = (
        dataframe.groupby("category")["hallucination_flag"].mean().sort_values(ascending=False)
    )
    insights.append(
        f"**Hallucination hotspot:** {category_risk.index[0]} has the highest "
        f"hallucination rate at {category_risk.iloc[0]:.1%}; prioritize this category "
        "for targeted error review."
    )

    language_performance = (
        dataframe.groupby("language")["final_score"].mean().sort_values(ascending=False)
    )
    if len(language_performance) > 1:
        language_gap = language_performance.iloc[0] - language_performance.iloc[-1]
        insights.append(
            f"**Language quality gap:** {language_performance.index[0]} performs best "
            f"at {language_performance.iloc[0]:.2f}/5, while "
            f"{language_performance.index[-1]} averages "
            f"{language_performance.iloc[-1]:.2f}/5, a {language_gap:.2f}-point gap."
        )
    else:
        insights.append(
            f"**Language performance:** {language_performance.index[0]} averages "
            f"{language_performance.iloc[0]:.2f}/5 in the current filtered view."
        )

    prompt_performance = (
        dataframe.groupby("prompt_type")["final_score"].mean().sort_values()
    )
    insights.append(
        f"**Hardest prompt type:** {prompt_performance.index[0]} prompts produce the "
        f"lowest average final score at {prompt_performance.iloc[0]:.2f}/5, suggesting "
        "a useful focus area for model testing."
    )

    annotator_consistency = (
        dataframe.groupby("annotator_id")
        .agg(
            score_std_dev=("final_score", "std"),
            evaluations=("evaluation_id", "count"),
        )
        .query("evaluations >= 2")
        .dropna()
        .sort_values("score_std_dev")
    )
    if not annotator_consistency.empty:
        most_consistent = annotator_consistency.iloc[0]
        insights.append(
            f"**Annotator consistency:** {annotator_consistency.index[0]} has the "
            f"lowest score variation ({most_consistent['score_std_dev']:.2f} standard "
            f"deviation) across {int(most_consistent['evaluations'])} evaluations."
        )
    else:
        insights.append(
            "**Annotator consistency:** The current filters do not leave enough "
            "repeated evaluations per annotator for a reliable consistency comparison."
        )

    return insights


def generate_recommended_actions(dataframe: pd.DataFrame) -> list[str]:
    """Generate metric-backed actions from the currently filtered data."""
    recommendations = []
    overall_hallucination_rate = dataframe["hallucination_flag"].mean()

    risk_candidates = []
    dimension_labels = {
        "model": "model",
        "language": "language",
        "category": "category",
        "prompt_type": "prompt type",
    }
    minimum_group_size = max(5, round(len(dataframe) * 0.03))
    for dimension, label in dimension_labels.items():
        if dataframe[dimension].nunique() < 2:
            continue
        risk_summary = dataframe.groupby(dimension).agg(
            hallucination_rate=("hallucination_flag", "mean"),
            evaluations=("evaluation_id", "count"),
        )
        risk_summary = risk_summary.query("evaluations >= @minimum_group_size")
        if risk_summary.empty:
            continue
        highest_risk = risk_summary["hallucination_rate"].idxmax()
        highest_rate = risk_summary.loc[highest_risk, "hallucination_rate"]
        risk_candidates.append(
            (highest_rate - overall_hallucination_rate, highest_rate, label, highest_risk)
        )

    if risk_candidates:
        excess_rate, highest_rate, label, segment = max(risk_candidates)
        if excess_rate >= 0.03 or highest_rate >= 0.15:
            recommendations.append(
                f"Prioritize hallucination audits for **{segment}** ({label}) because "
                f"its {highest_rate:.1%} rate exceeds the filtered dataset average of "
                f"{overall_hallucination_rate:.1%}."
            )

    quality_candidates = []
    for dimension, label in {"model": "model", "category": "category"}.items():
        if dataframe[dimension].nunique() < 2:
            continue
        quality_summary = dataframe.groupby(dimension)["final_score"].mean().sort_values()
        score_gap = quality_summary.iloc[-1] - quality_summary.iloc[0]
        quality_candidates.append(
            (score_gap, label, quality_summary.index[0], quality_summary.iloc[0])
        )

    if quality_candidates:
        score_gap, label, segment, lowest_score = max(quality_candidates)
        if score_gap >= 0.15:
            recommendations.append(
                f"Run targeted error analysis on **{segment}** ({label}); its "
                f"{lowest_score:.2f}/5 average final score trails the best-performing "
                f"peer by {score_gap:.2f} points."
            )

    annotator_summary = (
        dataframe.groupby("annotator_id")
        .agg(
            score_std_dev=("final_score", "std"),
            evaluations=("evaluation_id", "count"),
        )
        .query("evaluations >= 5")
        .dropna()
    )
    if len(annotator_summary) >= 2:
        team_median_variance = annotator_summary["score_std_dev"].median()
        highest_variance_annotator = annotator_summary["score_std_dev"].idxmax()
        highest_variance = annotator_summary.loc[
            highest_variance_annotator, "score_std_dev"
        ]
        variance_gap = highest_variance - team_median_variance
        if variance_gap >= 0.10:
            recommendations.append(
                f"Review calibration for **{highest_variance_annotator}**; score "
                f"variation is {highest_variance:.2f}, compared with the annotator "
                f"median of {team_median_variance:.2f}."
            )

    coverage_candidates = []
    for dimension, label in {"category": "category", "prompt_type": "prompt type"}.items():
        if dataframe[dimension].nunique() < 2:
            continue
        coverage = dataframe[dimension].value_counts(normalize=True).sort_values()
        expected_share = 1 / len(coverage)
        lowest_share = coverage.iloc[0]
        if lowest_share < expected_share * 0.80:
            coverage_candidates.append(
                (expected_share - lowest_share, label, coverage.index[0], lowest_share)
            )

    if coverage_candidates:
        _, label, segment, lowest_share = max(coverage_candidates)
        recommendations.append(
            f"Increase evaluation coverage for **{segment}** ({label}); it represents "
            f"only {lowest_share:.1%} of the current filtered dataset."
        )

    if dataframe["language"].nunique() >= 2:
        language_scores = dataframe.groupby("language")["final_score"].mean().sort_values()
        language_gap = language_scores.iloc[-1] - language_scores.iloc[0]
        if language_gap >= 0.10:
            recommendations.append(
                f"Investigate language-specific quality issues for "
                f"**{language_scores.index[0]}**; its {language_scores.iloc[0]:.2f}/5 "
                f"average score trails the leading language by {language_gap:.2f} points."
            )

    quality_metrics = dataframe[
        ["helpfulness_score", "accuracy_score", "safety_score"]
    ].mean()
    lowest_metric = quality_metrics.idxmin()
    metric_label = lowest_metric.replace("_score", "").title()
    fallback_actions = [
        (
            f"Prioritize {metric_label.lower()} error analysis; **{metric_label}** is "
            f"the weakest quality dimension at {quality_metrics[lowest_metric]:.2f}/5."
        ),
        (
            f"Review score dispersion across the **{len(dataframe):,} filtered "
            f"evaluations**; final scores have a standard deviation of "
            f"{dataframe['final_score'].std(ddof=0):.2f}."
        ),
    ]

    hallucination_count = int(dataframe["hallucination_flag"].sum())
    if hallucination_count:
        response_label = "response" if hallucination_count == 1 else "responses"
        fallback_actions.append(
            f"Sample and review the **{hallucination_count} hallucinated "
            f"{response_label}** "
            f"in this view ({overall_hallucination_rate:.1%} of evaluations) to identify "
            "repeat failure patterns."
        )

    low_score_count = int((dataframe["final_score"] < 3.5).sum())
    low_score_rate = low_score_count / len(dataframe)
    if low_score_count:
        fallback_actions.append(
            f"Conduct root-cause analysis on the **{low_score_count} evaluations below "
            f"3.5/5**, which account for {low_score_rate:.1%} of the filtered sample."
        )

    for action in fallback_actions:
        if len(recommendations) >= 3:
            break
        recommendations.append(action)

    if len(recommendations) < 3:
        recommendations.append(
            f"Expand the current evaluation slice beyond its **{len(dataframe):,} "
            "records** with additional edge cases before making broader quality decisions."
        )

    return recommendations[:5]


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

st.subheader("Automated Key Insights")
st.caption("Business-oriented findings generated from the current filtered view.")
for insight in generate_key_insights(filtered):
    st.markdown(f"- {insight}")

st.subheader("Recommended Actions")
st.caption("Rule-based next steps generated from the current filtered metrics.")
for recommendation in generate_recommended_actions(filtered):
    st.markdown(f"- {recommendation}")

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
