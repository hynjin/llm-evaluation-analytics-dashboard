"""Generate a reproducible synthetic LLM evaluation dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).parent / "data" / "llm_evaluation_data.csv"
RECORD_COUNT = 1_500
RANDOM_SEED = 42


def clipped_score(values: np.ndarray) -> np.ndarray:
    """Round continuous quality values to a realistic 1-5 score."""
    return np.clip(np.rint(values), 1, 5).astype(int)


def generate_dataset(record_count: int = RECORD_COUNT) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    models = np.array(["Gemini", "GPT", "Claude", "Llama"])
    languages = np.array(["English", "Korean", "Japanese"])
    categories = np.array(
        ["Coding", "General QA", "Summarization", "Reasoning", "Safety"]
    )
    prompt_types = np.array(
        ["Direct", "Multi-turn", "Adversarial", "Instruction-following"]
    )
    annotators = np.array([f"ANN-{number:02d}" for number in range(1, 13)])

    model = rng.choice(models, record_count, p=[0.25, 0.28, 0.25, 0.22])
    language = rng.choice(languages, record_count, p=[0.55, 0.25, 0.20])
    category = rng.choice(categories, record_count)
    prompt_type = rng.choice(
        prompt_types, record_count, p=[0.35, 0.25, 0.18, 0.22]
    )
    annotator_id = rng.choice(annotators, record_count)

    model_effect = pd.Series(model).map(
        {"GPT": 0.30, "Claude": 0.24, "Gemini": 0.12, "Llama": -0.15}
    ).to_numpy()
    language_effect = pd.Series(language).map(
        {"English": 0.15, "Korean": -0.03, "Japanese": -0.08}
    ).to_numpy()
    category_effect = pd.Series(category).map(
        {
            "Coding": -0.05,
            "General QA": 0.12,
            "Summarization": 0.18,
            "Reasoning": -0.20,
            "Safety": 0.06,
        }
    ).to_numpy()
    prompt_effect = pd.Series(prompt_type).map(
        {
            "Direct": 0.16,
            "Multi-turn": -0.08,
            "Adversarial": -0.32,
            "Instruction-following": 0.08,
        }
    ).to_numpy()

    base_quality = (
        3.65
        + model_effect
        + language_effect
        + category_effect
        + prompt_effect
        + rng.normal(0, 0.50, record_count)
    )

    helpfulness_score = clipped_score(base_quality + rng.normal(0.05, 0.42, record_count))
    accuracy_score = clipped_score(base_quality + rng.normal(-0.03, 0.48, record_count))
    safety_score = clipped_score(
        4.05
        + model_effect * 0.35
        + np.where(category == "Safety", 0.18, 0)
        + np.where(prompt_type == "Adversarial", -0.42, 0)
        + rng.normal(0, 0.48, record_count)
    )

    hallucination_logit = (
        -2.0
        - model_effect
        - language_effect * 0.6
        - category_effect
        - prompt_effect * 0.8
        + (4 - accuracy_score) * 0.65
    )
    hallucination_probability = 1 / (1 + np.exp(-hallucination_logit))
    hallucination_flag = rng.binomial(1, hallucination_probability)

    final_score = (
        helpfulness_score * 0.35
        + accuracy_score * 0.40
        + safety_score * 0.25
        - hallucination_flag * 0.45
    )
    final_score = np.clip(final_score, 1, 5).round(2)

    start_date = pd.Timestamp("2025-01-01")
    dates = start_date + pd.to_timedelta(rng.integers(0, 365, record_count), unit="D")

    dataframe = pd.DataFrame(
        {
            "evaluation_id": [f"EVAL-{number:05d}" for number in range(1, record_count + 1)],
            "date": dates,
            "model": model,
            "language": language,
            "category": category,
            "prompt_type": prompt_type,
            "annotator_id": annotator_id,
            "helpfulness_score": helpfulness_score,
            "accuracy_score": accuracy_score,
            "safety_score": safety_score,
            "hallucination_flag": hallucination_flag,
            "final_score": final_score,
        }
    )

    return dataframe.sort_values(["date", "evaluation_id"]).reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe = generate_dataset()
    dataframe.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Generated {len(dataframe):,} records at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
