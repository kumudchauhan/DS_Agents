# Autonomous Data Science Agent (LangGraph + LLM)

**Live Demo:** [kc-agents.streamlit.app](https://kc-agents.streamlit.app/)

## Overview

An **autonomous data science agent** that iteratively builds, evaluates, and improves ML pipelines — driven by LLM feedback, not hardcoded rules.

Upload a CSV, pick a target column, and the agent will:

- Perform exploratory data analysis (EDA) with visualizations
- Generate a data quality report with ranked issues
- Clean and preprocess the dataset
- Engineer features from a registry of 14 safe transforms
- Train and evaluate models from a pool of 4 classifiers
- Use an **LLM critic** to analyze results and recommend improvements
- **Loop** — applying the critic's recommendations to the next iteration
- Surface diagnostic **Key Takeaways** (zero-recall, accuracy paradox, F1 regression, etc.)

Built with **LangGraph** for stateful graph execution, **OpenRouter** for LLM access, and **Streamlit** for the interactive UI.

---

## Architecture — The REACT Loop

The agent runs a closed feedback loop where the LLM critic's recommendations actually drive the next iteration:

```
EDA → Cleaning → Feature Engineering → Modeling → Evaluation → Critic → Decision
                        ↑                                         |
                        └─── recommendations (features, model) ───┘
```

| Component | What it does |
|---|---|
| **Critic Node** | Two LLM calls: (1) text feedback for the UI, (2) structured JSON recommendations for the pipeline |
| **Feature Engineering** | Reads `recommendations["features_to_add"]` — picks from a registry of 14 transforms |
| **Modeling** | Reads `recommendations["model_config"]` — instantiates from 4 models with clamped hyperparameters |
| **Decision** | Stops on F1 >= 0.85, max iterations, or LLM `should_stop` signal |

Iteration 0 uses sensible defaults (5 base features + LogisticRegression). Every subsequent iteration is shaped by the LLM's analysis of the previous results.

---

## Key Features

### Adaptive Pipeline (Not Hardcoded)

The LLM critic produces **structured JSON recommendations** that control:
- Which features to create (from 14 available transforms)
- Which model to train (LogisticRegression, RandomForest, GradientBoosting, SVC)
- Hyperparameters (validated and clamped to safe ranges)
- Whether to stop iterating

### Feature Registry

14 safe, self-contained transforms — each handles its own column dependencies:

| Feature | Description |
|---|---|
| `account_age_days` | Days between signup and transaction |
| `hour_of_day` | Transaction hour |
| `day_of_week` | Day of week (0=Mon) |
| `amount_to_avg_ratio` | Amount / 7-day average |
| `is_high_amount` | Above 95th percentile |
| `log_amount` | Log-transform of amount |
| `is_new_account` | Account < 30 days old |
| `is_night_txn` | Hour 23–5 |
| `amount_deviation` | |amount - avg_amount_7d| |
| `high_velocity` | 4+ transactions in 24h |
| `amount_squared` | Non-linear amount effect |
| `amount_x_velocity` | Amount x transaction velocity |
| `is_weekend` | Saturday or Sunday |
| `txn_per_avg_ratio` | Transaction count / avg spending |

### Model Factory

| Model | Hyperparameter Ranges |
|---|---|
| LogisticRegression | C: 0.01–10 |
| RandomForestClassifier | n_estimators: 50–300, max_depth: 5–30 |
| GradientBoostingClassifier | n_estimators: 50–300, learning_rate: 0.01–0.3, max_depth: 3–10 |
| SVC | C: 0.1–10, kernel: rbf/linear |

All models use `class_weight="balanced"`. Scaling is auto-applied for models that need it.

### Key Takeaways (Diagnostics)

The UI automatically surfaces actionable issues:
- **Zero recall** — model classifies everything as negative (class-imbalance problem)
- **Accuracy paradox** — high accuracy but near-zero F1 on imbalanced data
- **F1 regression** — performance dropped between iterations
- **Feature set changes** — what was added/removed across iterations
- **LLM engagement** — whether recommendations were applied or fell back to defaults

### Streamlit UI

Six sections rendered during and after the pipeline run:

1. **Upload Dataset** — CSV upload with target column selection
2. **Run Pipeline** — step-by-step status tracking with live progress
3. **Data Exploration** — data quality report (ranked issues, missing values, target distribution) + EDA visualizations
4. **Pipeline Details** — cleaning summary, feature engineering per iteration, feature importance charts
5. **Model Results** — grouped metrics chart, comparison table, Key Takeaways diagnostics, LLM critic feedback
6. **Q&A** — ask natural language questions about the dataset

---

## Project Structure

```
ds_agent/
├── streamlit_app.py              # Streamlit UI entry point
├── app/
│   ├── main.py                   # CLI entry point — run_agent()
│   ├── graph/
│   │   ├── builder.py            # LangGraph workflow with conditional loop
│   │   ├── state.py              # AgentState TypedDict (shared state)
│   │   └── nodes/
│   │       ├── eda.py            # Load CSV, profile data, generate visualizations
│   │       ├── cleaning.py       # Normalise strings, fill NaNs, parse dates
│   │       ├── feature_eng.py    # Feature registry — 14 transforms, LLM-driven selection
│   │       ├── modeling.py       # Model factory — 4 classifiers, LLM-driven config
│   │       ├── evaluation.py     # F1, precision, recall, confusion matrix
│   │       ├── critic.py         # LLM critique + structured JSON recommendations
│   │       └── decision.py       # Stop on F1 threshold, max iter, or LLM signal
│   ├── llm/
│   │   └── llm_provider.py      # OpenRouter via ChatOpenAI
│   └── ui/
│       ├── components.py         # Render helpers (data quality, features, metrics, takeaways)
│       ├── pipeline_runner.py    # Pipeline execution generator for Streamlit
│       └── qa.py                 # Natural language Q&A over the dataset
├── data/
│   └── transactions.csv          # 1005-row fraud detection dataset (demo)
├── outputs/                      # Generated visualizations
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/kumudchauhan/DS_Agents.git
cd DS_Agents
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys).

```bash
export OPENROUTER_API_KEY='your-key-here'
```

### 3. Run

**Streamlit UI (recommended):**
```bash
streamlit run streamlit_app.py
```

**CLI:**
```bash
python -m app.main
```

---

## Using Your Own Dataset

Upload any CSV through the Streamlit UI — no code changes required. Select your target column from the dropdown.

For CLI usage, update `main.py`:
```python
run_agent("data/your_dataset.csv", target_column="your_target")
```

The agent handles data profiling, cleaning, and feature engineering automatically. The feature registry is general-purpose and will skip transforms whose required source columns don't exist.

---

## Deployment

The app is deployed on **Streamlit Community Cloud**:
- Auto-deploys from `main` branch on push
- API key configured via Streamlit Secrets (`OPENROUTER_API_KEY`)
- Live at [kc-agents.streamlit.app](https://kc-agents.streamlit.app/)

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (stateful graph execution) |
| LLM | OpenRouter (Mistral Small 3.1 24B) |
| ML | scikit-learn |
| UI | Streamlit |
| Data | pandas, numpy, matplotlib |

---

## Contributing

Fork, extend, and make it your own. PRs welcome.
