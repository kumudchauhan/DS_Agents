# Autonomous Data Science Agent (LangGraph + LLM)

## Overview

This project implements an **autonomous data science agent** that can:

* Perform Exploratory Data Analysis **(EDA)**
* Create Data Visualizations
* Provide Data Quality report
* Clean and preprocess datasets
* Engineer meaningful features
* Train and evaluate machine learning models
* Iteratively improve itself using an **LLM-based feedback loop**

The system is built using a **stateful graph-based architecture**, enabling structured workflows, conditional execution, and iterative refinement.

> ⚠️ This is a **starter framework / demo system** — designed to showcase agentic AI design patterns. It is intentionally flexible and can be adapted to **any dataset or ML problem**, not just fraud detection.

---

## Key Features

### Agentic Workflow - Using LangGraph

* Stateful execution across multiple steps
* Modular nodes for each stage of the DS pipeline
* Conditional looping for iterative improvement

---

### 🔁 Self-Improvement Loop 

The agent doesn’t just run once — it **observe, react, learns and improves**:

```
EDA → Cleaning → Feature Engineering → Modeling → Evaluation → Critic → (loop)
```

Use LLM as a judge to
* Evaluates model performance
* critique results
* Refines pipeline automatically

---

### 🤖 LLM-as-Judge

* Provides feedback on:

  * Feature engineering quality
  * Model choice
  * Data issues (imbalance, leakage)
* Drives iterative improvements

---

### 💰 Fully Local & Free

* Runs entirely on your machine
* Uses local LLMs (via Ollama)
* No API keys required

---

## 📁 Project Structure

```
ds_agent/
│
├── app/
│   ├── main.py                  # Entry point — run_agent()
│   ├── graph/
│   │   ├── builder.py           # LangGraph workflow with conditional loop
│   │   ├── state.py             # AgentState TypedDict (shared state)
│   │   └── nodes/
│   │       ├── eda.py           # Load CSV, profile data, report distributions
│   │       ├── cleaning.py      # Normalise strings, fill NaNs, parse dates
│   │       ├── feature_eng.py   # Progressive features (more each iteration)
│   │       ├── modeling.py      # Rotates models across iterations
│   │       ├── evaluation.py    # F1, precision, recall, confusion matrix
│   │       ├── critic.py        # LLM critique via Ollama (with fallback)
│   │       └── decision.py      # Stop on F1 threshold or max iterations
│   │
│   └── llm/
│       └── llm_provider.py      # ChatOllama wrapper
│
├── data/
│   └── transactions.csv         # 1005-row fraud detection dataset
│
├── outputs/
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd autonomous-ds-agent
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install and run local LLM (Ollama)

```bash
# Install Ollama (if not installed)
# https://ollama.com

# Pull a model
ollama pull mistral
```

---

### 4. Run the agent

```bash
python -m app.main
```

---

## 📊 Using Your Own Dataset (IMPORTANT)

This project is **NOT limited to fraud detection**.

You can plug in **any CSV dataset**.

---

### 🧾 Requirements for your dataset

* Must be a `.csv` file (for now using CSV, code can be tweaked for reading JSON or any other datatype) 
* Should contain:

  * Features (X)
  * Target variable (y)

---

### 🔧 How to Customize

#### Step 1: Replace dataset

Put your file in:

```
/data/your_dataset.csv
```

---

#### Step 2: Update dataset path

In `main.py`:

```python
run_agent("data/your_dataset.csv")
```

---

#### Step 3: Define target column

In `main.py`, update the `run_agent` call:

```python
run_agent("data/your_dataset.csv", target_column="your_target")
```

---

#### Step 4: Adjust task type (if needed)

| Task Type      | What to change                                     |
| -------------- | -------------------------------------------------- |
| Classification | Use classifiers (RandomForest, LogisticRegression) |
| Regression     | Use regressors (LinearRegression, XGBoost)         |

---

## 🧠 Example Use Cases

You can use this framework for:

* Fraud detection (default demo)
* Churn prediction
* Credit risk modeling
* Sales forecasting (regression)
* Customer segmentation (with modifications)

---

## 🔄 How the Agent Works

### 1. EDA Node

* Understands data distributions
* Identifies missing values

---

### 2. Cleaning Node

* Handles missing values
* Fixes data types

---

### 3. Feature Engineering Node

* Creates derived features
* Improves signal quality

---

### 4. Modeling Node

* Trains ML model
* Splits train/test

---

### 5. Evaluation Node

* Computes metrics (F1, accuracy, etc.)

---

### 6. Critic Node (LLM)

* Analyzes performance
* Suggests improvements

---

### 7. Decision Node

* Decides:

  * Continue improving
  * OR stop

---

## Example Output

```
Iteration 1:
F1 Score: 0.58
Feedback: Add ratio-based features, handle class imbalance

Iteration 2:
F1 Score: 0.64
Feedback: Try different model, normalize features
```

---

## Extending the System

### Easy Extensions

* Add new models (XGBoost, LightGBM)
* Add new feature engineering steps
* Improve prompts for better LLM reasoning

---

### Advanced Extensions

* Multi-model comparison
* Hyperparameter tuning loop
* Dataset-agnostic auto-detection
* Experiment tracking
* UI dashboard (Streamlit)

---

## Switching to API-Based LLM (Optional)

If you want stronger reasoning:

### Replace:

* Local LLM (Ollama)

### With:

* OpenAI / GPT API

Only update:

```
app/llm/llm_provider.py
```

---

## 🎯 Why This Project Matters

This project demonstrates:

* ✅ Agentic AI system design
* ✅ LangGraph-based orchestration
* ✅ Iterative self-improving pipelines
* ✅ Real-world ML workflow automation
* ✅ Cost-efficient AI engineering

---

## Note:

This is a **starter framework**, can be customized to any Data Science ML usecase.

> The goal is to apply my learning of Agentic systems to showcase **how to design autonomous systems**, not just train models.

You are encouraged to:

* Modify it
* Break it
* Extend it
* Make it your own

---

## Contributing

Feel free to fork and extend.

---

## ⭐ If you found this useful

Give it a star and share your improvements!
