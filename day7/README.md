# Day 7 · Sentiment Classification — From Text to Decisions

## Objective
Train a text-classification pipeline on movie reviews using scikit-learn, evaluate with accuracy + classification report, test on 5 hand-written reviews, and analyse where the model fails and why.

## What's Inside

| File | Description |
|------|-------------|
| `day7_sentiment_classification.ipynb` | Complete notebook: dataset, two pipelines (Naive Bayes & Logistic Regression), evaluation, 5 manual test cases, error analysis, feature visualisation |
| `sentiment_model.joblib` | Saved trained model (generated after running the notebook) |

## Key Concepts
- **TF-IDF vectorisation** — convert raw text into numerical features weighted by term frequency and inverse document frequency
- **Naive Bayes vs Logistic Regression** — two classic text classification algorithms compared head-to-head
- **Confusion matrix & classification report** — precision, recall, F1 beyond raw accuracy
- **Error analysis** — understanding failure modes: sarcasm, mixed sentiment, subtle negativity

## How to Run
```bash
pip install scikit-learn joblib matplotlib
jupyter notebook day7_sentiment_classification.ipynb
```

## Connection to Day 6
Day 6 used a pre-built visual classifier (Teachable Machine) with zero code. Day 7 goes under the hood — we build the entire ML pipeline from scratch: tokenisation → TF-IDF → classifier → evaluation. Same pattern, full control.
