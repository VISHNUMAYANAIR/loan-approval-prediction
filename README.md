
# 🚀 Loan Approval Predictor

An interactive machine learning web app to predict whether a loan application will be approved, based on user-provided details like income, loan amount, employment status, and more.

---

## 🔍 Project Overview

This app demonstrates a supervised machine learning pipeline using real-world structured data to make loan approval predictions.

- Cleaned and preprocessed data from [OpenML](https://www.openml.org/d/31)
- Compared **Logistic Regression (L1)** and **Random Forest Classifier**
- Deployed with an interactive **Gradio UI**

---

## 🔍 Encoding Strategy Comparison

To ensure optimal model performance, we compared three different encoding strategies on categorical features:

| Encoding Type              | Logistic Regression AUC | Random Forest AUC |
|---------------------------|--------------------------|--------------------|
| One-Hot Encoding           | **0.7589**               | **0.7820**         |
| Label Encoding (sklearn)   | 0.7120                   | 0.7490             |
| Manual Ordinal Mapping     | 0.7121                   | 0.7679             |

### 🧠 Interpretation & Insights:

- **Logistic Regression** performed best with **One-Hot Encoding**, as expected — it treats encoded columns as independent and avoids assuming ordinal relationships.
- **Random Forest** handled **manual ordinal mappings** quite well — tree-based models can leverage inherent order in features.
- **Label Encoding** underperformed due to its arbitrary, alphabet-based numeric assignment, which introduces false orderings.

### ✅ Final Decision:
We chose **One-Hot Encoding** as the default for deployment due to its superior performance on both models. However, we documented and preserved the manual encoding experiments for transparency and future optimization.

> 📌 _This experiment reflects the real-world ML practice of comparing multiple preprocessing strategies rather than assuming a "one-size-fits-all" approach._


---

## 📊 Models Used

| Model                                    | ROC AUC Score   |
|------------------------------------------|-----------------|
| Logistic Regression (L1 penalty)         | 0.7589          |
| Random Forest (n=140)                    | 0.7820          |

We used **feature importance analysis** and **model interpretability** to compare results visually and statistically.

---

## 📈 Features

- One-hot encoding for categorical variables
- L1 regularization to remove irrelevant features
- Feature importance plot from Random Forest
- Model selection dropdown in the UI
- Clean user interface with input validation

---

## 🧠 How It Works

1. Load trained models (`.pkl`)
2. Take input from user via Gradio
3. Preprocess inputs to match training
4. Predict outcome using selected model

---

## 🛠 How to Run Locally

```bash
pip install -r requirements.txt
python loan_app.py
```

---

## 🌐 Demo App 
**[🔗 Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/your-username/loan-approval-predictor)**

---

## 📁 Project Structure

```
📦 loan-approval-predictor/
├── loan_approval_prediction.py
├── model_lr.pkl
├── model_rf.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md

```
📬 Author

Vishnumaya R. Nair
Connect with me on LinkedIn : https://www.linkedin.com/in/vishnumaya-r-nair-658523202