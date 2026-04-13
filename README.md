# Credit Card Customer Churn Analysis 💳

An end-to-end exploratory data analysis project investigating why credit card customers churn, built with Python, Plotly Express, and Streamlit.

---

## Problem Statement

A bank manager is facing an increasing number of customers cancelling their credit card services. This project explores the behavioral, demographic, and financial patterns that distinguish churned customers from existing ones — with the goal of identifying the key drivers of churn and building a foundation for a predictive model.

---

## Dataset

- **Source:** [Kaggle — Credit Card Customers by Sakshi Goyal](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **File:** `BankChurners.csv`
- **Size:** 10,127 customers × 21 features (after cleaning)
- **Target Variable:** `Attrition_Flag` — Existing Customer / Attrited Customer
- **Churn Rate:** ~16.1%

---

## Project Structure

```
├── BankChurners.csv          # Dataset (download from Kaggle)
├── FinalProject.ipynb        # Full EDA notebook
├── app.py                    # Streamlit dashboard app
└── README.md
```

---

## Business Questions Explored

- Does gender, age, or marital status affect churn rate?
- Which income bracket and card category loses the most customers?
- Do customers with more bank products churn less?
- Is there a transaction spending threshold below which customers are likely to churn?
- Do inactive customers churn more?
- Do churned customers show a declining transaction trend over time?
- Which features are most correlated with churn?

---

## Key Findings

| Finding | Insight |
|---|---|
| Transaction count & amount | Strongest churn predictor — customers with <40 transactions/year churn at much higher rates |
| Credit utilization | Near-zero utilization is a strong churn signal, especially with low credit limits |
| Bank relationships | Fewer products = higher churn risk; cross-selling is a key retention lever |
| Transaction trend (Q4 vs Q1) | Declining transaction frequency (ratio < 1.0) signals upcoming churn |
| Demographics | Gender, income, education, and marital status show no meaningful churn difference |
| Credit limit | High-limit customers (>$15k) almost never churn |

---

## Streamlit App

The interactive dashboard includes 5 pages:

1. **Introduction** — Problem statement, data source, column definitions, and dataset metrics
2. **Business Questions** — Organized list of analytical questions guiding the project
3. **Exploratory Data Analysis** — All visualizations organized by category (demographics, financial behavior, transaction behavior)
4. **Dashboard & Findings** — Key charts paired with business insights and a KPI summary row
5. **Next Steps & Recommendations** — Business recommendations and modeling roadmap

### Running the App

```bash
# Install dependencies
pip install streamlit pandas numpy plotly

# Place BankChurners.csv in the same directory as app.py, then run:
streamlit run app.py
```

---

## EDA Notebook

The notebook (`FinalProject.ipynb`) covers:

- Data loading and cleaning
- Null and duplicate checks
- Target variable analysis
- Univariate & bivariate analysis across all features
- Correlation heatmap
- Multivariate scatter plots
- Observations and conclusions after each chart

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Plotly Express | Interactive visualizations |
| Streamlit | Dashboard / web app |
| Jupyter Notebook | EDA environment |

---

## Next Steps

- Drop redundant features (`Avg_Open_To_Buy`, `CLIENTNUM`, `Months_on_book`)
- Encode categorical variables and handle `Unknown` values
- Address class imbalance (83.9% vs 16.1%) using SMOTE or class weights
- Train classification models: Random Forest, XGBoost, Logistic Regression
- Evaluate using AUC-ROC, Precision-Recall, and F1-Score

---

## Author

Abdu — Data Analysis Project
