# 🏦 Bank Customer Churn Prediction

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-bankcapp.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bankcapp.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-IshanGupta09-181717?style=for-the-badge&logo=github)](https://github.com/IshanGupta09/Bank_Customer_Churn_Prediction)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Predict which bank customers are about to leave — before they do.**  
End-to-end ML project with EDA, feature engineering, 3 models, and a live interactive app.

[**🔴 Try the Live App →**](https://bankcapp.streamlit.app/)

</div>

---

## 📌 Problem Statement

A European bank is experiencing customer churn. Every lost customer means lost revenue — the goal is to **proactively identify at-risk customers** so the retention team can intervene in time.

> **Input:** Customer profile (demographics, account info, activity)  
> **Output:** Churn probability + risk tier + personalised retention recommendations

---

## 🎬 App Preview

| Prediction Tab | Model Performance | Data Insights |
|:-:|:-:|:-:|
| Churn risk gauge + key factors | ROC curves + confusion matrices | Churn by geography, age, products |

> 👆 **[Open the live app](https://bankcapp.streamlit.app/)** — no setup needed

---

## 📊 Dataset

- **Source:** [Churn Modelling — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size:** 10,000 customers from France, Germany, Spain
- **Target:** `Exited` (1 = churned, 0 = retained)
- **Class split:** ~80% retained / ~20% churned

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France / Germany / Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Has a credit card (0/1) |
| `IsActiveMember` | Active in last 6 months (0/1) |
| `EstimatedSalary` | Estimated annual salary |

---

## 🔬 Project Workflow

```
Data Loading → EDA → Feature Engineering → Preprocessing → Model Training → Evaluation → Deployment
```

### 1️⃣ Exploratory Data Analysis
- Class imbalance analysis (80/20 split)
- Distribution plots for all numeric features, split by churn status
- Churn rate by geography, gender, products, and activity
- Correlation heatmap
- Age × Geography churn heatmap

### 2️⃣ Feature Engineering
Four new features engineered on top of the base dataset:

| Feature | Logic | Intuition |
|---|---|---|
| `BalancePerProduct` | `Balance / (NumOfProducts + 1)` | Wealth concentration per product |
| `AgeGroup` | Binned age (18–30, 31–40, ...) | Non-linear age effect |
| `IsHighBalance` | Balance > median | Binary wealth flag |
| `TenurePerAge` | `Tenure / Age` | Loyalty relative to lifetime |

### 3️⃣ Models Trained

| Model | Accuracy | ROC-AUC | F1 Score | CV AUC |
|---|---|---|---|---|
| 🌲 **Random Forest** | **~86%** | **~88%** | **~74%** | **~87%** |
| 📈 Gradient Boosting | ~86% | ~88% | ~73% | ~87% |
| 📉 Logistic Regression | ~81% | ~83% | ~60% | ~83% |

> ✅ **Best model:** Random Forest — deployed in the Streamlit app

### 4️⃣ Key Findings

- 🇩🇪 **Germany** has the highest churn rate (~32%) vs France (~16%)
- 👴 Customers aged **40–60** churn significantly more than younger ones
- 😴 **Inactive members** are ~2× more likely to churn
- 📦 Customers with **3–4 products** show >80% churn rate (likely over-sold)
- 💳 Having a credit card alone shows no significant churn difference

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/IshanGupta09/Bank_Customer_Churn_Prediction.git
cd Bank_Customer_Churn_Prediction

# Install
pip install -r requirements.txt

# Launch app
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📁 Repository Structure

```
Bank_Customer_Churn_Prediction/
│
├── app.py                              # Streamlit application (live demo)
├── Bank_Customer_Churn_Prediction.ipynb  # Full analysis notebook
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Python + Streamlit gitignore
├── LICENSE                             # MIT
└── README.md                           # You are here
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression) |
| Data | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 💡 Business Impact

Assuming 10,000 customers and $1,200 average annual revenue per customer:

- Current churn (~20%) = **$2.4M revenue at risk / year**
- Model catches ~60% of churners; retention rate of 40%
- **Estimated $576K annual revenue saved** through model-driven interventions

---

## 👨‍💻 Author

**Ishan Gupta** — CS Engineer specialising in Big Data Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ishan--gupta091-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/ishan-gupta091/)
[![GitHub](https://img.shields.io/badge/GitHub-IshanGupta09-181717?style=flat&logo=github)](https://github.com/IshanGupta09)
[![Portfolio](https://img.shields.io/badge/DS%20Portfolio-View%20Projects-orange?style=flat)](https://github.com/IshanGupta09/Data_Science_Portfolio)

---

<div align="center">
⭐ If you found this useful, consider starring the repo!
</div>
