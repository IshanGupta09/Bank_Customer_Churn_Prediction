# 🏦 ChurnGuard — Bank Customer Churn Predictor

A production-ready Streamlit application that predicts whether a bank customer is likely to churn, built with Random Forest, Gradient Boosting, and Logistic Regression.

## 🚀 Live Features

- **Real-time Prediction** — Fill in a customer profile and instantly get a churn risk score
- **Gauge Chart** — Visual risk meter with colour-coded result
- **Risk Factor Breakdown** — Key drivers contributing to churn risk
- **Retention Recommendations** — Actionable steps based on the customer's profile
- **Model Comparison** — Side-by-side Accuracy, ROC-AUC, and F1 across 3 models
- **Feature Importances** — Random Forest feature importance chart
- **Data Insights** — Churn patterns by Geography, Products, and Age

## 📦 Setup & Run

```bash
# 1. Clone the repo (or copy this folder)
cd churn_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit, Plotly |
| ML Models | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression) |
| Data Processing | Pandas, NumPy |
| Styling | Custom CSS (dark theme) |

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | F1 |
|-------|----------|---------|-----|
| Random Forest | ~86% | ~88% | ~74% |
| Gradient Boosting | ~86% | ~88% | ~73% |
| Logistic Regression | ~81% | ~83% | ~60% |

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select `app.py`
4. Click **Deploy** — live URL in ~2 minutes!

## 📁 Project Structure

```
churn_app/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 👨‍💻 Author

**Ishan Gupta** — [LinkedIn](https://www.linkedin.com/in/ishan-gupta091/) | [GitHub](https://github.com/IshanGupta09)
