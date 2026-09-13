# 🛒 E-Commerce Customer Churn Prediction

**Predict which customers are likely to leave — and take action before they do.**

This project builds an end-to-end machine learning system that predicts customer churn for an e-commerce platform.  
It not only predicts churn, but also segments customers into **High / Medium / Low risk** groups so marketing and customer-success teams can act on the results.

---

## 📌 What This Project Does

Given a customer’s demographics, engagement behavior, purchase history, and service interactions, the model answers:

> **“How likely is this customer to churn?”**

Then it groups customers into actionable risk tiers:

| Risk Level     | Churn Probability | Suggested Action                          |
|----------------|-------------------|-------------------------------------------|
| 🔴 High Risk   | ≥ 0.60            | Immediate personal outreach + offers      |
| 🟡 Medium Risk | 0.35 – 0.60       | Automated re-engagement campaigns         |
| 🟢 Low Risk    | < 0.35            | Standard engagement & loyalty programs    |

---

## 📊 Dataset Overview

| Item              | Details                                      |
|-------------------|----------------------------------------------|
| Records           | 50,000 customers                             |
| Features          | 25 columns                                   |
| Target            | `Churned` (0 = Active, 1 = Churned)          |
| Overall Churn Rate| ~28.9%                                       |
| Data Types        | Numerical + Categorical                      |
| Missing Values    | Present in several columns                   |

### Feature Categories

- **Demographics** → Age, Gender, Country, City, Membership Years  
- **Engagement** → Login Frequency, Session Duration, Pages/Session, Cart Abandonment, Wishlist, Email Open Rate, Mobile App Usage, Social Media Score  
- **Purchase Behavior** → Total Purchases, Average Order Value, Days Since Last Purchase, Discount Usage, Return Rate, Payment Diversity  
- **Customer Service** → Service Calls, Product Reviews Written, Lifetime Value  
- **Financial & Status** → Credit Balance, Signup Quarter, Churned  

---

## 🛠️ Project Workflow

1. **Data Loading & Inspection** – shape, dtypes, missing values, duplicates  
2. **Exploratory Data Analysis** – distributions, correlations, churn patterns  
3. **Data Cleaning** – handle missing values and anomalies  
4. **Feature Engineering** – encode categorical variables, scale numerical features  
5. **Model Training** – train 7 different algorithms  
6. **Model Comparison** – select the best performing model  
7. **Hyperparameter Tuning** – optimize XGBoost  
8. **Threshold Optimization** – improve recall for churn detection  
9. **Risk Segmentation** – convert probabilities into business-ready segments  
10. **Model Persistence** – save the full pipeline for future use  

---

## 🏆 Model Performance

| Model                        | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|------------------------------|----------|-----------|---------|----------|---------|
| Logistic Regression          | 78.96%   | 70.06%    | 47.51%  | 56.62%   | 80.68%  |
| K-Nearest Neighbors          | 77.82%   | 68.54%    | 42.95%  | 52.81%   | 76.33%  |
| Naive Bayes                  | 68.47%   | 46.69%    | 64.40%  | 54.14%   | 72.08%  |
| Support Vector Machine       | 85.51%   | 83.99%    | 61.58%  | 71.06%   | 89.33%  |
| Decision Tree                | 88.21%   | 88.36%    | 68.20%  | 76.98%   | 90.39%  |
| Random Forest                | 87.22%   | 90.37%    | 62.42%  | 73.84%   | 91.41%  |
| **XGBoost (Selected)**       | **91.71%** | **90.89%** | **79.24%** | **84.67%** | **92.69%** |

### Final Tuned XGBoost (after threshold optimization)

| Metric     | Score   |
|------------|---------|
| Accuracy   | 91.81%  |
| Precision  | 86.64%  |
| Recall     | 84.71%  |
| F1-Score   | 85.66%  |
| ROC-AUC    | 92.89%  |

**Why XGBoost?**  
It delivered the best balance of accuracy, recall (important for catching churners), and ROC-AUC among all models tested.

---

## 🚀 How to Run This Project

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
```

### Clone Repository
```bash
git clone <repository-url>
cd <Navigate to the project folder>
```

## Usage

### Run Analysis Notebook
```bash
jupyter notebook 1_Customer_churn.ipynb
```

### Launch Interactive Prediction App
```bash
streamlit run 1_streamlit.py
```

## Future Enhancements

- Deploy to cloud (AWS, GCP, Azure)
- Add real-time prediction API
- Implement model monitoring and retraining pipeline
- Expand to multiclass churn scenarios

## License

Open source - Available for educational and commercial use.

---

**Built with ❤️ for customer retention analytics**
