# Customer Churn Prediction System

## Overview
A machine learning solution for predicting customer churn in e-commerce platforms. This project analyzes behavioral, demographic, and transactional data to identify at-risk customers and segment them into actionable risk tiers for targeted retention strategies.

## Dataset
- **50,000 customers** from global e-commerce/subscription platform
- **25 features** including demographics, engagement metrics, purchase behavior, and financial indicators
- **Target Variable**: Binary churn indicator (0 = Retained, 1 = Churned)

## Project Workflow

### 1. Data Cleaning
- Handle missing values (mode for categorical, median for numerical)
- Outlier detection using 1st-99th percentile clipping
- Data validation and duplicate removal

### 2. Exploratory Data Analysis (EDA)
- Feature correlations with churn
- Behavioral pattern visualization
- Geographic and demographic insights

### 3. Feature Engineering
- Behavioral ratios (Recency Ratio, Value Per Purchase)
- Engagement metrics (Session Engagement, Platform Stickiness)
- Loyalty indicators (Loyalty Velocity)
- Log transformations for skewed distributions

### 4. Feature Preprocessing
- One-Hot Encoding for categorical variables
- StandardScaler normalization
- 80-20 train-test split with stratification

### 5. Model Development
Trained and compared 7 algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- **XGBoost** (Best performer)

### 6. Hyperparameter Optimization
- RandomizedSearchCV with 5-fold cross-validation
- Optimized XGBoost parameters for maximum ROC-AUC

### 7. Risk Segmentation
Customers classified into 3 risk tiers:
- **High Risk** (≥0.60): Immediate retention intervention
- **Medium Risk** (0.35-0.60): Automated re-engagement
- **Low Risk** (<0.35): Standard engagement

## Model Performance (XGBoost at 0.35 Threshold)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 94.08% | 91.81% |
| Precision | 91.48% | 86.64% |
| Recall | 87.66% | 84.71% |
| F1-Score | 89.53% | 85.66% |
| ROC-AUC | 97.13% | 92.89% |

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
```

### Clone Repository
```bash
git clone <repository-url>
cd ML_PROJECTS
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

Or use the batch file:
```bash
./run_streamlit.bat
```

## Files Description

| File | Description |
|------|-------------|
| `1_Customer_churn.ipynb` | Complete ML pipeline with code and visualizations |
| `1_streamlit.py` | Interactive web app for real-time churn predictions |
| `ecommerce_customer_churn_dataset.csv` | Raw dataset (50K customer records) |
| `run_streamlit.bat` | Batch script to launch Streamlit app |
| `.gitignore` | Git ignore configuration |

## Key Insights

- **Login Frequency & Session Duration** are strong retention indicators
- **Cart Abandonment Rate** is a critical churn predictor
- **Days Since Last Purchase** (Recency) indicates engagement
- **Customer Service Interactions** correlate with satisfaction
- **Geographic variations** in churn rates suggest regional factors

## Business Applications

✅ Identify high-risk customers for proactive outreach  
✅ Segment customers for targeted retention campaigns  
✅ Optimize resource allocation for support teams  
✅ Early warning system for churn indicators  
✅ Improve customer lifetime value predictions  

## Technologies Used

- **Python 3.x**
- **Pandas & NumPy** - Data manipulation
- **Scikit-Learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting classifier
- **Matplotlib & Seaborn** - Data visualization
- **Streamlit** - Interactive web application
- **Jupyter** - Interactive development

## Model Training Notes

The trained model is saved locally after running the notebook. To regenerate:
1. Open `1_Customer_churn.ipynb`
2. Run all cells
3. Uncomment the model saving code at the end
4. The model will be saved as `1_Customer_churn_pipeline.pkl`

## Future Enhancements

- Deploy to cloud (AWS, GCP, Azure)
- Add real-time prediction API
- Implement model monitoring and retraining pipeline
- Expand to multiclass churn scenarios

## License

Open source - Available for educational and commercial use.

---

**Built with ❤️ for customer retention analytics**
