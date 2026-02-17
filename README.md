# SaaS Subscription Churn Prediction  
Machine Learning for Social Data Science (SSIM916)

## Project Overview

This project develops and evaluates machine learning models to predict customer churn in a SaaS (Software-as-a-Service) subscription business.

Using the RavenStack Synthetic SaaS Subscription and Churn Analytics Dataset, the analysis compares Logistic Regression and Random Forest models to see which method better identifies accounts that are at risk of cancelling.

The final model (Random Forest) achieved a ROC-AUC score of **0.726** on the held-out test set.

---

## Research Question

What factors have the strongest impact on customer churn in SaaS subscription services? To what extent can machine learning (ML) models accurately predict customers before cancellation?

---

## Dataset

**Source:** RavenStack Synthetic SaaS Subscription and Churn Analytics Dataset  
**Author:** Rivalytics (Kaggle)

The dataset contains five related CSV tables:

- `ravenstack_accounts.csv`
- `ravenstack_subscriptions.csv`
- `ravenstack_feature_usage.csv`
- `ravenstack_support_tickets.csv`
- `ravenstack_churn_events.csv`

The data are fully synthetic and contain no personally identifiable information.

All tables were aggregated to the **account level (n = 500)** to create the modelling dataset.

---

## Project Structure

```
ML_PROJECT/
│  
├── data/                    # Raw dataset files  
├── models/                  # Saved trained models  
├── outputs/                 # Model outputs, plots, metrics  
├── src/  
│   ├── 1_data_preparation.py  
│   ├── 2_feature_engineering.py  
│   ├── 3_model_training.py  
│   └── 4_model_evaluation.py  
└── README.md  
```

---

## Reproducibility Instructions

### 1. Install Dependencies

```
pip install pandas numpy scikit-learn matplotlib joblib
```

Python version used: **3.9+**

---

### 2. Run the Full Pipeline (in order)

From the project root:

```
python src/1_data_preparation.py
python src/2_feature_engineering.py
python src/3_model_training.py
python src/4_model_evaluation.py
```

---

## Outputs Generated

After running all scripts, the following files will be created:

### Feature Engineering
- `outputs/feature_table.csv`  
- `outputs/feature_table_preview.csv`  

### Model Training
- `outputs/model_metrics.json`  
- `outputs/test_predictions.csv`  
- `outputs/feature_importance_rf.csv`  
- `models/logreg.joblib`  
- `models/rf.joblib`  

### Model Evaluation
- `outputs/roc_curve_test.png`  
- `outputs/confusion_matrix_rf_0_5.png`  
- `outputs/confusion_matrix_rf_best.png`  
- `outputs/threshold_tuning_rf.csv`  
- `outputs/evaluation_summary.txt`  

---

## Model Summary (Test Set)

**Logistic Regression**
- ROC-AUC: 0.654  
- Accuracy: 0.656  

**Random Forest**
- ROC-AUC: 0.726  
- Accuracy: 0.776  

Random Forest performed better and was selected as the final model.

The dataset shows a **high churn rate (77.4%)**, which creates class imbalance and makes the results sensitive to the choice of decision threshold.

---

## Key Predictive Drivers

The most important predictors identified by the Random Forest model include:

- Number of subscriptions  
- Customer tenure (days since signup)  
- Usage duration over the past 60–90 days  
- Support resolution and response times  
- Seats per subscription  
- Urgent support ticket rate  
- Maximum monthly recurring revenue (MRR)  

Overall, behavioural engagement and service quality indicators were more informative than demographic attributes.

---

## Notes

- All modelling was performed at the **account level**.  
- A reference date approach was used during feature engineering to reduce data leakage.  
- Because of the high churn rate (77.4%), threshold selection strongly affects classification results.  
- Probability scores may be more useful for ranking risk than using a single cutoff value.  
- This project was completed for academic purposes as part of SSIM916.

---