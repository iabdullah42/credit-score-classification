# Credit Score Classification

**Predicting a customer's credit score band (Poor / Standard / Good) from banking behavior, and comparing nine machine-learning classifiers.**

## Problem

Lenders sort customers into credit tiers to price risk. Given 100,000 monthly customer records with income, debt, payment and credit-inquiry history, can a model predict which tier a customer belongs in?

## Approach

| Step | Notebook | What happens |
|---|---|---|
| 1. Cleaning | `Step_1 Cleaning.ipynb` | Fixes corrupted numeric fields (stray `_` characters in age, income, loans, debt and balances), converts types, and flags blanks as missing |
| 2. EDA + extra cleaning | `Step_2_EDA_and_Extra_Cleaning_Modified_Dataset.ipynb` | Drops identifiers and unreliable columns (ID, SSN, Type_of_Loan and others), removes outliers above thresholds, and treats negative counts as missing. 100,000 → 76,324 rows |
| 3. Modeling | `ML.ipynb` | One-hot encodes categoricals, scales features, 80/20 train/test split, then trains and compares nine classifiers |

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Gradient Boosting** | **77.5%** | 73.5% | 75.8% | 74.1% |
| Random Forest | 77.1% | 72.9% | 76.3% | 73.8% |
| CatBoost | 76.4% | 72.0% | 73.8% | 72.5% |
| SVM | 74.8% | 71.2% | 76.1% | 71.7% |
| Logistic Regression | 72.6% | 68.6% | 70.4% | 69.0% |
| Decision Tree | 70.8% | 66.5% | 66.5% | 66.4% |
| Naive Bayes | 69.8% | 68.9% | 78.2% | 68.6% |
| KNN | 68.1% | 63.4% | 66.6% | 64.2% |
| AdaBoost | 62.6% | 59.0% | 55.9% | 50.3% |

*Precision, recall and F1 are macro-averaged across the three classes. Full results are in `model_comparison_results.csv`.*

**Takeaways**
- Tree ensembles (Gradient Boosting, Random Forest, CatBoost) clearly beat linear and distance-based models, which points to non-linear interactions between debt, delays and credit inquiries.
- The best model's F1 by class is Good 82.4%, Standard 75.7% and **Poor 64.1%**. The "Poor" tier, the one a lender most needs to catch, is the hardest to predict. A next step would be class weighting or threshold tuning to raise recall on that tier.

## Run it

1. Download the Kaggle **"Credit Score Classification"** dataset and place `train.csv` in this folder.
2. `pip install pandas numpy scikit-learn catboost matplotlib seaborn`
3. Run the notebooks in order: Step 1 → Step 2 → ML.

**Tools:** Python, pandas, NumPy, scikit-learn, CatBoost, matplotlib, seaborn

*Built as a project for the OSU AI Bootcamp.*
