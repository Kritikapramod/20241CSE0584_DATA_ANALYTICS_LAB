# Data Analytics Lab - CA3 (20241CSE0584)

**Student Roll No:** 20241CSE0584  
**Course:** Data Analytics Laboratory  
**Assessment:** Continuous Assessment 3 (CA3)  
**File:** `20241CSE0584_CA3`

---

## Overview
This repository contains the complete implementation and analysis for **Anime Rating Classification using Support Vector Machines (SVM)** on the Anime dataset (`Anime.csv`).

The task is to classify whether an anime is **Highly Rated** (Rating >= 4.0) versus **Below 4.0** based on key metadata features including episode count, release year, anime type, release season, and content tags.

## Repository Contents
- **`20241CSE0584_CA3.ipynb`**: Complete Jupyter / Google Colab notebook with executed code cells, performance metrics tables, and visualization charts.
- **`20241CSE0584_CA3.py`**: Clean, standalone Python script version for direct execution.
- **`.gitignore`**: Ignores temporary Python cache and notebook checkpoint files.

## Methodology & Machine Learning Pipeline
1. **Data Cleaning & Target Definition**:
   - Filter records with valid, non-null ratings.
   - Clean and strip categorical columns (`Type`, `Release_season`).
   - Define binary target variable: `Highly_Rated = 1` for Rating >= 4.0, and `0` otherwise.

2. **Exploratory Data Analysis**:
   - Histogram distribution of anime ratings with a vertical reference line at the 4.0 cutoff threshold.

3. **Feature Preprocessing & Engineering**:
   - Stratified train-test split (80% / 20%) to maintain target class ratio.
   - **Numerical Pipeline**: Median Imputation (`SimpleImputer`) followed by Feature Standardization (`StandardScaler`) on `Episodes` and `Release_year`.
   - **Categorical Pipeline**: One-Hot Encoding (`OneHotEncoder`) on `Type` and `Release_season`.
   - **Tag Text Processing**: Binary `CountVectorizer` with custom tag tokenization across top 40 frequent tags.
   - Combined via `ColumnTransformer`.

4. **Model Architecture**:
   - **Support Vector Classifier (SVC)** with Radial Basis Function (RBF) kernel.
   - Hyperparameters: C = 1.0, gamma = 'scale', class_weight = 'balanced'.

5. **Evaluation & Visualization**:
   - Confusion Matrix (True Positives, True Negatives, False Positives, False Negatives).
   - Performance Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, Classification Report.
   - Receiver Operating Characteristic (ROC) curve analysis.
   - SVM Decision Score distribution histogram showing separation between classes against the decision threshold.

## Requirements
```bash
pip install numpy pandas matplotlib scikit-learn
```
