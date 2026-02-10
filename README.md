# Heart Disease Classification - ML Model Comparison

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and accurate prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. This project aims to predict the presence of heart disease in patients based on various medical attributes using multiple machine learning classification algorithms. The goal is to compare the performance of six different ML models and identify the most effective approach for heart disease classification.

## Dataset Description

**Dataset Name:** Heart Disease UCI Dataset

**Source:** UCI Machine Learning Repository / Kaggle

**Dataset Size:** 1,025 samples

**Number of Features:** 13 independent features + 1 target variable

### Features Description:

1. **age**: Age of the patient (in years)
2. **sex**: Sex of the patient (1 = male, 0 = female)
3. **cp**: Chest pain type (0-3)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol level (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment (0-2)
12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)

**Target Variable:**
- **target**: Presence of heart disease (0 = no disease, 1 = disease present)

**Class Distribution:**
- Class 0 (No Heart Disease): 499 samples (48.7%)
- Class 1 (Heart Disease Present): 526 samples (51.3%)

## Models Used

This project implements and compares six different machine learning classification algorithms:

1. **Logistic Regression** - A linear model for binary classification
2. **Decision Tree Classifier** - A tree-based model that makes decisions based on feature values
3. **K-Nearest Neighbors (kNN)** - A distance-based classifier
4. **Naive Bayes (Gaussian)** - A probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** - An ensemble of decision trees
6. **XGBoost (Ensemble)** - An optimized gradient boosting algorithm

### Comparison Table: Model Performance Metrics

<table style="width:100%; border-collapse:collapse; text-align:center;">
   <thead>
      <tr>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">ML Model Name</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">Accuracy</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">AUC</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">Precision</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">Recall</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">F1 Score</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">MCC</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; text-align:left;">Logistic Regression</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8098</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9298</td>
         <td style="border:1px solid #ddd; padding:8px;">0.7619</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9143</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8312</td>
         <td style="border:1px solid #ddd; padding:8px;">0.6309</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; text-align:left;">Decision Tree</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9854</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9857</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9714</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9855</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9712</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; text-align:left;">k-Nearest Neighbors</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8634</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9629</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8738</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8571</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8654</td>
         <td style="border:1px solid #ddd; padding:8px;">0.7269</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; text-align:left;">Gaussian Naive Bayes</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8293</td>
         <td style="border:1px solid #ddd; padding:8px;">0.9043</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8070</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8762</td>
         <td style="border:1px solid #ddd; padding:8px;">0.8402</td>
         <td style="border:1px solid #ddd; padding:8px;">0.6602</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; text-align:left;">Random Forest (Ensemble)</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; text-align:left;">XGBoost (Ensemble)</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
         <td style="border:1px solid #ddd; padding:8px;">1.0000</td>
      </tr>
   </tbody>
</table>

### Observations on Model Performance

<table style="width:100%; border-collapse:collapse;">
   <thead>
      <tr>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">ML Model Name</th>
         <th style="border:1px solid #bbb; padding:8px; background:#f4f4f4;">Observation about model performance</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; font-weight:600;">Logistic Regression</td>
         <td style="border:1px solid #ddd; padding:8px;">Shows decent performance with 80.98% accuracy. High recall (91.43%) indicates strong positive-case detection, while lower precision (76.19%) suggests more false positives. Serves as a good baseline model with a solid AUC of 0.93.</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; font-weight:600;">Decision Tree</td>
         <td style="border:1px solid #ddd; padding:8px;">Delivers 98.54% accuracy with perfect precision (100%) and strong MCC (0.97). Slight overfitting may exist because recall dips to 97.14%, but performance remains outstanding overall.</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; font-weight:600;">k-Nearest Neighbors</td>
         <td style="border:1px solid #ddd; padding:8px;">Achieves balanced precision (87.38%) and recall (85.71%) with 86.34% accuracy. Performance depends on proper feature scaling and k selection, but it remains a reliable non-parametric baseline.</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; font-weight:600;">Gaussian Naive Bayes</td>
         <td style="border:1px solid #ddd; padding:8px;">Posts 82.93% accuracy with solid recall (87.62%). The independence assumption limits precision (80.70%), yet the classifier is lightweight and offers quick probabilities.</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; font-weight:600;">Random Forest (Ensemble)</td>
         <td style="border:1px solid #ddd; padding:8px;">Exhibits perfect metrics across the board, highlighting the strength of ensemble averaging and feature bagging. No signs of overfitting on the test split, making it the top classical choice.</td>
      </tr>
      <tr>
         <td style="border:1px solid #ddd; padding:8px; font-weight:600;">XGBoost (Ensemble)</td>
         <td style="border:1px solid #ddd; padding:8px;">Matches Random Forest with flawless results due to gradient-boosted trees with built-in regularization. Handles complex interactions elegantly and is equally suitable for deployment.</td>
      </tr>
   </tbody>
</table>

## Key Findings

1. **Best Performing Models:** Random Forest and XGBoost both achieved perfect scores across all evaluation metrics (100% accuracy, precision, recall, F1, MCC, and AUC).

2. **Ensemble Superiority:** Ensemble methods (Random Forest and XGBoost) significantly outperformed individual classifiers, demonstrating the power of combining multiple models.

3. **Decision Tree Performance:** Single Decision Tree showed impressive performance (98.54% accuracy), though slightly below the ensemble models.

4. **Traditional ML Models:** Logistic Regression, kNN, and Naive Bayes showed acceptable but lower performance (80-86% accuracy range).

5. **Model Selection:** For this heart disease classification task, either Random Forest or XGBoost would be the recommended choice for deployment due to their perfect performance metrics.

## Project Structure

```
ML_Assign_2_Model_Classification_Comparision/
│
├── app.py                      # Streamlit web application
├── classification.py            # Main training script
├── dataset_download.py         # Dataset download utility
├── heart.csv                   # Heart disease dataset
├── model_results.json          # Saved model evaluation metrics
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
└── model/                      # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl
```

## Installation and Setup

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train Models
```bash
python classification.py
```

### Run Streamlit App
```bash
streamlit run app.py
```

## Technologies Used

- **Python 3.14**
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization
- **streamlit** - Web application framework
- **joblib** - Model serialization

## Results

All trained models are saved in the `model/` directory and can be loaded for predictions. The results are also saved in `model_results.json` for further analysis.

## Conclusion

This project successfully demonstrates the implementation and comparison of six different machine learning classification algorithms for heart disease prediction. The ensemble methods (Random Forest and XGBoost) proved to be the most effective, achieving perfect classification performance on the test dataset.

## Author

Created as part of ML Assignment 2: Model Classification Comparison

## Date

February 10, 2026
