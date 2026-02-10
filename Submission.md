# ML Assignment 2 – Submission

**Student Name:** Aryan Jain  
**BITS ID:** 2025AB05273

---

## 1. GitHub Repository Link

**Repository URL:**  
https://github.com/aryanjain17/ML_Assign_2_Model_Classification_Comparision


## 2. Live Streamlit App Link

**Deployed App URL (Streamlit Community Cloud):**  
https://aryan-jain-ml-assign-2-modelclassificationcomparision.streamlit.app/


## 3. Screenshot – Execution on BITS Virtual Lab

<img width="2838" height="1720" alt="image" src="https://github.com/user-attachments/assets/fb718fde-b758-499b-82a1-96994c7bbd6d" />


---

## 4. README Content

# **Heart Disease Classification - ML Model Comparison**

### a. Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and accurate prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. This project aims to predict the presence of heart disease in patients based on various medical attributes using multiple machine learning classification algorithms. The goal is to compare the performance of six different ML models and identify the most effective approach for heart disease classification.

---

### b. Dataset Description

**Dataset Name:** Heart Disease UCI Dataset  
**Source:** UCI Machine Learning Repository / Kaggle  
**Dataset Size:** 1,025 samples  
**Number of Features:** 13 independent features + 1 target variable

**Features Description:**

1. **age**: Age of the patient (in years)  
2. **sex**: Sex of the patient (1 = male, 0 = female)  
3. **cp**: Chest pain type (0–3)  
   - 0: Typical angina  
   - 1: Atypical angina  
   - 2: Non-anginal pain  
   - 3: Asymptomatic  
4. **trestbps**: Resting blood pressure (mm Hg)  
5. **chol**: Serum cholesterol level (mg/dl)  
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
7. **restecg**: Resting electrocardiographic results (0–2)  
8. **thalach**: Maximum heart rate achieved  
9. **exang**: Exercise induced angina (1 = yes, 0 = no)  
10. **oldpeak**: ST depression induced by exercise relative to rest  
11. **slope**: Slope of the peak exercise ST segment (0–2)  
12. **ca**: Number of major vessels colored by fluoroscopy (0–3)  
13. **thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)

**Target Variable:**  
- **target**: Presence of heart disease (0 = no disease, 1 = disease present)

**Class Distribution:**  
- Class 0 (No Heart Disease): 499 samples (48.7%)  
- Class 1 (Heart Disease Present): 526 samples (51.3%)

---

### c. Models Used and Comparison Table

This project implements and compares six different machine learning classification algorithms on the same dataset:

1. **Logistic Regression** – Linear model for binary classification  
2. **Decision Tree Classifier** – Tree-based model using feature splits  
3. **K-Nearest Neighbors (kNN)** – Distance-based non-parametric classifier  
4. **Naive Bayes (Gaussian)** – Probabilistic model based on Bayes' theorem  
5. **Random Forest (Ensemble)** – Ensemble of decision trees with bagging  
6. **XGBoost (Ensemble)** – Gradient boosting ensemble with regularization  

#### Comparison Table: Evaluation Metrics for All Models

| ML Model Name                 | Accuracy |   AUC  | Precision | Recall | F1 Score |   MCC  |
|------------------------------|:--------:|:------:|:---------:|:------:|:--------:|:------:|
| Logistic Regression          |  0.8098  | 0.9298 |  0.7619   | 0.9143 |  0.8312  | 0.6309 |
| Decision Tree                |  0.9854  | 0.9857 |  1.0000   | 0.9714 |  0.9855  | 0.9712 |
| k-Nearest Neighbors (kNN)    |  0.8634  | 0.9629 |  0.8738   | 0.8571 |  0.8654  | 0.7269 |
| Gaussian Naive Bayes         |  0.8293  | 0.9043 |  0.8070   | 0.8762 |  0.8402  | 0.6602 |
| Random Forest (Ensemble)     |  1.0000  | 1.0000 |  1.0000   | 1.0000 |  1.0000  | 1.0000 |
| XGBoost (Ensemble)           |  1.0000  | 1.0000 |  1.0000   | 1.0000 |  1.0000  | 1.0000 |

*(Metrics computed on the test split; AUC = Area Under the ROC Curve, MCC = Matthews Correlation Coefficient.)*

---

#### Observations on Model Performance

| ML Model Name              | Observation about model performance |
|---------------------------|--------------------------------------|
| **Logistic Regression**   | Shows decent performance with 80.98% accuracy. High recall (91.43%) indicates strong positive-case detection, while lower precision (76.19%) suggests more false positives. Serves as a good baseline model with a solid AUC of 0.93. |
| **Decision Tree**         | Delivers 98.54% accuracy with perfect precision (100%) and strong MCC (0.97). Slight overfitting may exist because recall dips to 97.14%, but performance remains outstanding overall. |
| **kNN**                   | Achieves balanced precision (87.38%) and recall (85.71%) with 86.34% accuracy. Performance depends on proper feature scaling and k selection, but it remains a reliable non-parametric baseline. |
| **Naive Bayes**           | Posts 82.93% accuracy with solid recall (87.62%). The independence assumption limits precision (80.70%), yet the classifier is lightweight and offers quick probability estimates. |
| **Random Forest (Ens.)**  | Exhibits perfect metrics across the board, highlighting the strength of ensemble averaging and feature bagging. No signs of overfitting on the test split, making it the top classical choice. |
| **XGBoost (Ens.)**        | Matches Random Forest with flawless results due to gradient-boosted trees with built-in regularization. Handles complex feature interactions effectively and is equally suitable for deployment. |
