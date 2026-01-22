import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #FF4B4B;
    }
    h2 {
        color: #0068C9;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Heart Disease Classification Comparison")
st.markdown("### Compare 6 Machine Learning Models for Heart Disease Prediction")
st.markdown("---")

# Load models and results
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbors': 'model/knn.pkl',
        'Gaussian Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    try:
        for name, file_path in model_files.items():
            if os.path.exists(file_path):
                models[name] = joblib.load(file_path)
        
        # Load scaler
        scaler = joblib.load('model/scaler.pkl')
        models['scaler'] = scaler
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_results():
    """Load model evaluation results"""
    try:
        with open('model_results.json', 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

@st.cache_data
def load_dataset():
    """Load the heart disease dataset"""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load data
models = load_models()
results = load_results()
dataset = load_dataset()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Model Comparison", "🔮 Make Prediction", "📈 Dataset Overview"])

# Page 1: Model Comparison
if page == "📊 Model Comparison":
    st.header("Model Performance Comparison")
    
    if results:
        # Create DataFrame from results
        df_results = pd.DataFrame(results).T
        
        # Display metrics table
        st.subheader("Evaluation Metrics for All Models")
        
        # Format the dataframe
        df_display = df_results.copy()
        for col in df_display.columns:
            if col != 'AUC Score':
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(df_display, use_container_width=True)
        
        # Visualization
        st.subheader("Visual Comparison of Models")
        
        # Create columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for accuracy
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            df_results['Accuracy'].plot(kind='barh', ax=ax1, color='skyblue')
            ax1.set_xlabel('Accuracy Score')
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_xlim([0, 1.05])
            for i, v in enumerate(df_results['Accuracy']):
                ax1.text(v + 0.01, i, f'{v:.4f}', va='center')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # Bar chart for F1 Score
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df_results['F1 Score'].plot(kind='barh', ax=ax2, color='lightcoral')
            ax2.set_xlabel('F1 Score')
            ax2.set_title('Model F1 Score Comparison')
            ax2.set_xlim([0, 1.05])
            for i, v in enumerate(df_results['F1 Score']):
                ax2.text(v + 0.01, i, f'{v:.4f}', va='center')
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Multi-metric comparison
        st.subheader("All Metrics Comparison")
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
        df_plot = df_results[metrics_to_plot]
        
        df_plot.plot(kind='bar', ax=ax3)
        ax3.set_ylabel('Score')
        ax3.set_title('Comprehensive Model Performance Comparison')
        ax3.set_ylim([0, 1.1])
        ax3.legend(loc='lower right')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Best model highlight
        st.subheader(" Best Performing Models")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_accuracy = df_results['Accuracy'].idxmax()
            best_acc_value = df_results['Accuracy'].max()
            st.metric("Highest Accuracy", best_accuracy, f"{best_acc_value:.4f}")
        
        with col2:
            best_f1 = df_results['F1 Score'].idxmax()
            best_f1_value = df_results['F1 Score'].max()
            st.metric("Highest F1 Score", best_f1, f"{best_f1_value:.4f}")
        
        with col3:
            best_mcc = df_results['MCC Score'].idxmax()
            best_mcc_value = df_results['MCC Score'].max()
            st.metric("Highest MCC Score", best_mcc, f"{best_mcc_value:.4f}")

# Page 2: Make Prediction
elif page == "🔮 Make Prediction":
    st.header("Predict Heart Disease")
    st.markdown("Enter patient information to predict heart disease using different models")
    
    if models and 'scaler' in models:
        # Create input form
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                               format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", options=[1, 2, 3], 
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
        
        # Predict button
        if st.button("🔍 Predict", type="primary"):
            # Prepare input data
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                   thalach, exang, oldpeak, slope, ca, thal]])
            
            # Scale the input data
            scaler = models['scaler']
            input_scaled = scaler.transform(input_data)
            
            st.subheader("Prediction Results")
            
            # Create columns for results
            results_cols = st.columns(3)
            
            model_names = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 
                          'Gaussian Naive Bayes', 'Random Forest', 'XGBoost']
            
            # Models that need scaled data
            scaled_models = ['Logistic Regression', 'K-Nearest Neighbors', 'Gaussian Naive Bayes']
            
            predictions = {}
            for i, model_name in enumerate(model_names):
                if model_name in models:
                    model = models[model_name]
                    
                    # Use scaled or unscaled data based on model
                    if model_name in scaled_models:
                        prediction = model.predict(input_scaled)[0]
                        proba = model.predict_proba(input_scaled)[0]
                    else:
                        prediction = model.predict(input_data)[0]
                        proba = model.predict_proba(input_data)[0]
                    
                    predictions[model_name] = {
                        'prediction': prediction,
                        'probability': proba[1] * 100
                    }
                    
                    # Display in columns
                    with results_cols[i % 3]:
                        if prediction == 1:
                            st.error(f"**{model_name}**")
                            st.error(f"⚠️ Heart Disease Detected")
                            st.error(f"Confidence: {proba[1]*100:.2f}%")
                        else:
                            st.success(f"**{model_name}**")
                            st.success(f"✅ No Heart Disease")
                            st.success(f"Confidence: {proba[0]*100:.2f}%")
            
            # Consensus prediction
            st.markdown("---")
            positive_predictions = sum([1 for p in predictions.values() if p['prediction'] == 1])
            total_predictions = len(predictions)
            
            st.subheader("📊 Consensus Prediction")
            if positive_predictions > total_predictions / 2:
                st.error(f"⚠️ **MAJORITY CONSENSUS: Heart Disease Detected**")
                st.error(f"{positive_predictions} out of {total_predictions} models predict heart disease")
            else:
                st.success(f"✅ **MAJORITY CONSENSUS: No Heart Disease**")
                st.success(f"{total_predictions - positive_predictions} out of {total_predictions} models predict no heart disease")
    
    else:
        st.error("Models not loaded. Please ensure the models are trained and saved.")

# Page 3: Dataset Overview
elif page == "📈 Dataset Overview":
    st.header("Dataset Overview")
    
    if dataset is not None:
        st.subheader("Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(dataset))
        with col2:
            st.metric("Number of Features", len(dataset.columns) - 1)
        with col3:
            st.metric("Target Classes", dataset['target'].nunique())
        
        # Display dataset sample
        st.subheader("Sample Data")
        st.dataframe(dataset.head(10), use_container_width=True)
        
        # Class distribution
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            class_counts = dataset['target'].value_counts()
            st.write(class_counts)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            class_counts.plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'])
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Target Classes')
            ax.set_xticklabels(['No Disease (0)', 'Disease (1)'], rotation=0)
            for i, v in enumerate(class_counts):
                ax.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Feature statistics
            st.subheader("Feature Statistics")
            st.dataframe(dataset.describe().T, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(14, 10))
        correlation = dataset.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        st.error("Dataset not loaded. Please ensure 'heart.csv' is in the project directory.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Heart Disease Classification ML Project | Created by Aryan Jain using Streamlit</p>
        <p>Models: Logistic Regression | Decision Tree | kNN | Naive Bayes | Random Forest | XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
