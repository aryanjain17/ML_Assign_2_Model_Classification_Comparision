import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os
import io

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
page = st.sidebar.radio("Go to", ["📊 Model Comparison", "🔮 Make Prediction", "📈 Dataset Overview", "🧪 Test Your Data"])

# Add download button for dataset in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Dataset")
if dataset is not None:
    csv_data = dataset.to_csv(index=False)
    st.sidebar.download_button(
        label="Download heart.csv",
        data=csv_data,
        file_name="heart_disease_dataset.csv",
        mime="text/csv"
    )
else:
    st.sidebar.info("Dataset not available")

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
        
        # Confusion Matrices for all models
        st.markdown("---")
        st.subheader("🔲 Confusion Matrices for All Models")
        
        if models and dataset is not None:
            try:
                # Prepare test data from dataset
                from sklearn.model_selection import train_test_split
                
                X = dataset.drop('target', axis=1)
                y = dataset['target']
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Get scaler
                scaler = models['scaler']
                X_test_scaled = scaler.transform(X_test)
                
                # Models that need scaled data
                scaled_models = ['Logistic Regression', 'K-Nearest Neighbors', 'Gaussian Naive Bayes']
                
                model_names = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 
                              'Gaussian Naive Bayes', 'Random Forest', 'XGBoost']
                
                # Create 3x2 grid for confusion matrices
                st.markdown("**Confusion matrices show how each model classifies the test data:**")
                
                for row in range(2):  # 2 rows
                    cols = st.columns(3)  # 3 columns per row
                    
                    for col_idx in range(3):
                        model_idx = row * 3 + col_idx
                        if model_idx < len(model_names):
                            model_name = model_names[model_idx]
                            
                            if model_name in models:
                                model = models[model_name]
                                
                                # Use scaled or unscaled data based on model
                                if model_name in scaled_models:
                                    y_pred = model.predict(X_test_scaled)
                                else:
                                    y_pred = model.predict(X_test)
                                
                                # Generate confusion matrix
                                cm = confusion_matrix(y_test, y_pred)
                                
                                with cols[col_idx]:
                                    st.markdown(f"**{model_name}**")
                                    
                                    # Create confusion matrix plot
                                    fig, ax = plt.subplots(figsize=(5, 4))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                               xticklabels=['No Disease', 'Disease'],
                                               yticklabels=['No Disease', 'Disease'],
                                               ax=ax, cbar=False, square=True,
                                               annot_kws={'size': 14, 'weight': 'bold'})
                                    ax.set_ylabel('True Label', fontsize=10)
                                    ax.set_xlabel('Predicted Label', fontsize=10)
                                    ax.set_title(f'{model_name}', fontsize=11, pad=10)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    # Display TN, FP, FN, TP
                                    st.caption(f"TN: {cm[0,0]} | FP: {cm[0,1]} | FN: {cm[1,0]} | TP: {cm[1,1]}")
                
                st.info("💡 **Reading Confusion Matrices**: TN=True Negatives (correctly predicted no disease), FP=False Positives (incorrectly predicted disease), FN=False Negatives (missed disease), TP=True Positives (correctly predicted disease)")
                
            except Exception as e:
                st.error(f"Error generating confusion matrices: {str(e)}")
        else:
            st.warning("Models or dataset not loaded. Cannot generate confusion matrices.")

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

# Page 4: Test Your Data
elif page == "🧪 Test Your Data":
    st.header("Test Your Own Dataset")
    st.markdown("Upload your test data (CSV) and evaluate models with confusion matrix and classification reports")
    
    # File uploader
    st.subheader("📤 Upload Test Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            test_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
            
            # Display uploaded data
            with st.expander("View Uploaded Data"):
                st.dataframe(test_data.head(20), use_container_width=True)
            
            # Check if target column exists
            if 'target' not in test_data.columns:
                st.error("❌ Error: 'target' column not found in the uploaded file!")
                st.info("Your CSV file must include a 'target' column with the true labels.")
            else:
                # Separate features and target
                X_test = test_data.drop('target', axis=1)
                y_test = test_data['target']
                
                st.success(f"✅ Test samples: {len(X_test)}, Features: {len(X_test.columns)}")
                
                # Model selection dropdown
                st.subheader("🎯 Select Model for Evaluation")
                model_names = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 
                              'Gaussian Naive Bayes', 'Random Forest', 'XGBoost']
                
                selected_model = st.selectbox(
                    "Choose a model to evaluate:",
                    model_names,
                    key="model_selector"
                )
                
                if st.button("🚀 Evaluate Model", type="primary"):
                    if models and selected_model in models and 'scaler' in models:
                        with st.spinner(f'Evaluating {selected_model}...'):
                            try:
                                model = models[selected_model]
                                scaler = models['scaler']
                                
                                # Models that need scaled data
                                scaled_models = ['Logistic Regression', 'K-Nearest Neighbors', 'Gaussian Naive Bayes']
                                
                                # Prepare data
                                if selected_model in scaled_models:
                                    X_test_processed = scaler.transform(X_test)
                                else:
                                    X_test_processed = X_test
                                
                                # Make predictions
                                y_pred = model.predict(X_test_processed)
                                y_pred_proba = model.predict_proba(X_test_processed)
                                
                                # Calculate metrics
                                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                                
                                # Display metrics
                                st.success(f"✅ Evaluation Complete for {selected_model}")
                                st.markdown("---")
                                
                                # Metrics in columns
                                st.subheader("📊 Evaluation Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Accuracy", f"{accuracy:.4f}")
                                with col2:
                                    st.metric("Precision", f"{precision:.4f}")
                                with col3:
                                    st.metric("Recall", f"{recall:.4f}")
                                with col4:
                                    st.metric("F1 Score", f"{f1:.4f}")
                                
                                st.markdown("---")
                                
                                # Confusion Matrix and Classification Report side by side
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("🔲 Confusion Matrix")
                                    cm = confusion_matrix(y_test, y_pred)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                               xticklabels=['No Disease', 'Disease'],
                                               yticklabels=['No Disease', 'Disease'],
                                               ax=ax, cbar_kws={'label': 'Count'})
                                    ax.set_ylabel('True Label')
                                    ax.set_xlabel('Predicted Label')
                                    ax.set_title(f'Confusion Matrix - {selected_model}')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Display confusion matrix values
                                    st.markdown("**Confusion Matrix Values:**")
                                    st.markdown(f"- True Negatives (TN): {cm[0, 0]}")
                                    st.markdown(f"- False Positives (FP): {cm[0, 1]}")
                                    st.markdown(f"- False Negatives (FN): {cm[1, 0]}")
                                    st.markdown(f"- True Positives (TP): {cm[1, 1]}")
                                
                                with col2:
                                    st.subheader("📋 Classification Report")
                                    
                                    # Get classification report as dict
                                    report = classification_report(y_test, y_pred, 
                                                                  target_names=['No Disease', 'Disease'],
                                                                  output_dict=True,
                                                                  zero_division=0)
                                    
                                    # Convert to DataFrame for better display
                                    report_df = pd.DataFrame(report).transpose()
                                    
                                    # Format the dataframe
                                    report_df_display = report_df.copy()
                                    for col in report_df_display.columns:
                                        if col == 'support':
                                            report_df_display[col] = report_df_display[col].apply(
                                                lambda x: f"{int(x)}" if not pd.isna(x) else ""
                                            )
                                        else:
                                            report_df_display[col] = report_df_display[col].apply(
                                                lambda x: f"{x:.4f}" if not pd.isna(x) else ""
                                            )
                                    
                                    st.dataframe(report_df_display, use_container_width=True)
                                    
                                    # Download classification report
                                    report_csv = report_df.to_csv()
                                    st.download_button(
                                        label="📥 Download Classification Report",
                                        data=report_csv,
                                        file_name=f"classification_report_{selected_model.replace(' ', '_')}.csv",
                                        mime="text/csv"
                                    )
                                
                                # Additional visualizations
                                st.markdown("---")
                                st.subheader("📈 Additional Insights")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Prediction distribution
                                    st.markdown("**Prediction Distribution**")
                                    pred_counts = pd.Series(y_pred).value_counts()
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    pred_counts.plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'])
                                    ax.set_xlabel('Predicted Class')
                                    ax.set_ylabel('Count')
                                    ax.set_title('Distribution of Predictions')
                                    ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
                                    for i, v in enumerate(pred_counts):
                                        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                with col2:
                                    # True vs Predicted comparison
                                    st.markdown("**Actual vs Predicted Comparison**")
                                    true_counts = pd.Series(y_test).value_counts()
                                    
                                    comparison_df = pd.DataFrame({
                                        'Actual': true_counts.values,
                                        'Predicted': pred_counts.values
                                    }, index=['No Disease', 'Disease'])
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    comparison_df.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
                                    ax.set_xlabel('Class')
                                    ax.set_ylabel('Count')
                                    ax.set_title('Actual vs Predicted Distribution')
                                    ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"❌ Error during evaluation: {str(e)}")
                                st.info("Please ensure your CSV file has the same features as the training data.")
                    else:
                        st.error("Models not loaded properly. Please check the model files.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
    
    else:
        st.info("👆 Please upload a CSV file to begin testing")
        
        # Show expected format
        st.subheader("📝 Expected File Format")
        st.markdown("""
        Your CSV file should have the following structure:
        - **Required columns**: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
        - The **target** column should contain binary values (0 = No Disease, 1 = Disease)
        - All other columns should contain numerical values
        
        **Example:**
        """)
        
        example_data = pd.DataFrame({
            'age': [63, 67, 67],
            'sex': [1, 1, 1],
            'cp': [1, 4, 4],
            'trestbps': [145, 160, 120],
            'chol': [233, 286, 229],
            'fbs': [1, 0, 0],
            'restecg': [2, 2, 2],
            'thalach': [150, 108, 129],
            'exang': [0, 1, 1],
            'oldpeak': [2.3, 1.5, 2.6],
            'slope': [3, 2, 2],
            'ca': [0, 3, 2],
            'thal': [6, 3, 7],
            'target': [0, 1, 1]
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        # Download sample data
        if dataset is not None:
            st.markdown("---")
            st.subheader("📥 Download Sample Test Data")
            st.markdown("You can download the full dataset and use a portion as test data:")
            
            # Create a sample test dataset (20% of original)
            sample_size = int(len(dataset) * 0.2)
            sample_data = dataset.sample(n=sample_size, random_state=42)
            
            csv_sample = sample_data.to_csv(index=False)
            st.download_button(
                label="Download Sample Test Data (20% of full dataset)",
                data=csv_sample,
                file_name="sample_test_data.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Heart Disease Classification ML Project | Created by Aryan Jain using Streamlit</p>
        <p>Models: Logistic Regression | Decision Tree | kNN | Naive Bayes | Random Forest | XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
