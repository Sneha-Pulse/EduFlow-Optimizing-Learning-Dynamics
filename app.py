import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from streamlit_option_menu import option_menu

# Configure Streamlit page
st.set_page_config(page_title="Adaptive Learning Analytics Dashboard", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/HP/Downloads/ADA PROJECT/student_success_dataset.csv')
    return data

df = load_data()

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Home", 
            "Exploratory Data Analysis", 
            "Visualizations", 
            "Student Performance Prediction",
            "Need for Intervention Prediction",
            "Learning Path Optimization",
            "Engagement Prediction",
            "Conclusion"
        ],
        icons=[
            "house", 
            "bar-chart", 
            "pie-chart", 
            "book", 
            "activity", 
            "graph-up-arrow", 
            "people", 
            "check-circle"
        ],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
if selected == "Home":
    st.title("📚 Adaptive Learning Analytics Dashboard")
    st.write("""
    Welcome to the Adaptive Learning Analytics Dashboard. This application provides comprehensive insights into student learning behaviors 
    and outcomes by leveraging advanced data analytics and machine learning techniques. Navigate through the different sections using the sidebar 
    to explore data analysis, visualizations, and predictive models designed to optimize adaptive learning strategies.
    """)
    st.image("https://images.unsplash.com/photo-1584697964153-baba33841056?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", caption="Adaptive Learning", use_column_width=True)
    st.markdown("""
    **Key Features:**
    - **Exploratory Data Analysis:** Understand the distribution and relationships within the data.
    - **Interactive Visualizations:** Gain insights through interactive charts and plots.
    - **Predictive Modeling:** Predict student performance, intervention needs, learning path effectiveness, and engagement levels.
    - **Model Evaluation:** Compare different machine learning models and evaluate their performance using various metrics.
    """)

# Exploratory Data Analysis Page
elif selected == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.subheader("Dataset Overview")
    
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Correlation Matrix")
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Visualizations Page
elif selected == "Visualizations":
    st.title("📈 Data Visualizations")
    
    st.subheader("Distribution of Correct Response Rate")
    fig, ax = plt.subplots()
    sns.histplot(df['Correct Response Rate'], kde=True, bins=30, color='skyblue', ax=ax)
    st.pyplot(fig)

    st.subheader("Response Time vs. Correct Response Rate")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Response Time (sec)', 
        y='Correct Response Rate', 
        hue='Confidence Level', 
        data=df, 
        palette='viridis', 
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Number of Tutor Sessions by Content Difficulty Level")
    fig, ax = plt.subplots()
    sns.boxplot(
        x='Content Difficulty Level', 
        y='Number of Tutor Sessions', 
        data=df, 
        palette='pastel', 
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Engagement with Supplemental Resources")
    fig, ax = plt.subplots()
    sns.violinplot(
        x='Adaptive Learning Feature Used', 
        y='Engagement with Supplemental Resources', 
        data=df, 
        palette='muted', 
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Personalization Score Distribution by Recommended Learning Path")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(
        x='Recommended Learning Path', 
        y='Personalization Score', 
        data=df, 
        palette='deep', 
        estimator=np.mean, 
        ci='sd', 
        ax=ax
    )
    st.pyplot(fig)

# Student Performance Prediction Page
elif selected == "Student Performance Prediction":
    st.title("🎯 Student Performance Prediction")
    st.write("""
    Predict future academic performance to identify students who may be at risk of underperforming using:
    - **Correct Response Rate**
    - **Response Time**
    - **Confidence Level**
    """)
    
    # Prepare Data
    df_performance = df.copy()
    df_performance['Performance_Label'] = np.where(df_performance['Correct Response Rate'] >= 75, 1, 0)  # 1: Good Performance, 0: Poor Performance

    # Encode Categorical Variables
    le_confidence = LabelEncoder()
    df_performance['Confidence Level'] = le_confidence.fit_transform(df_performance['Confidence Level'])

    # Features and Target
    X = df_performance[['Correct Response Rate', 'Response Time (sec)', 'Confidence Level']]
    y = df_performance['Performance_Label']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the scaler
    scaler = StandardScaler()

    # Feature Scaling
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Selection
    st.subheader("Select Classification Model")
    model_option = st.selectbox(
        "Choose a machine learning model:",
        ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost")
    )

    # Initialize Model
    if model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC Score": roc_auc
    }

    st.write(pd.DataFrame(metrics, index=[0]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Additional Pages
elif selected == "Need for Intervention Prediction":
    st.title("🚨 Need for Intervention Prediction")
    # Content for this page

elif selected == "Learning Path Optimization":
    st.title("📘 Learning Path Optimization")
    # Content for this page

elif selected == "Engagement Prediction":
    st.title("📈 Engagement Prediction")
    # Content for this page

elif selected == "Conclusion":
    st.title("✅ Conclusion")
    st.write("""
    This dashboard has provided a detailed analysis of student performance and engagement through various predictive models and visualizations.
    Use the insights gained to tailor adaptive learning strategies and interventions effectively.
    """)

# Need for Intervention Prediction Page
elif selected == "Need for Intervention Prediction":
    st.title("🚨 Need for Intervention Prediction")
    st.write("""
    Predict the need for student interventions based on their engagement and performance metrics. 
    This section uses predictive models to determine students who might need additional support.
    """)

    # Data Preparation
    df_intervention = df.copy()
    df_intervention['Intervention_Needed'] = np.where(df_intervention['Engagement Score'] < 50, 1, 0)  # 1: Need Intervention, 0: No Intervention

    # Features and Target
    X_intervention = df_intervention[['Correct Response Rate', 'Response Time (sec)', 'Confidence Level', 'Engagement Score']]
    y_intervention = df_intervention['Intervention_Needed']

    # Train-Test Split
    X_train_intervention, X_test_intervention, y_train_intervention, y_test_intervention = train_test_split(X_intervention, y_intervention, test_size=0.2, random_state=42)

    # Feature Scaling
    X_train_intervention = scaler.fit_transform(X_train_intervention)
    X_test_intervention = scaler.transform(X_test_intervention)

    # Model Selection
    st.subheader("Select Intervention Model")
    model_option_intervention = st.selectbox(
        "Choose a machine learning model for intervention prediction:",
        ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost")
    )

    # Initialize Model
    if model_option_intervention == "Logistic Regression":
        model_intervention = LogisticRegression()
    elif model_option_intervention == "Decision Tree":
        model_intervention = DecisionTreeClassifier()
    elif model_option_intervention == "Random Forest":
        model_intervention = RandomForestClassifier()
    elif model_option_intervention == "XGBoost":
        model_intervention = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train Model
    model_intervention.fit(X_train_intervention, y_train_intervention)
    y_pred_intervention = model_intervention.predict(X_test_intervention)
    y_proba_intervention = model_intervention.predict_proba(X_test_intervention)[:,1]

    # Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    accuracy_intervention = accuracy_score(y_test_intervention, y_pred_intervention)
    precision_intervention = precision_score(y_test_intervention, y_pred_intervention)
    recall_intervention = recall_score(y_test_intervention, y_pred_intervention)
    f1_intervention = f1_score(y_test_intervention, y_pred_intervention)
    roc_auc_intervention = roc_auc_score(y_test_intervention, y_proba_intervention)

    metrics_intervention = {
        "Accuracy": accuracy_intervention,
        "Precision": precision_intervention,
        "Recall": recall_intervention,
        "F1-Score": f1_intervention,
        "ROC AUC Score": roc_auc_intervention
    }

    st.write(pd.DataFrame(metrics_intervention, index=[0]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix_intervention = confusion_matrix(y_test_intervention, y_pred_intervention)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_intervention, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr_intervention, tpr_intervention, thresholds_intervention = roc_curve(y_test_intervention, y_proba_intervention)
    fig, ax = plt.subplots()
    ax.plot(fpr_intervention, tpr_intervention, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_intervention)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Learning Path Optimization Page
elif selected == "Learning Path Optimization":
    st.title("📘 Learning Path Optimization")
    st.write("""
    Optimize learning paths for students by analyzing their performance and engagement data. 
    This section uses machine learning models to recommend the most effective learning paths.
    """)

    # Data Preparation
    df_learning_path = df.copy()
    df_learning_path['Learning Path Recommended'] = np.where(df_learning_path['Performance Score'] >= 80, 'Path A', 'Path B')  # Example logic

    # Features and Target
    X_learning_path = df_learning_path[['Correct Response Rate', 'Response Time (sec)', 'Confidence Level', 'Engagement Score']]
    y_learning_path = df_learning_path['Learning Path Recommended']

    # Encode Categorical Variables
    le_path = LabelEncoder()
    y_learning_path = le_path.fit_transform(y_learning_path)

    # Train-Test Split
    X_train_learning_path, X_test_learning_path, y_train_learning_path, y_test_learning_path = train_test_split(X_learning_path, y_learning_path, test_size=0.2, random_state=42)

    # Feature Scaling
    X_train_learning_path = scaler.fit_transform(X_train_learning_path)
    X_test_learning_path = scaler.transform(X_test_learning_path)

    # Model Selection
    st.subheader("Select Learning Path Optimization Model")
    model_option_learning_path = st.selectbox(
        "Choose a machine learning model for learning path optimization:",
        ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost")
    )

    # Initialize Model
    if model_option_learning_path == "Logistic Regression":
        model_learning_path = LogisticRegression()
    elif model_option_learning_path == "Decision Tree":
        model_learning_path = DecisionTreeClassifier()
    elif model_option_learning_path == "Random Forest":
        model_learning_path = RandomForestClassifier()
    elif model_option_learning_path == "XGBoost":
        model_learning_path = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train Model
    model_learning_path.fit(X_train_learning_path, y_train_learning_path)
    y_pred_learning_path = model_learning_path.predict(X_test_learning_path)
    y_proba_learning_path = model_learning_path.predict_proba(X_test_learning_path)[:,1]

    # Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    accuracy_learning_path = accuracy_score(y_test_learning_path, y_pred_learning_path)
    precision_learning_path = precision_score(y_test_learning_path, y_pred_learning_path, average='weighted')
    recall_learning_path = recall_score(y_test_learning_path, y_pred_learning_path, average='weighted')
    f1_learning_path = f1_score(y_test_learning_path, y_pred_learning_path, average='weighted')
    roc_auc_learning_path = roc_auc_score(y_test_learning_path, y_proba_learning_path, multi_class='ovr')

    metrics_learning_path = {
        "Accuracy": accuracy_learning_path,
        "Precision": precision_learning_path,
        "Recall": recall_learning_path,
        "F1-Score": f1_learning_path,
        "ROC AUC Score": roc_auc_learning_path
    }

    st.write(pd.DataFrame(metrics_learning_path, index=[0]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix_learning_path = confusion_matrix(y_test_learning_path, y_pred_learning_path)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_learning_path, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr_learning_path, tpr_learning_path, thresholds_learning_path = roc_curve(y_test_learning_path, y_proba_learning_path)
    fig, ax = plt.subplots()
    ax.plot(fpr_learning_path, tpr_learning_path, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_learning_path)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Engagement Prediction Page
elif selected == "Engagement Prediction":
    st.title("📈 Engagement Prediction")
    st.write("""
    Predict student engagement levels based on various academic and behavioral metrics. 
    This section focuses on analyzing factors that contribute to higher engagement.
    """)

    # Data Preparation
    df_engagement = df.copy()
    df_engagement['Engagement Level'] = np.where(df_engagement['Engagement Score'] > 75, 'High', 'Low')  # Example logic

    # Features and Target
    X_engagement = df_engagement[['Correct Response Rate', 'Response Time (sec)', 'Confidence Level', 'Personalization Score']]
    y_engagement = df_engagement['Engagement Level']

    # Encode Categorical Variables
    le_engagement = LabelEncoder()
    y_engagement = le_engagement.fit_transform(y_engagement)

    # Train-Test Split
    X_train_engagement, X_test_engagement, y_train_engagement, y_test_engagement = train_test_split(X_engagement, y_engagement, test_size=0.2, random_state=42)

    # Feature Scaling
    X_train_engagement = scaler.fit_transform(X_train_engagement)
    X_test_engagement = scaler.transform(X_test_engagement)

    # Model Selection
    st.subheader("Select Engagement Model")
    model_option_engagement = st.selectbox(
        "Choose a machine learning model for engagement prediction:",
        ("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost")
    )

    # Initialize Model
    if model_option_engagement == "Logistic Regression":
        model_engagement = LogisticRegression()
    elif model_option_engagement == "Decision Tree":
        model_engagement = DecisionTreeClassifier()
    elif model_option_engagement == "Random Forest":
        model_engagement = RandomForestClassifier()
    elif model_option_engagement == "XGBoost":
        model_engagement = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train Model
    model_engagement.fit(X_train_engagement, y_train_engagement)
    y_pred_engagement = model_engagement.predict(X_test_engagement)
    y_proba_engagement = model_engagement.predict_proba(X_test_engagement)[:,1]

    # Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    accuracy_engagement = accuracy_score(y_test_engagement, y_pred_engagement)
    precision_engagement = precision_score(y_test_engagement, y_pred_engagement, average='weighted')
    recall_engagement = recall_score(y_test_engagement, y_pred_engagement, average='weighted')
    f1_engagement = f1_score(y_test_engagement, y_pred_engagement, average='weighted')
    roc_auc_engagement = roc_auc_score(y_test_engagement, y_proba_engagement, multi_class='ovr')

    metrics_engagement = {
        "Accuracy": accuracy_engagement,
        "Precision": precision_engagement,
        "Recall": recall_engagement,
        "F1-Score": f1_engagement,
        "ROC AUC Score": roc_auc_engagement
    }

    st.write(pd.DataFrame(metrics_engagement, index=[0]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix_engagement = confusion_matrix(y_test_engagement, y_pred_engagement)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_engagement, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr_engagement, tpr_engagement, thresholds_engagement = roc_curve(y_test_engagement, y_proba_engagement)
    fig, ax = plt.subplots()
    ax.plot(fpr_engagement, tpr_engagement, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_engagement)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
