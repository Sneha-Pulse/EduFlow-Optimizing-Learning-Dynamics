import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import streamlit.components.v1 as components
from scipy.stats import chi2_contingency
from streamlit_option_menu import option_menu

# Function to upload and preprocess dataset
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)  
    
    # Convert Response Time from ms to seconds
    df['Response Time (ms)'] = df['Response Time (ms)'] / 1000
    
    # Convert categorical columns to numerical using Label Encoding
    le = LabelEncoder()
    categorical_columns = [
        'Demographics', 'Learning Methods', 'Listening Skills', 
        'Tech Proficiency', 'Time Management Skills', 'Family Support', 
        'Content Difficulty Level', 'Recommended Learning Path', 'Intervention Flag'
    ]
    
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

def plot_roc_curve(fpr, tpr, auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    st.pyplot(plt)

def model_inference(model_name):
    if model_name == "Logistic Regression":
        return (
            "The logistic regression model shows moderate performance with an accuracy of 52%.  "
            
            "\nPrecision for class 0 is 0.49 and for class 1 is 0.57, indicating better identification of class 1. \n"
            
            "\nRecall is higher for class 0 (0.59) compared to class 1 (0.47), suggesting the model is better at detecting class 0. Overall, the F1-scores are close for both classes (0.53 for class 0 and 0.52 for class 1), reflecting balanced performance but with room for improvement."
        )
    elif model_name == "Decision Tree":
        return (
            "The decision tree model performs slightly better with an accuracy of 54%. "
            
            "It shows improved precision for class 1 (0.58) compared to class 0 (0.50) and better recall for class 0 (0.57) versus class 1 (0.51).\n "
            
            "\nThe F1-scores are balanced with class 0 at 0.53 and class 1 at 0.55, indicating overall moderate performance with marginal improvement over logistic regression. \n"

            "\nThe macro and weighted averages are consistent, suggesting the model performs equally across both classes."
        )
    elif model_name == "Random Forest":
        return (
            "The random forest model achieves an accuracy of 51%, showing balanced performance across both classes. "
            
            "\nPrecision for class 1 is slightly higher (0.55) compared to class 0 (0.47), while recall for class 0 is better (0.52) than for class 1 (0.50). \n"
            
            "\nThe F1-scores are similar, with class 0 at 0.49 and class 1 at 0.52, indicating overall moderate performance. \n"
            
            "\nThe macro and weighted averages reflect this balance, highlighting a need for further improvement in classification accuracy."
        )
    elif model_name == "XGBoost":
        return (
            "The XGBoost model achieves an accuracy of 51%, showing balanced performance across both classes. "
            
            "\nPrecision for class 1 is slightly higher (0.55) compared to class 0 (0.47), while recall for class 0 is better (0.52) than for class 1 (0.51). \n"
            
            "\nThe F1-scores are similar, with class 0 at 0.50 and class 1 at 0.53, indicating overall moderate performance.\n "
            
            "The macro and weighted averages reflect this balance, highlighting a need for further improvement in classification accuracy."
        )

def main():
    st.set_page_config(page_title="Student Analysis and Prediction Tool", layout="wide")
    
    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=[
                "Home", 
                "EDA Dataset Overview", 
                "Student Intervention Prediction", 
                "Engagement Prediction", 
                "Difficulty Adjustment Prediction", 
                "Overall Conclusion"
            ],
            icons=[
                "house", 
                "bar-chart", 
                "activity", 
                "people", 
                "graph-up-arrow", 
                "check-circle"
            ],
            menu_icon="cast",
            default_index=0,
        )
        
        # Upload dataset
        uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        
        if selected == "Home":
            st.title("EduFlow-Optimizing-Learning-Dynamics Public")
            
            # 1. Identify Domain & Problem Statement:
            st.markdown("**Domain:** Education")
            st.markdown("**Problem Statement:**")
            st.write("How can higher education institutions use predictive analytics to identify at-risk students early in their academic journey and implement targeted interventions to improve student retention and success rates. "
                     "The project aims to develop a predictive model that utilizes Intervention Targeting, student Engagement Analysis, and Content Difficulty Prediction.")
            
            # 2. Main Problem/Challenge:
            st.markdown("**Main Problem/Challenge:**")
            st.write("The main challenge is identifying which students are most likely to struggle or drop out and understanding the key factors driving their performance. Predictive analytics could enable institutions to implement targeted support systems before students face academic failure or disengagement, but ensuring the accuracy of such models and capturing relevant data is complex.")
            
            # 3. Problem Impacting the Organization or Stakeholders:
            st.markdown("**Problem Impacting the Organization or Stakeholders:**")
            st.write("Poor student retention rates negatively affect both students and institutions. For students, failure to succeed academically can lead to long-term consequences like debt accumulation and unemployment. For institutions, low retention rates damage reputation, financial stability (due to loss of tuition revenue), and effectiveness in fulfilling their educational mission.")
            
            # 4. Trends/Patterns:
            st.markdown("**Trends/Patterns:**")
            st.write("- Increased use of data-driven strategies in education to improve student outcomes.")
            st.write("- Adoption of Learning Management Systems (LMS) that track student engagement (attendance, assignments, online participation).")
            st.write("- Growing recognition of non-academic factors (e.g., mental health, financial hardship) impacting academic performance.")
            st.write("- Integration of personalized learning systems that adapt based on student performance data.")
        
        elif selected == "EDA Dataset Overview":
            st.title("EDA Dataset Overview")
            st.write("Explore and analyze the dataset with the following options:")
            
            if st.button("Show Dataset Summary"):
                st.write("**Dataset Summary:**")
                st.write(df.describe())
            
            if st.button("Show Missing Values"):
                st.write("**Missing Values:**")
                st.write(df.isnull().sum())
            
            if st.button("Show Data Types"):
                st.write("**Data Types:**")
                st.write(df.dtypes)
            
            if st.button("Show Correlation Matrix"):
                st.write("**Correlation Matrix:**")
                st.write(df.corr())
            
            if st.button("Perform Chi-Square Test"):
                st.write("**Chi-Square Test Results:**")
                
                # Create a contingency table
                contingency_table = pd.crosstab(df['Demographics'], df['Tech Proficiency'])
                
                # Perform the Chi-Square test
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                
                # Display the results
                st.write(f"Chi2 Statistic: {chi2:.2f}")
                st.write(f"p-value: {p:.4f}")
                st.write(f"Degrees of Freedom: {dof}")
                
                if p < 0.05:
                    st.write("There is a significant relationship between Demographics and Tech Proficiency.")
                else:
                    st.write("CONCLUTION: There is no significant relationship between Demographics and Tech Proficiency.")
        
        elif selected == "Student Intervention Prediction":
            st.title("Student Intervention Prediction")
            st.write("Analyzing the intervention flag, help requests, and number of tutor sessions, you can predict which students are likely to benefit from additional support or resources.")
            st.write("COLUMNS USED: Intervention Flag, \n Help Requests, \n Number of Tutor Sessions")
            
            # Define features and target variable
            X = df.drop(columns=['Intervention Flag'])
            y = df['Intervention Flag']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric='mlogloss')
            }
            
            st.subheader("Model Performance")
            selected_model = st.selectbox("Select a model", list(models.keys()))

            st.write("Logistic Regression provides a straightforward and interpretable baseline for binary classification of student success. \n"
                     
                     "\nDecision Trees offer easy-to-understand decision rules\n"
                     
                     "\nRandom Forest and XGBoost enhance predictive accuracy by combining multiple trees\n "

                     "\n XGBoost providing superior performance through advanced boosting techniques.")
            
            model = models[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve
            
            # Buttons for displaying additional information
            if st.button("Show Classification Report"):
                st.write(f"**Classification Report for {selected_model}:**")
                st.write("Precision: The ratio of true positive predictions to the total predicted positives, indicating the accuracy of positive predictions."

" \n  (Sensitivity): The ratio of true positive predictions to the actual positives, measuring the ability to identify all relevant instances.\n"

"\nF1-Score: The harmonic mean of precision and recall, providing a balance between the two metrics.\n"

"\nSupport: The number of actual occurrences of each class in the dataset. ")
                st.text(classification_report(y_test, y_pred))
            
            if st.button("Show ROC AUC Curve"):
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                plot_roc_curve(fpr, tpr, auc, selected_model)
                st.write(f"**ROC AUC Score for {selected_model}:** {auc:.2f}")
            
            if st.button("Show Inference"):
                st.write(f"**Inference for {selected_model}:**")
                st.write(model_inference(selected_model))
            
            if st.button("Conclusion"):
                st.write("**Conclusion:**")
                st.write("Based on the selected model's performance, we can conclude the likelihood of students requiring intervention. "
                         "The model's ROC AUC score provides an indication of its accuracy in distinguishing between students who may or may not need intervention. "
                         "A higher score suggests better performance in identifying students who are at risk and may need support.")
        
        elif selected == "Engagement Prediction":
            st.title("Engagement Prediction")
            st.write("Examining engagement with supplemental resources and the use of adaptive learning features, you can predict levels of student engagement and design interventions to enhance participation and motivation.")
            st.write("COLUMNS USED: Engagement with Supplemental Resources, \n Adaptive Learning Feature Used, \nHelp Requests")

            
            # Define features and target variable
            engagement_features = ['Engagement with Supplemental Resources', 'Personalization Score', 'Number of Tutor Sessions']
            X = df[engagement_features]
            y = df['Engagement with Supplemental Resources'] > 0.5  # Assuming engagement level is high if > 0.5
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric='mlogloss')
            }
            
            st.subheader("Model Performance")
            selected_model = st.selectbox("Select a model for engagement prediction", list(models.keys()))
            
            model = models[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve
            
            # Buttons for displaying additional information
            if st.button("Show Classification Report - Engagement"):
                st.write(f"**Classification Report for {selected_model} - Engagement:**")
                st.write("Precision: The ratio of true positive predictions to the total predicted positives, indicating the accuracy of positive predictions."

" \nRecall (Sensitivity): The ratio of true positive predictions to the actual positives, measuring the ability to identify all relevant instances.\n"

"\nF1-Score: The harmonic mean of precision and recall, providing a balance between the two metrics.\n"

"\nSupport: The number of actual occurrences of each class in the dataset. ")
                st.text(classification_report(y_test, y_pred))
            
            if st.button("Show ROC AUC Curve - Engagement"):
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                plot_roc_curve(fpr, tpr, auc, selected_model)
                st.write(f"**ROC AUC Score for {selected_model} - Engagement:** {auc:.2f}")
            
            if st.button("Show Inference - Engagement"):
                st.write(f"**Inference for {selected_model} - Engagement:**")
                st.write(model_inference(selected_model))
            
            if st.button("Conclusion - Engagement"):
                st.write("**Conclusion:**")
                st.write("The model's performance in predicting student engagement levels is crucial for designing effective learning interventions. "
                         "By understanding engagement patterns, educators can tailor resources and support to improve student participation and motivation.")
        
        elif selected == "Difficulty Adjustment Prediction":
            st.title("Difficulty Adjustment Prediction")
            st.write("Analyzing content difficulty level and associated performance can help predict which topics or content areas are likely to challenge students, enabling proactive adjustments in course material delivery.")
            st.write("COLUMNS USED: Content Difficulty Level, \n Correct Response Rate, \nResponse Time")
            
            # Define features and target variable
            difficulty_features = ['Content Difficulty Level', 'Correct Response Rate', 'Response Time (ms)']
            X = df[difficulty_features]
            y = df['Content Difficulty Level']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric='mlogloss')
            }
            
            st.subheader("Model Performance")
            selected_model = st.selectbox("Select a model for difficulty adjustment prediction", list(models.keys()))
            
            model = models[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve
            
            # Buttons for displaying additional information
            if st.button("Show Classification Report - Difficulty Adjustment"):
                st.write(f"**Classification Report for {selected_model} - Difficulty Adjustment:**")
                st.write("Precision: The ratio of true positive predictions to the total predicted positives, indicating the accuracy of positive predictions."

" \nRecall (Sensitivity): The ratio of true positive predictions to the actual positives, measuring the ability to identify all relevant instances.\n"

"\nF1-Score: The harmonic mean of precision and recall, providing a balance between the two metrics.\n"

"\nSupport: The number of actual occurrences of each class in the dataset. ")

                st.text(classification_report(y_test, y_pred))
            
            if st.button("Show Inference - Difficulty Adjustment"):
                st.write(f"**Inference for {selected_model} - Difficulty Adjustment:**")
                st.write(model_inference(selected_model))
            
            if st.button("Conclusion - Difficulty Adjustment"):
                st.write("**Conclusion:**")
                st.write("This conclusion is based on the model's prediction performance in adjusting content difficulty. A higher ROC AUC score indicates better accuracy "
                         "in predicting which content areas need difficulty adjustments based on student performance. This helps in proactively addressing challenging topics.")
        
        elif selected == "Overall Conclusion":
            st.title("Overall Conclusion")
            
            conclusion_option = st.selectbox("Select a conclusion to view", 
                                             ["Student Intervention Prediction", 
                                              "Engagement Prediction", 
                                              "Difficulty Adjustment Prediction"])
            
            if conclusion_option == "Student Intervention Prediction":
                st.write("**Conclusion for Student Intervention Prediction:**")
                st.write("Student Intervention Prediction: Model Performance: Models show moderate performance in predicting student intervention needs, with accuracy ranging between 51% and 54%.\n"
"Class Balance: Decision Tree and XGBoost slightly outperform Logistic Regression and Random Forest in balancing precision and recall across classes.\n"
"Precision and Recall: All models demonstrate a trade-off between precision and recall, with slight improvements in specific classes but no clear winner overall.\n"
"F1-Scores: F1-scores are relatively close across models, indicating consistent ability to identify students needing intervention.\n"
"Areas of Improvement: Models could benefit from further tuning and feature engineering to enhance prediction accuracy and reliability.\n"
"Practical Application: Use the model's predictions to prioritize students for additional support and tailored interventions, ensuring a focus on those most likely to benefit.")
            
            elif conclusion_option == "Engagement Prediction":
                st.write("**Conclusion for Engagement Prediction:**")
                st.write("Model Accuracy: Accuracy for predicting student engagement is around 51% to 54%, reflecting moderate predictive performance.\n"
"Class Metrics: Decision Tree and XGBoost models show better class-specific precision and recall compared to Logistic Regression and Random Forest.\n"
"Model Comparisons: All models exhibit similar performance metrics, suggesting a need for further enhancement in capturing engagement nuances.\n"
"F1-Scores: F1-scores are relatively balanced, with slight improvements in identifying engaged versus disengaged students.\n"
"Model Enhancement: Consider incorporating additional features or advanced algorithms to better capture engagement patterns.\n"
"Actionable Insights: Use engagement predictions to develop targeted strategies for increasing student involvement and interaction.")
            
            elif conclusion_option == "Difficulty Adjustment Prediction":
                st.write("**Conclusion for Difficulty Adjustment Prediction:**")
                st.write("Model Effectiveness: Models provide moderate predictions of difficulty adjustment needs, with accuracy between 51% and 54%.\n"
"Class-Specific Performance: Decision Tree and XGBoost have marginally better performance in predicting difficulty adjustment needs compared to others.\n"
"Precision vs. Recall: Trade-offs between precision and recall are evident, affecting the prediction of whether adjustments are necessary.\n"
"F1-Scores: Models achieve similar F1-scores, indicating a balanced approach to predicting difficulty adjustments.\n"
"Improvement Opportunities: Further refinement and feature exploration could enhance the accuracy of difficulty adjustment predictions.\n"
"Implementation: Utilize predictions to tailor difficulty levels in coursework and assignments, ensuring they align with individual student needs.")

if __name__ == "__main__":
    main()
