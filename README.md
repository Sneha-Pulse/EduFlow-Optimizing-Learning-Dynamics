# EduFlow-Optimizing-Learning-Dynamics

![Ed](https://github.com/user-attachments/assets/5196a25e-d84d-49a7-a1b4-4f670bb31fa6)


## **EduFlow-Optimizing-Learning-Dynamics**

This project aims to enhance student retention and success rates in higher education institutions by leveraging predictive analytics to identify at-risk students early in their academic journey and implement targeted interventions. The project utilizes Intervention Targeting, Student Engagement Analysis, and Content Difficulty Prediction to achieve these goals.

#### Project Description

EduFlow-Optimizing-Learning-Dynamics is designed to address the challenge of identifying students who are likely to struggle or drop out and understanding the key factors driving their performance. By using predictive analytics, institutions can implement targeted support systems before students face academic failure or disengagement. This project includes features such as:

- **Student Intervention Prediction:** Analyzes factors like intervention flags, help requests, and tutor sessions to predict which students are likely to benefit from additional support or resources.
- **Engagement Prediction:** Examines engagement with supplemental resources and adaptive learning features to predict levels of student engagement and design interventions to enhance participation and motivation.

#### How to Install and Run the Project

To run the EduFlow-Optimizing-Learning-Dynamics project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/yourusername/EduFlow-Optimizing-Learning-Dynamics.git
   cd EduFlow-Optimizing-Learning-Dynamics
   ```

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```sh
   streamlit run main.py
   ```

4. **Open in Browser:**
   Open your browser and navigate to `http://localhost:8501` to access the application.

![image](https://github.com/user-attachments/assets/b636f8f0-d8cd-407b-a07d-e218e57e82ed)
   

#### How to Use the Project

1. **Upload Dataset:**
   Use the sidebar to upload your dataset (CSV file). The dataset should include columns for various student metrics and features.

2. **Navigate Through Sections:**
   - **Home:** Understand the project's domain, problem statement, and real-world impact.
   - **EDA Dataset Overview:** Explore the dataset with summary statistics, missing values, data types, correlation matrix, and Chi-Square test results.
   - **Student Intervention Prediction:** Analyze intervention flags, help requests, and tutor sessions to predict which students need additional support.
   - **Engagement Prediction:** Examine engagement with supplemental resources and adaptive learning features to predict engagement levels.
   - **Overall Conclusion:** Review the conclusions for both student intervention and engagement predictions.

3. **Model Performance:**
   - Select a model (Logistic Regression, Decision Tree, Random Forest, XGBoost) to view its performance metrics, including classification reports and ROC AUC curves.
   - Review the inference and conclusions for each model to understand its strengths and weaknesses.

#### Real-World Analysis Benefits

- **Improved Student Retention:** By identifying at-risk students early, institutions can implement targeted interventions to improve retention rates.
- **Enhanced Academic Performance:** Predictive analytics help in understanding the factors driving student performance, enabling institutions to provide tailored support and resources.
- **Resource Allocation:** Institutions can allocate resources more effectively by focusing on students who are most likely to benefit from additional support.
- **Personalized Learning Paths:** Adaptive learning features and content difficulty predictions allow for personalized learning paths, improving student engagement and success rates.
