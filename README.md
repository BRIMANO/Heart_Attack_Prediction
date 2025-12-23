# Heart Attack Prediction

## Project Overview
This project focuses on predicting the likelihood of a patient having a heart attack based on various medical attributes. By analyzing features such as age, cholesterol levels, and heart rate, the goal is to build a machine learning model that can accurately classify patients into low-risk or high-risk categories.

The primary analysis and model training are conducted in the Jupyter Notebook `Heart Attack Prediction.ipynb`.

## Dataset
The dataset used in this project contains medical details of patients. It typically includes the following 14 variables:

* **Age**: Age of the patient.
* **Sex**: Gender of the patient (1 = Male, 0 = Female).
* **cp**: Chest Pain type (4 values).
* **trtbps**: Resting blood pressure (in mm Hg).
* **chol**: Serum cholestoral in mg/dl.
* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
* **restecg**: Resting electrocardiographic results (values 0, 1, 2).
* **thalachh**: Maximum heart rate achieved.
* **exng**: Exercise induced angina (1 = yes; 0 = no).
* **oldpeak**: ST depression induced by exercise relative to rest.
* **slp**: The slope of the peak exercise ST segment.
* **caa**: Number of major vessels (0-3) colored by flourosopy.
* **thall**: Thalassemia (0 = null, 1 = fixed defect, 2 = normal, 3 = reversable defect).
* **output**: Diagnosis of heart disease (Target Variable: 0 = Low Risk, 1 = High Risk).

## Technologies Used
* **Python 3.11**
* **Jupyter Notebook**
* **Pandas**: Data manipulation and analysis.
* **NumPy**: Numerical operations.
* **Matplotlib / Seaborn**: Data visualization and EDA.
* **Scikit-Learn**: Machine learning algorithms and metrics.

## Project Workflow
The notebook follows a standard data science pipeline:
1. **Data Loading & Inspection**: Importing the dataset and checking for null values or duplicates.
2. **Exploratory Data Analysis (EDA)**:
    * Analyzing the distribution of features (e.g., Age vs. Output).
    * Correlation heatmap to identify relationships between variables.
    * Visualizing categorical features like Chest Pain (cp) and Thalassemia (thall).
3. **Data Preprocessing**:
    * Scaling numerical features (StandardScaler/MinMaxScaler).
    * Encoding categorical variables (One-Hot Encoding).
    * Splitting the data into Training and Testing sets.
4. **Modeling**:
    * Implementing Logistic Regression classification algorithms.
5. **Evaluation**:
    * Model performance was evaluated using **Accuracy Score**, **Confusion Matrix**, and **Classification Report**.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/BRIMANO/Heart_Attack_Prediction.git](https://github.com/BRIMANO/Heart_Attack_Prediction.git)
2. Navigate to the project directory:
   ```bash
   cd Heart_Attack_Prediction
3. Install the required dependencies:
    ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heart\ Attack\ Prediction.ipynb
5. Open `Heart Attack Prediction.ipynb` and execute the cells.


## Future Improvements
* **Implementing various classification algorithms such as:**
  * **K-Nearest Neighbors (KNN)**
  * **Support Vector Machine (SVM)**
  * **Decision Tree Classifier**
  * **Random Forest Classifier**
* **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize model parameters.
* **Feature Engineering**: Create new features or use techniques like PCA for dimensionality reduction.
* **Model Deployment**: Save the best model using Pickle and deploy it using Streamlit or Flask.
* **Handling Imbalance**: If the dataset is imbalanced, apply techniques like SMOTE (Synthetic Minority Over-sampling Technique).