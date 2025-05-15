# Loan Approval Prediction

End-to-end loan approval prediction pipeline using Logistic Regression and Decision Tree Classifier, with interactive deployment via Flask.

---

# Table of Content

- [Background](#background)
- [Purpose](#purpose)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
  - Loan_Approval.ipynb
  - Flask_App.py
  - routes.py
  - test_app.py
  - loan_model.pkl
  - requirements.txt
  - results.png
  - templates/index.html
  - .env
  - .gitignore
- [Steps](#steps)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Data Visualization](#data-visualization)
- [Conclusion](#results-and-conclusion)
- [How to Run](#how-to-run)
- [User Input](#user-input)
- [Deployment with Flask](#deployment-with-flask)
- [Testing](#testing)
- [Limitations](#limitations)
- [Recommendations](#recommendations)
- [Future Work](#future-work)
- [Author](#author)

---

## Background

Build models to predict loan approval using the Kaggle Loan Approval Prediction Dataset. We explore the dataset, perform data preprocessing, and apply a comparison between a Logistic Regression Model and a Decision Tree Classifier, along with a winner based on measured performance.

The program also allows users to input their own data and receive a loan approval prediction based on the best performed trained model.

Note: The dataset was originally sourced from Kaggle but used locally to ensure long-term accessibility and avoid internet dependency.

---

## Purpose

This project aims to help users assess their eligibility for home loans by predicting loan approvals based on financial and personal data. It also serves as a learning exercise to enhance machine learning, data preprocessing, modular software design, and deployment skills using real-world datasets.

---

## Features


#### Data Handling
- Real Kaggle dataset integration.
- Missing value imputation using medians.
- One-hot encoding of categorical variables.
- Scaled features using `StandardScaler`.

#### Model Training and Evaluation
- Logistic Regression and Decision Tree Classifier implementation.
- Model comparison using Accuracy, ROC AUC, Confusion Matrix, and Classification Report.
- Model serialization using `joblib`.

#### Deployment
- Flask web app for real-time predictions.
- Secure environment handling using `.env`.
- Modular `routes.py` to ensure clean routing.

#### Development and Testing
- Highly modular code structure for low coupling and high cohesion.
- Visual result representation using Matplotlib and Seaborn.
- Dependency management via `requirements.txt`.

---

## Dataset

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. This dataset is commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features.

Key features include:

- loan_id

- No_of_dependents: Number of Dependents of the Applicant.

- education: Education of the applicant.

- self_employed: Employment Status of the applicant.

- income_annum: Annual income of the applicant.

- loan_amount: Loan Amount.

- loan_term: Loan Term in Years.

- cibil_score: Credit Score.

- residential_assets_value.

- commercial_assets_value.

- luxury_assets_value.

- bank_asset_value.

- loan_status: Final decision of the Loan Approval.

---

# Project Structure 

```plaintext
Project/
├── Loan_Approval.ipynb
├── requirements.txt
├── loan_approval_dataset.csv
├── Flask_App.py
├── routes.py
├── test_app.py
├── loan_model.pkl
├── templates/
│   └── index.html
├── .env
└── results.png
```

---

# Architecture Overview

## Loan_Approval.ipynb

- Contains the steps for data loading, data preprocessing, model selection, model training, model evaluation and comparison.
- Contains markup code explaining each step of the python code.
- Imports a data set from kaggle.
- Perform data preprocessing.
- One-hot encoded categorical values.
- Defines features, target and scales the data for model training and evaluation.
- Trains a Logistic Regression and a Decision Tree classifier.
- Evaluates the models' results and compares them to come up with the best model.
- Saves the best model into a 'loan_model.pkl' for Flask Deployment.

## requirements.txt

Contains the necessary libraries for correct usage.
This project uses the following Python libraries:

- pandas: Used for data preprocessing.
- scikit-learn: Allows to implement the Decision Tree Classifier and Logistic Regression Models.
- matplotlib: Permits output data visualization.
- seaborn: Allows both models' performance to be seen back-to-back.
- joblib: Encapsulates the best performing model after output analysis.
- flask: Permits trained model deployment to allow users to access the data predictions.

Note: To install the required libraries, you can run:

```bash
pip install -r requirements.txt
```

## loan_approval_dataset.csv

- Contains the dataset used for model training.
- It's purpose is to determine if a loan will be approved or denied.
- Downloaded from Kaggle to ensure long-term accessibility and avoid internet dependency.

## Flask_App.py

- Loads the code for deploying the Decision Tree Classifier Model through loan_mode.pkl file.
- Defines the necessary features for model prediction.
- Allows users to access the trained model and receive predictions based on their inputs regarding loan approvals.

## routes.py

- Contains the routes for the Flask App.
- Obtains the data from the form created in the index.html file and converts those form values into numerical values through a 'predict' function.
- Returns the output of user inputs.
- Divides into different functions to achieve high modularity and improve code corrections or update.
- Manages URL endpoints and binds user input to prediction logic, separating web routes from core Flask logic for modularity.

## test_app.py

- Imports pytest for testing automatization.
- Creates a fake pytest client, which is going to test the functions related to the UI code.
- Tests the home page by verifying that the status code is 200.
- Tests the prediction endpoint by simulating a form submission and verifying that a valid prediction response is returned.

## loan_model.pkl

- Encapsulates the trained Decision Tree Classifier Model.
- It contains the DTC model as it outperformed the Logistic Regression Model.
- Encapsulation was performed through the 'joblib' library.

## templates/index.html

- Contains an HTML file to create a page where users can directly enter the necessary values for a model prediction to take place.
- Directly interacts with the Flask_App.py file.

## .env

- File template if environment variables are needed later.

## .gitignore

- File containing data or documents that should be ignored when uploading to github.
- Contains .env file to ensure data security.

## results.png

- An image representing the results of model evaluation.

---

## Steps

1. Data Loading: Load the Loan Approval dataset.

2. Data Exploration: Inspect the first few rows, summary statistics, and check for missing values.

3. Data Preprocessing:
- Handle missing values by filling them with the median.
- One-hot encode the categorical features.
- In this case, 'education' and 'self_employed' features are one-hot encoded using pandas' `get_dummies()`. The 'loan_status' column (Approved/Rejected) was label-encoded as the binary target variable (1 for Approved, 0 for Rejected).

4. Feature Engineering:
- Define the features (X) and the target (y).
- Scale the features using StandardScaler.

5. Model Training:
- Split the data into training and testing sets (80% train, 20% test).
- A random_state feature was added at the moment of data splitting (random_state=42).
- Train a Logistic Regression model.
- Train a Decision Tree Classifier model.
- StratifiedShuffleSplit or cross_val_score is recommended in future work.

6. Model Evaluation:
- Evaluate the model using Accuracy Score, Confusion Matrix, Roc Auc Score, and Classification Report.

Notes: 
- Evaluation metrics like Accuracy and ROC AUC help us understand how well the models predict loan approval. 
- The ROC AUC is particularly useful in binary classification problems, as it evaluates the model's performance across all thresholds.
- The Classification Report provides precision, recall, F1-score, and support for each class — useful for evaluating how well the model handles imbalanced or binary classification.


- Print the results.
- Plot the Confusion Matrix of both models with Seaborn.

---

## Code Explanation

- Data Exploration: We explore the dataset to understand its structure and clean it for further analysis.
- Preprocessing: Missing values are handled, and categorical features are one-hot encoded.
- Modeling: A Logistic Regression model and a Decision Tree Classifier model are used to make predictions.
- Evaluation: We evaluate the model's performance with key metrics and visualize the results.

---

## Results

The models were trained and evaluated with the following results:

Logistic Regression Model
- Accuracy: 0.9074941451990632
- ROC AUC: 0.9674094151882099
- Classification Report results

```
          precision    recall  f1-score   support

       0       0.92      0.93      0.93        536
       1       0.88      0.86      0.87        318

accuracy                           0.91       854
macro avg      0.90      0.90      0.90       854
weighted avg   0.91      0.91      0.91       854
```

Decision Tree Classifier Model
- Accuracy: 0.9754098360655737
- ROC AUC: 0.9733760443067679
- Classification Report results

```
          precision    recall  f1-score   support

       0       0.98      0.98      0.98        536
       1       0.97      0.97      0.97        318

accuracy                           0.98       854
macro avg      0.97      0.97      0.97       854
weighted avg   0.98      0.98      0.98       854
```

Based on the above metrics, the Decision Tree Classifier outperforms Logistic Regression and is used as the production model for deployment.
However, there are more reasons why the DTC model is often a better choice:
- DTC handles categorical splits and non-linear patterns better.
- DTC does not require data scaling.
- Interpretability you can visualize the decision tree.

---

## Data Visualization

The image below shows a side-by-side comparison of the Confusion Matrices for the Logistic Regression and Decision Tree models.

![Confusion Matrix comparing the performance of the Logistic Regression Models and the Decision Tree Classifier.](results.png)

---

## Results and Conclusion

### Model Choice: Decision Tree Classifier

Why the Decision Tree?

- It achieved better evaluation metrics (likely higher accuracy and ROC AUC).
- Handles non-linear relationships and categorical variables more effectively.
- No need for feature scaling (StandardScaler was applied before model training; while it benefits models like Logistic Regression, it has no impact on tree-based models.).

General Strengths of Decision Trees in This Task:
- Can model complex patterns in data (e.g., combinations of income, credit history, etc.).
- Robust to irrelevant features if properly pruned.

---

This project aimed to predict a binary target variable (`loan_status`), representing loan approval (1) or rejection (0). We discovered the Decision Tree Classifier has a better performance than a Logistic Regression model.
Future work can explore more advanced models and techniques to improve prediction accuracy.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook Loan_Approval.ipynb
```

---

## User Input

In the final section of the code, users can manually enter the features that describe their financial situation (e.g., income, credit score, assets). The program then outputs a prediction using the **Decision Tree Classifier**, which was selected due to its superior accuracy.

This interactive feature allows users to explore potential loan approval outcomes in real time, based on their individual data.

---

## Deployment with Flask

### Specific Flask Project Structure
loan_approval_app/
├── Flask_App.py
├── routes.py
├── test_app.py
├── loan_model.pkl
├── templates/
│   └── index.html

### How to Run

- Ensure the 'loan_model.pkl' file is downloaded or created by using the 'Loan_Approval.ipynb' file.
- Run the 'Flask_App.py' file directly, as it contains 'if __name_ _ == '__main_ _' function.
- Access the 'index.html' file or copy the expected localhost URL and paste it in your preferred browser.
- Expected localhost URL: http://127.0.0.1:5000/
- Enter the required data into the form and click 'Predict' to receive an approval prediction using the trained model.

NOTE: The model will not predict or run until all the required information is proportioned. This was done through the 'index.html' code.

---

## Testing

Automated testing is handled using `pytest`.

### Tests Include:
- **Home Page Test**: Verifies that the root URL (`'/'`) loads successfully with status code 200.
- **Prediction Form Test**: Mocks a POST request to the `/predict` endpoint with sample data to ensure the model responds and renders a result.

### Run tests using:
```bash
python pytest.py
```

---

## Limitations

PLEASE HAVE IN MIND:
While the Decision Tree Classifier achieved the best results in this project, it is prone to overfitting. 

---

## Recommendations

In case this project is directly used by or a source of inspiration to another user, the following practices are highly encouraged:
- Proper hyperparameter tuning (e.g., max depth, min samples split) is recommended to ensure generalization.
- Implement security through input validation or sanitization.
- Validate and/or sanitize user inputs in the Flask Deployment.
- Perform unit testing (recommended through pytest) or manual testing to measure model performance accurately and validate the prediction logic.

---

## Future Work

- Incorporate more advanced models such as Random Forests or Gradient Boosting.
- Perform hyperparameter tuning to further optimize model performance and avoid overfitting.
- Implement cross-validation for more robust evaluation.
- Add histograms of feature distributions, pairplots, or SHAP/TreeExplainer plots to give deeper insight into model behavior.
- Visualizing which features contributed most to the Decision Tree’s decisions (e.g., model.feature_importances_) to increase transparency.

---

## Author

Dennis Alejandro Guerra Calix -- AGCalixto 
