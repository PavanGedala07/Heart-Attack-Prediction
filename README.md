# Heart-Attack-Prediction
Heart Attack Prediction using Machine Learning Models
Project Overview: Heart Attack Prediction using Machine Learning Models
This project aims to predict the likelihood of a heart attack based on various clinical and demographic features using machine learning models. The dataset used is publicly available, containing multiple features such as age, blood pressure, cholesterol levels, and others, along with a binary output indicating whether a patient had a heart attack (1) or not (0).

Detailed Description of the Code and Project Workflow

The provided project aims to predict heart attack occurrences based on clinical data using multiple machine learning (ML) models. The dataset contains various features, such as age, cholesterol levels, and exercise-induced angina, with a binary target indicating the occurrence of a heart attack (1: heart attack, 0: no heart attack).

1. Data Loading and Exploration
- Libraries Used**: Common libraries like `numpy`, `pandas`, `matplotlib`, and `seaborn` are imported for data manipulation and visualization. Additionally, the `Pathlib` module is used to navigate the file system.
- Data Source**: The dataset (`heart.csv`) is loaded and explored using `.head()`, `.describe()`, and `.info()` methods to examine the structure, summary statistics, and data types of the features.

2. Exploratory Data Analysis (EDA)
- Missing Value and Unique Value Analysis**: Missing values are checked and counted using `df.isnull().sum()`, showing that the dataset has no missing values. Additionally, unique value counts for each column are determined to assess the categorical feature distribution.
- Visualization: 
  - Countplots are created for categorical variables (e.g., `sex`, `cp`, `fbs`) using Seaborn's `countplot` to show their distribution against the target variable (`output`).
  - Correlation Analysis**: A heatmap is generated to visualize the correlation between different features. This helps in identifying strong relationships between variables, such as `age` and `output`.

3. Data Preprocessing
- Standardization**: Numerical features like `age`, `trtbps` (resting blood pressure), `chol` (cholesterol), and others are standardized using `StandardScaler` to normalize them for models that are sensitive to feature scaling (e.g., Logistic Regression, SVM).
- Outlier Detection**: Interquartile Range (IQR) is used to detect and remove outliers from the numeric data.

4. Feature Engineering
- Dummy Variables**: Categorical features (e.g., `sex`, `cp`, `fbs`) are one-hot encoded using `pd.get_dummies()` to convert them into a format suitable for machine learning models.
  
5. Model Building and Tuning
The following machine learning models are implemented and compared:

A. Logistic Regression
- Logistic Regression is a baseline model used to predict the target variable. 
- Hyperparameter Tuning: `GridSearchCV` is used to optimize the penalty term (`l1` or `l2`). 
- Test Accuracy: Achieved 0.90 on the test set.

B. Random Forest
- Hyperparameter Tuning**: The model is tuned for several parameters, including the number of estimators (`n_estimators`), tree depth (`max_depth`), and split criteria (`min_samples_split`, `min_samples_leaf`).
- Feature Importance: Random Forest allows visualization of feature importance, showing which features contributed most to predictions.
- Test Accuracy: Achieved  0.80 .

C. Decision Tree
- A simple, interpretable tree-based model with hyperparameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`) to prevent overfitting.
- Test Accuracy: Achieved 0.60, lower due to overfitting tendencies of decision trees without an ensemble.

D. K-Nearest Neighbors (KNN)
- Hyperparameter Tuning: Tuned based on the number of neighbors (`n_neighbors`), weights, and distance metrics (Euclidean, Manhattan, Minkowski).
- Test Accuracy: Achieved 0.77.

E. Support Vector Machine (SVM)
- Hyperparameter Tuning: Tuned for various kernels (`linear`, `poly`, `rbf`, `sigmoid`), the regularization parameter (`C`), and the kernel coefficient (`gamma`).
- Test Accuracy: Achieved 0.87.
- ROC Curve: Plotted to visualize performance in distinguishing between classes (True Positive Rate vs. False Positive Rate).

F. Gradient Boosting
- Hyperparameter Tuning: Tuned for learning rate (`learning_rate`), maximum depth (`max_depth`), and the number of estimators (`n_estimators`).
- Test Accuracy**: Achieved 0.77, providing a balance between bias and variance.
  
6. ROC Curves for All Models
- ROC curves were plotted to compare the models' ability to distinguish between the positive and negative classes. Logistic Regression and SVM performed best based on AUC values.

Model Comparison Based on Test Accuracy

| Model                | Test Accuracy  |
|----------------------|----------------|
| **Logistic Regression**   | 0.90      |
| **Random Forest**         | 0.80      |
| **Decision Tree**         | 0.60      |
| **K-Nearest Neighbors**   | 0.77      |
| **Support Vector Machine**| 0.87      |
| **Gradient Boosting**     | 0.77      |

Conclusion: Best Model for the Project

Based on the test accuracy, **Logistic Regression** (accuracy: **0.90**) emerged as the best model for this project. It achieved the highest accuracy with good generalization on unseen data. Additionally, it offers the benefit of interpretability, making it suitable for understanding the impact of different clinical features on heart attack risk.

Support Vector Machine (SVM) also performed well (accuracy: **0.87**) and may be a good alternative, especially in scenarios where nonlinear relationships are important.

Logistic Regression is recommended as the best-performing model due to its simplicity, interpretability, and high accuracy.
