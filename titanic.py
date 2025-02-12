
# advanced_data_science_project.py

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# 1. Load and Clean the Data
# --------------------------
# Load the Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Drop columns that are redundant or have too many missing values
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1)

# For debugging, print the first few rows and info
print("First 5 rows of the dataset:")
print(titanic.head(), "\n")
print("Dataset info:")
print(titanic.info(), "\n")

# --------------------------
# 2. Define Features and Target
# --------------------------
# We want to predict the 'survived' column.
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Identify numerical and categorical features
num_cols = ['age', 'sibsp', 'parch', 'fare']
cat_cols = ['pclass', 'sex', 'embarked']

# --------------------------
# 3. Build Preprocessing Pipelines
# --------------------------
# Numeric pipeline: impute missing values and scale features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# --------------------------
# 4. Create the Full Pipeline
# --------------------------
# Pipeline that first preprocesses the data and then trains a RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# --------------------------
# 5. Split Data into Training and Test Sets
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 6. Hyperparameter Tuning with GridSearchCV
# --------------------------
# Define hyperparameter grid for the RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# --------------------------
# 7. Evaluate the Model on the Test Set
# --------------------------
y_pred = grid_search.predict(X_test)
print("\nTest set classification report:")
print(classification_report(y_test, y_pred))
print("Test set confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------
# 8. Extract and Visualize Feature Importances
# --------------------------
# The classifier is embedded within a pipeline that includes a ColumnTransformer.
# To extract feature importances, we first need to recover the feature names after one-hot encoding.

def get_feature_names(column_transformer):
    """
    Retrieve the feature names from a ColumnTransformer.
    """
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        if name == 'remainder':
            continue
        # If the transformer is a pipeline, get the last step
        if hasattr(transformer, 'named_steps'):
            # If onehot encoder is in the pipeline, extract feature names from it
            if 'onehot' in transformer.named_steps:
                transformer = transformer.named_steps['onehot']
        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(cols)
            feature_names.extend(names)
        else:
            feature_names.extend(cols)
    return feature_names

# Get feature names from our preprocessor
feature_names = get_feature_names(grid_search.best_estimator_.named_steps['preprocessor'])

# Extract feature importances from the classifier
classifier = grid_search.best_estimator_.named_steps['classifier']
importances = classifier.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
