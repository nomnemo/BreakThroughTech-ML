#!/usr/bin/env python
# coding: utf-8

# # Confusion Matrix Demo

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ##  Step 1: Build Your DataFrame
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)


# ## Step 2: Create Labeled Examples
y = df['Churn'] 
X = df.drop(columns = 'Churn', axis=1)


# ## Step 3: Create Training and Test Data Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)


# ## Step 4: Train a Decision Tree Classifier and Make Predictions
# Create a DecisionTreeClassifier model object
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf = 50)
    
# Fit the model to the training data 
model.fit(X_train, y_train)

# Make predictions on the test data
class_label_predictions = model.predict(X_test)


# ## Step 5: Check the Accuracy of Your Model
# Compute and print model's accuracy score
acc_score = accuracy_score(y_test, class_label_predictions)
print('Accuracy score: {0}\n'.format(acc_score))

# Display a confusion matrix
print('Confusion Matrix for the model: ')

c_m = confusion_matrix(y_test, class_label_predictions, labels=[True, False])

# Create a Pandas DataFrame out of the confusion matrix for display purposes
pd.DataFrame(
c_m,
columns=['Predicted: Customer Will Leave', 'Predicted: Customer Will Stay'],
index=['Actual: Customer Will Leave', 'Actual: Customer Will Stay']
)

