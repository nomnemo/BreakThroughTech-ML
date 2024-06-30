# # Cross-Validation for KNN

# In this exercise, you will perform k-fold cross-validation when training a KNN classifier.
# You will use built-in cross-validation tools from scikit-learn.
# You will train the KNN model on "cell2cell," a telecom company churn prediction data set.

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ## Step 1: Build Your DataFrame and Define Your ML Problem
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)

# ## Step 2. Create Labeled Examples from the Data Set 
y = df['Churn']
X = df.drop(columns = 'Churn', axis=1)

# ## Step 3: Create Training and Test Data Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)
print(X_train.shape)
print(X_test.shape)
X_train.head()

# ## Step 4: Train a KNN Classifier and Perform k-Fold Cross-Validation
model = KNeighborsClassifier(n_neighbors=3)

from sklearn.model_selection import KFold
num_folds = 5
folds = KFold(n_splits = num_folds, random_state=None)

acc_scores = []

for train_row_index , test_row_index in folds.split(X_train): 
    # our new partition of X_train and X_val
    X_train_new  = X_train.iloc[train_row_index] 
    X_val = X_train.iloc[test_row_index]
    
    # our new partition of y_train and y_val
    y_train_new = y_train.iloc[train_row_index]
    y_val = y_train.iloc[test_row_index]
    
    model.fit(X_train_new, y_train_new)
    predictions = model.predict(X_val)
     
    iteration_accuracy = accuracy_score(predictions , y_val)
    acc_scores.append(iteration_accuracy)
     
        
for i in range(len(acc_scores)):
    print('Accuracy score for iteration {0}: {1}'.format(i+1, acc_scores[i]))

avg_scores = sum(acc_scores)/num_folds
print('\nAverage accuracy score: {}'.format(avg_scores))


# ###  Scikit-Learn's `cross_val_score` Approach
from sklearn.model_selection import cross_val_score
# YOUR CODE HERE
accuracy_scores = cross_val_score(model, X_train, y_train, cv=5)
print('Done')
print('Accuracies for the five training/test iterations on the validation sets:')
print(accuracy_scores)

acc_mean = accuracy_scores.mean()
print('The mean accuracy score across the five iterations:')
print(acc_mean)

# Find the standard deviation of the accuracy score and save to variable 'acc_std'
# YOUR CODE HERE
acc_std = accuracy_scores.std()

# Print the standard deviation of the accuracy scores using the std() method to see the degree of variance.
print('The standard deviation of the accuracy score across the five iterations:')
print(acc_std)

# You'll notice that the five resulting accuracy scores are good, 
# and the standard deviation among the scores are low, indicating that our model performs well. 
