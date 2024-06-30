#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Model Selection 
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# ## Step 1: Build Your DataFrame and Define Your ML Problem
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)

# ## Step 2: Create Labeled Examples from the Data Set 
y = df['Churn']
X = df.drop('Churn', axis=1)

# ## Step 3: Create Training and Test Data Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)
print(X_train.shape)
print(X_test.shape)
X_train.head()


# ## Step 4: Perform Decision Tree Model Selection
hyperparams = [2**n for n in range(2,5)]
hyperparams

print('Running k-fold Cross-Validation...')

accuracy_scores = []

for md in hyperparams:
    
    # 1. Create a DecisionTreeClassifier model object
    model = DecisionTreeClassifier(max_depth=md, min_samples_leaf=1)    
    
    # 2. Perform a k-fold cross-validation for the decision tree
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # 3. Find the mean of the resulting accuracy scores 
    acc_mean = cv_scores.mean()
    
    # 4. Append the mean score to the list accuracy_scores
    accuracy_scores.append(acc_mean)
    
print('Done\n')

for s in range(len(accuracy_scores)):
    print('Accuracy score for max_depth {0}: {1}'.format(hyperparams[s], accuracy_scores[s]))

# 1. Create a DecisionTreeClassifier model object and assign it to the variable 'model'
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1)
    
# 2. Fit the model to the training data 
model.fit(X_train, y_train)

# 3. Use the predic() method to make predictions on the test data and assign the results to 
# the variable 'class_label_predictions'
class_label_predictions = model.predict(X_test)

# 4. Compute the accuracy score and assign the result to the variable 'acc_score'
acc_score = accuracy_score(y_test, class_label_predictions)

print(acc_score)


# ## Step 5: Model Selection Using a Validation Curve


from sklearn.model_selection import validation_curve

print('Running Validation Curve Implementation...')

# Create a DecisionTreeClassifier model object without supplying arguments
model = DecisionTreeClassifier()

# Create a range of hyperparameter values for 'max_depth'. Note these are the same values as those we used above
hyperparams = [2**n for n in range(2,5)]

# Call the validation_curve() function with the appropriate parameters
training_scores, validation_scores = validation_curve(model, X_train, y_train,
                                       param_name = "max_depth",
                                       param_range = hyperparams,
                                        cv = 5)

print('Done\n')

print(validation_scores)


mean_validation_scores = np.mean(validation_scores, axis = 1)
for h in range(0, len(hyperparams)):
    print('Results of cross-validation for max_depth of {0}: {1}'.format(hyperparams[h], mean_validation_scores[h]))
    


# The code cell below plots the validation curve, with the values of the parameter `max_depth` on the $x$-axis, and the accuracy scores on the validation set on the $y$-axis.

# In[18]:


sns.lineplot(x=hyperparams, y=mean_validation_scores, label = "Cross-validation scores", color = 'g')

plt.title("Validation curve for the DT classifier model")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()

# ## Step 6: Model Selection Using Grid Search Cross-Validation

hyperparams_depth = [2**n for n in range(2,5)]

# Create a range of hyperparameter values for 'min_samples_leaf'. 
hyperparams_leaf = [25*2**n for n in range(0,3)]

# Create parameter grid.
param_grid={'max_depth':hyperparams_depth, 'min_samples_leaf':hyperparams_leaf}
param_grid


# ### b. Perform Grid Search Cross-Validation
from sklearn.model_selection import GridSearchCV

print('Running Grid Search...')

# 1. Create a DecisionTreeClassifier model object without supplying arguemnts
# YOUR CODE HERE
model = DecisionTreeClassifier()
 
# 2. Run a Grid Search with 5-fold cross-validation using the model.
#   Pass all relevant parameters to GridSearchCV and assign the output to the object 'grid'
# YOUR CODE HERE
grid = GridSearchCV(model, param_grid, cv=5)


# 3. Fit the model on the training data and assign the fitted model to the 
#    variable grid_search
# YOUR CODE HERE
grid.fit(X_train, y_train)

print('Done')


# ### c. Identify the Best Hyperparameter Values

# Print best-performing hyperparameter configuration
print('Optimal hyperparameters: {0}'.format(grid_search.best_params_))

# print best accuracy score resulting from this configuration
print('Accuracy score: {0}'.format(grid_search.best_score_))
