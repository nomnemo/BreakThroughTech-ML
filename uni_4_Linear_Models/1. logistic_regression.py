import pandas as pd
import numpy as np
import os 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

# load the file 
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)

# get col names
feature_list = df.select_dtypes(include =["float64"]).columns.to_list()

# define label and features 
X = df[feature_list]
y = df["Churn"]

print("Number of examples: " + str(X.shape[0]))
print("\nNumber of Features:" + str(X.shape[1]))
print(str(list(X.columns)))

# create training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.33, random_state = 1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

##### fit the logistic model

# 1. Create the LogisticRegression model object below and assign to variable 'model'
model  = LogisticRegression()

# 2. Fit the model to the training data below
model.fit(X_train, y_train)

# 3. Make predictions on the test data using the predict_proba() method and assign the 
# result to the variable 'probability_predictions' below
probability_predictions = model.predict_proba(X_test)

# print the first 5 probability class predictions
df_print = pd.DataFrame(probability_predictions, columns = ['Class: False', 'Class: True'])
print('Class Prediction Probabilities: \n' + df_print[0:5].to_string(index=False))
# 4. Compute the log loss on 'probability_predictions' and save the result to the variable
# 'l_loss' below
l_loss = log_loss(y_test, probability_predictions)
print('Log loss: ' + str(l_loss))
# 5. Make predictions on the test data using the predict() method and assign the result 
# to the variable 'class_label_predictions' below
class_label_predictions = model.predict(X_test)
# print the first 5 class label predictions 
print('Class labels: ' + str(class_label_predictions[0:5]))
# 6.Compute the accuracy score on 'class_label_predictions' and save the result 
# to the variable 'acc_score' below
acc_score = accuracy_score(y_test, class_label_predictions)
print('Accuracy: ' + str(acc_score))

# map probabilities to class model 
print(model.classes_)

# 
def computeAccuracy(threshold_value):
    labels=[]
    for p in probability_predictions[:,0]:
        if p >= threshold_value:
            labels.append(False)
        else:
            labels.append(True)
    
    acc_score = accuracy_score(y_test, labels)
    return acc_score

thresholds = [0.44, 0.50, 0.55, 0.67, 0.75]
for t in thresholds:
    print("Threshold value {:.2f}: Accuracy {}".format(t, str(computeAccuracy(t))))