
# # Assignment 4: Optimizing Logistic Regression

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename)

feature_list = df.select_dtypes(include='float64').columns.tolist()
label = 'Churn'
X = df[feature_list]
y = df[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def train_test_LR(X_train, y_train, X_test, y_test, c=1):
    '''
    Fit a Linear Regression classifier to the training data X_train, y_train.
    Return the loss and accuracy of resulting predictions on the test set.
    Parameters:
        C = Factor that controls how much regularization is applied to the model.
    '''

    #train the model
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)

    # predict probabilities
    prob_predictions = model.predict_proba(X_test)
    class_predictions = model.predict(X_test)

    # Compute log loss
    log_loss_value = log_loss(y_test, prob_predictions)

    # Compute accuracy
    accuracy = accuracy_score(y_test, class_predictions)

    return log_loss_value, accuracy

log_loss_value, accuracy = train_test_LR(X_train, y_train, X_test, y_test, c=1)
print("Log Loss:" ,log_loss_value)
print("Accuracy: ", accuracy)

cs = [10**i for i in range(-10,10)]
cs

log_losses = []
accuracies = []
for c in cs:
    log_loss_value, accuracy = train_test_LR(X_train, y_train, X_test, y_test, c=c)
    log_losses.append(log_loss_value)
    accuracies.append(accuracy)
    print("Log Loss:" ,log_loss_value)
    print("Accuracy: ", accuracy)


# Now let's visualize the results. 
cs_log10 = np.log10(cs)

print(cs)
print(cs_log10)

# #### Plot Log Loss
sns.lineplot(x=cs_log10, y=log_losses)
plt.xlabel("log10(C)")
plt.ylabel("Log Loss")
plt.title("Log Loss vs. C")
plt.show()

# #### Plot Accuracy
sns.lineplot(x=cs_log10, y=accuracies)
plt.xlabel("log10(C)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. C")
plt.show()


# Which value of $C$ yields the best results, in terms of accuracy?

# the best value of C in terms of minimizing the log loss is when value log10(C) is larger than -2.5. 
#  In terms of the accuracy, the highest accuracy is when log10(C) is less than -2.5. 
# We should balance the trade-off between the 2 and choose a value that's log10(C) is larger than -2.5.
