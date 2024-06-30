import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

#### PART 1: DEFINE THE ML PROBLEM

# DETERMINE WHETHER AN AIRBNB HOST IS SUPER_HOST OR NOT
filename = os.path.join(os.getcwd(), "data", "airbnbData_Prepared.csv")
df = pd.read_csv(filename)
print("number of rows: ", df.shape[0])
print("number of columns: ", df.shape[1])

# inspect the label
df['host_is_superhost']

# inspect possible features 
list(df.columns)

#### PART 2: PREPARE YOUR DATA 
to_encode = df.select_dtypes(include = ["object"])
to_encode.nunique()
enc = OneHotEncoder(sparse = False) 
df_enc = pd.DataFrame(enc.fit_transform(to_encode))

# Use the method enc.get_feature_names() to resintate the original column names. 
# Call the function with the original two column names as arguments.
# Save the results to 'df_enc.columns'
df_enc.columns = enc.get_feature_names(list(to_encode.columns))
df_enc.head(10)

# drop the original columns
df.drop(columns = list(to_encode.columns), inplace = True )
df.columns

# so the label is host_is_superhost
Y = df["host_is_superhost"] 
X = df.drop(columns=['host_is_superhost'])
print("Number of examples: " + str(X.shape[0]))
print("\nNumber of Features:" + str(X.shape[1]))
print(str(list(X.columns)))

# split the data set
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.33, random_state = 123)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# fit the model and return the accuracy on the prediction
def train_test_DT(X_train, X_test, y_train, y_test, depth, leaf=1, crit='entropy'):
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, criterion=crit)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# train different decision tree
print("max_depth: ", 32, " accuracy score: ", train_test_DT(X_train, X_test, y_train, y_test, 32))
print("max_depth: ", 8, " accuracy score: ", train_test_DT(X_train, X_test, y_train, y_test, 8))

#  visualize different values of depth of the decision tree with the accuracy score
def visualize_accuracy(lst_hp, lst_as):
    """
    lst_hp -> a list of hyperparamter values
    lst_as -> a list of accuracy scores
    
    requirement : same sized 2 input lists
    
    """
    
    sns.lineplot(x=lst_hp, y=lst_as)
    plt.xlabel('Hyperparameter Values (Max Depth)')
    plt.ylabel('Accuracy Scores')
    plt.title('Accuracy Scores vs. Hyperparameter Values')
    plt.show()

# for depts:
lst_hp = [32, 8]
lst_as =[]
for hp in lst_hp: 
    lst_as.append(round(train_test_DT(X_train, X_test, y_train, y_test, hp), 4))
visualize_accuracy(lst_hp,lst_as )

lst_hp = [1,2,4,8,16,32]
lst_as =[]
for hp in lst_hp: 
    lst_as.append(round(train_test_DT(X_train, X_test, y_train, y_test, hp), 4))
visualize_accuracy(lst_hp,lst_as )

def train_test_knn(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
      

k_vals = [3, 30, 300]
lst_as = []
for k in k_vals:
    accuracy = train_test_knn(X_train, X_test, y_train, y_test, k)
    lst_as.append(accuracy)
visualize_accuracy(k_vals, lst_as)
k_range = np.arange(1, 40, step = 3) 
k_range

visualize_accuracy(k_range, [train_test_knn(X_train, X_test, y_train, y_test, k) for k in k_range])









