import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

""" STEP 1: Matrix multiplication and inverse"""
# generate a synthetic data :  a  100𝑥3 matrix
np.random.seed(1234) 

mean_vector = [0, 0, 0]

cov_matrix = [[1, 0.25, 0.25],
               [0.25, 1, 0.25],
               [0.25, 0.25, 1]]

X = np.random.multivariate_normal(mean_vector, cov_matrix, size=100)


# compute the covariance matrix of X tp see if it differs from what we specified
cov_numpy = np.cov(X, rowvar=False)
cov_numpy

# compute  it manually 
# Compute the mean of each column
X_means = np.mean(X, axis=0)
X_centered = X - X_means

X_centered_transpose = X_centered.T
dot_prod = np.dot(X_centered_transpose, X_centered)
N = X_centered.shape[0]
cov_manual = dot_prod / (N - 1)
cov_manual

## compare the np method with manual loss function
tolerance=10
cov_numpy_round = np.round(cov_numpy, tolerance)
cov_manual_round = np.round(cov_manual, tolerance)
print(sum((cov_numpy_round == cov_manual_round).flatten()))
print(cov_numpy_round.size)
result = (sum((cov_numpy_round == cov_manual_round).flatten()) == cov_numpy_round.size)
result

# take inverse of the covaraince matrix 
cov_inv = np.linalg.inv(cov_manual)
cov_dot = np.dot(cov_manual, cov_inv)
cov_dot

""" STEP 2: Implement the Prediction Step: Predicting Probabilities """

def compute_lr_prob(X, W, alpha): # Do not remove this line of code
    '''
    X = Nxk data matrix (array)
    W = kx1 weight vector (array)
    alpha = scalar intercept (float)
    '''
    
    xw = np.dot(X,W) + alpha
    p = 1 / (1 + np.exp(-xw))
    
    return p # Do not remove this line of code

W = [1, -1, 2]
alpha = -1
p = (compute_lr_prob(X, W, alpha))
p

"""Step 3: Implement Log-Loss"""
def compute_log_loss(y, p): # Do not remove this line of code
    '''
    y = Nx1 vector of labels (array)
    p = Nx1 vector of probabilities (array)
    '''

    n = len(y) # Do not remove this line of code
    log_loss = -1/n * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    return log_loss # Do not remove this line of code

"""## Step 4: Use Log Loss to Evaluate the Model's Predictions"""
W = [1, -1, 2]
alpha = -1
y = (compute_lr_prob(X, W, alpha) > np.random.rand(X.shape[0])).astype(int)
y
p = compute_lr_prob(X, W, alpha)
log_loss_manual = compute_log_loss(y, p)
log_loss_manual
# compare 
log_loss_sklearn = log_loss(y, p)
print(log_loss_sklearn)
print(log_loss_manual == log_loss_sklearn)

