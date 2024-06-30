
# ## Analyzing the World Happiness Data: Computing Linear Regressions Among Variables

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:.2f}'.format
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ## Step 1: Build Your DataFrame and Define Your ML Problem

dfraw = pd.read_excel('WHR2018Chapter2OnlineData.xls', sheet_name='Table2.1')
cols_to_include = ['country', 'year', 'Life Ladder', 
                   'Positive affect','Negative affect',
                   'Log GDP per capita', 'Social support',
                   'Healthy life expectancy at birth', 
                   'Freedom to make life choices', 
                   'Generosity', 'Perceptions of corruption']
renaming = {'Life Ladder': 'Happiness', 
            'Log GDP per capita': 'LogGDP', 
            'Social support': 'Support', 
            'Healthy life expectancy at birth': 'Life', 
            'Freedom to make life choices': 'Freedom', 
            'Perceptions of corruption': 'Corruption', 
            'Positive affect': 'Positive', 
            'Negative affect': 'Negative'}
df = dfraw[cols_to_include].rename(renaming, axis=1)
df1517 = df[df.year.isin(range(2015,2018))]
df1517 = df1517.dropna() # remove missing values
df1517.head()


# The World Happiness Report is generally interested in how self-reported `Happiness` (Life Ladder) 
# is dependent on the variety of different factors that they measure (`LogGDP`, `Support`, `Life`, etc.).  
# The report carries out a detailed analysis explaining how much of each country's `Happiness` can be ascribed 
# to each of the explanatory factors.  We will consider later the specific analysis carried out in the WHR, 
# but start here with a simpler analysis.

# ### Visualize Variables of Interest
sns.regplot(x='LogGDP', y='Happiness', data=df1517);


# Visually, the relationship between the two variables plotted above looks something like a line, albeit with a fair amount of jitter above and below that line. 
# 
# Linear regression is a method that estimates a relationship between two variables by fitting a line to examples relating those variables.  That is, given a set of examples relating two variables, linear regression creates a <b>model</b> of the data by assuming that the data relationship is well described by a straight line &mdash; more specifically, a straight line *is* the model of the data.  Given our assumption that a line is a good description of the data relationship, we need to identify what is the specific line that best fits our particular dataset.
# 
# Note: the features are also referred to as independent variables, and the label is also referred to as the dependent variable.
# 
# Mathematically, a line relating an independent variable $x$ and a dependent variable $y$ is characterized by two parameters: the slope and the y-intercept.  Mathematically, we might write:
# 
# $$y = \alpha + w_1 x_1$$
# 
# where $w_1$ represents the slope (or weight) and $\alpha$ represents the y-intercept.  The y-intercept is where the line crosses the y-axis (i.e., for $x = 0$), and the slope indicates how a change $\Delta x$ in the independent variable corresponds to a change $\Delta y$ in the dependent variable (the slope is given $w_1 = \Delta y / \Delta x$).
# 
# In our case, we are interested in quantifying the relationship between `Happiness` and `LogGDP`, so we are interested in a specific model of the form:
# 
# $${\rm Happiness} = \alpha + w_1 * {\rm LogGDP}$$
# 
# Linear regression attempts to find the <b>best-fit line</b> that minimizes the least-squares error, that is, the squared difference between the actual training data's label and the model predicted label given by the equation above, summed over all examples. That is, linear regression produces a specific estimate for the model parameters $w_1$ and $\alpha$ that does the best job of fitting the examples.
# 
# Visually, we can see that the weight of the general trend in the data is approximately equal to 1, because the y-axis increases by around 4 units (`Happiness` increasing from approximately 3 to 7) at the same time that the x-axis also increases by around 4 units (`LogGDP` going from 7 to 11).  It is harder to estimate the intercept of the dataset, since at `LogGDP=7`, the data are far from the y-axis at $x = 0$.  You should recognize that some estimates for the model parameters $w_1$ and $\alpha$ would do a poor job of describing the data.  If we chose the weight to be $w_1=100$, for example, then our model would be predicting a much more rapid rise than we see in the actual data.  Or if we chose a very large intercept such as $\alpha=1000$, the model predictions would lie far above the data.  But we don't need to rely on visual inspection, since we can use tools to estimate these model parameters numerically.  (This is part of a general process typically referred to as "parameter estimation" or "estimating parameters from data", but it should be recognized that parameter estimation always occurs within the context of some assumed model, such as the straight line we are using here.)
# 
# The method from scikit-learn that we will use specifically is called Ordinary Least Squares (OLS), which is the simplest of linear regression methods to estimate the model parameters that minimize the mean-squared error (MSE) between the actual data and the model predictions. The MSE is computed as:
# 
# $$ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$
# 
# where $y_i$ is the $i$th example's label and $\hat{y}_i$ is the regression model's predicted value. OLS chooses to minimize the MSE loss function. There are many possible loss functions for a regression problem and they will all give different results. The MSE is one of the simplest and most theoretically understood.
# 
# 

# ### Define the Label and Identify Features
# 
# We will create a <b>simple linear regression</b> model that finds the linear relationship between one feature and one label. We will use the `LogGDP` feature and the `Happiness` label. Our model will predict a `Happiness` value for a given `LogGDP` value. This is a regression problem.
# 

# ## Step 2: Create Labeled Examples from the Data Set 
X = df1517['LogGDP'].to_frame()
y = df1517['Happiness']

print(X)
print(y)


# ## Step 3: Create Training and Test Data Sets
# 
# Now that we have specified examples, we will need to split them into a training set, which we will use to estimate the model parameters $w_1$ and $\alpha$, and a test set, which we will use to understand the performance of our model on new data. 
# 
# 
# Run the code cell below to run the ```train_test_split()``` function with `X` and `y` as inputs, along with the parameters:
# * `random_state=42` to ensure reproducible output each time the function is called
# * `test_size=0.15`, which will randomly set aside 15% of the data to be used for testing. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# ## Step 4: Train a Linear Regression Model
# 
# Now let's carry out the regression. Let's create a ```LinearRegression``` model object, and then fit the model to the training data, which is the process by which the best-fit model parameters are estimated.
# 
# The code cell below:
# 
# 1. Creates a ```LinearRegression``` model object and assigns the result to the variable ```model```.  You might find it useful to consult the scikit-learn [documentation](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares).
# 2. Calls the ```model.fit()``` method to fit the model to the training data. 
# 3. The cell below uses the ```model.predict()``` method with the argument ```X_test``` to use the trained linear regression model to predict values for the test data. It stores the outcome in the variable ```prediction```. We will compare these values to ```y_test``` later.
# 

# In[ ]:


# Create the  LinearRegression model object 
model = LinearRegression()

# Fit the model to the training data 
model.fit(X_train, y_train)

#  Make predictions on the test data 
prediction = model.predict(X_test)


# Let's inspect the model parameters that were determined during training.
# 
# * The weights $w$ are stored in numpy array ```model.coef_```. To access a particular weight, such as $w_1$, index into the array using the notation ```model.coef_[0]```. Note that we are using index notation, so the weight of feature 1 is found in element 0 of the NumPy array, the weight of feature 2 is found in element 1 of the NumPy array, and so on.
# * The intercept $\alpha$ is stored in ```model.intercept_```. 
# 
# Run the cell below and examine the results.

# In[ ]:


# Weight_1 (weight of feature LogGDP)
print('Model Summary\n\nWeight_1 =  ', model.coef_[0], '[ weight of feature LogGDP ]')
# alpha
print('Alpha = ', model.intercept_, '[ intercept ]')


# We can translate the output above the following way: the training phase has identified that the best straight-line model fitting `Happiness` to `LogGDP` is approximately:
# 
# $${\rm Happiness} = 0.7743 * {\rm LogGDP} - 1.7507$$
# 
# In the visual estimation of the slope that was done in <b>Step 2</b>, we noted that the slope should be approximately equal to 1.  That turned out to be an over-estimate since the best-fit value is actually around 0.77, but it's not wildly off.

# ## Step 5: Evaluate the Model on the Test Set
# 
# Now that we have trained the model and made some predictions we will want to examine how well the model performed.
# 
# To evaluate our model, we will compute the RMSE (root mean square error) on the test set. RMSE is a metric used to evaluate Regression models. Root Mean Square Error (RMSE) finds the differences between the predicted values and the actual values. 
# 
# To compute the RMSE, we will use the scikit-learn ```mean_squared_error()``` function, which computes the MSE between ```y_test``` and ```prediction```. We will then take the square root of the result to obtain the RMSE. 
# 
# Finally, we will use the coefficient of determination, also known as $R^2$. $R^2$ is a measure of the proportion of variability in the prediction that the model was able to make using the input data. An $R^2$ value of 1 is perfect and 0 implies no explanatory value. We can use scikit-learn's ```r2_score()``` function to compute it. Run the code below and examine the results.

# In[ ]:


# The mean squared error
print('\nModel Performance\n\nRMSE =   %.2f'
      % np.sqrt(mean_squared_error(y_test, prediction)))
# The coefficient of determination: 1 is perfect prediction
print(' R^2 =   %.2f'
      % r2_score(y_test, prediction))


# Examining the evaluation metrics, we have an RMSE of 0.71. This means that, on average, our predictions are off by 0.71 units. Since the `Happiness` feature in our data set ranges between about 2.5 and 8, this result is not bad! To truly evaluate this we would want to compare this result to the RMSE when using another simpler model, such as using the mean value of the `Happiness` feature as our prediction for every value of ```LogGDP```. 
# 
# The $R^2$ value of 0.62 implies that 62% of the variation in the ```Happiness``` feature was explained with the model by variation in ```LogGDP```. There is some subjectivity to interpreting what value is sufficient to justify the use of the model here, but let's just say that in the social sciences, it could also be a lot worse than 56%!

# ## Step 6: Visualize the Model
# 
# We can plot the data and the fit together using Matplotlib. The code cell below:
# 
# * Uses ```plt.scatter()``` to plot ```X``` and ```y```. It sets the point size s to a reasonable value to prevent overplotting of points on top of each other.
# * Uses ```plt.plot()``` to add a blue line to the plot using the values in ```X_test``` and ```prediction```.
# * Uses ```plt.xlabel()``` and ```plt.ylabel()``` to label the axes appropriately. 
# * The plot should look similar to the ```sns.regplot()``` created earlier in Step 2. 
# 
# Execute the code cell below and inspect the results.

# In[ ]:


plt.scatter(X, y,  color='black',s=15);

plt.plot(X_test, prediction, color='blue', linewidth=3);

plt.xlabel('LogGDP');
plt.ylabel('Happiness');


# ## Step 7: Create Labeled Examples for Multiple Linear Regression
# 
# Simple linear regression finds the linear relationship between one feature and one label, and multiple regression finds the linear relationship between multiple features and one label. 
# 
# We just performed a simple linear regression and found a relationship between `LogGDP` and `Happiness`.
# 
# But we are not just interested in how `Happiness` depends on `LogGDP`, but how it relates to the full set of features collected in the WHR data: `LogGDP`, `Support`, `Life`, `Freedom`, `Generosity`, and `Corruption`.  To analyze this full set of dependencies, we can set up a multiple linear regression, which aims to fit a label $y$ to a group of features $X$, by assuming that $y$ depends on each individual feature $X_i$ separately and in a linear manner.  Instead of a single weight $w_1$ as in the simple regression problem, there will now be a separate weight for each feature, i.e.:
# 
# $${\rm Happiness} =\alpha + [w_1 * {\rm LogGDP}] + [w_2 * {\rm Support}] + [w_3 * {\rm Life}] +[w_4 * {\rm Freedom}] + [w_5 * {\rm Generosity}] + [w_6 * {\rm Corruption}]$$
# 
# The code cell below creates the labeled examples for the multiple regression problem, similar as to what we did above. It performs the following tasks:
# 
# * Assigns to the variable ```features``` the list of column names for the features of interest.
# * Assigns to the variable ```y``` the `Happiness` columns in the ```df1517``` DataFrame.
# * Assigns to the variable ```X``` the columns in ```df1517``` that are listed in `features` . **Since you are extracting multiple columns from ```df1517``` the result will be a DataFrame. You do not have to coerce it back to one like you did before using .to_frame()**
# * Prints the value of ```X``` to verify that it has been constructed correctly
# 

# In[ ]:


features = ['LogGDP', 'Support', 'Life', 'Freedom', 'Generosity', 'Corruption']

X = df1517[features]
y = df1517['Happiness']

print(X)


# ## Step 8: Create Training and Test Data Sets
# 
# 
# The code cell below calls `train_test_split()` on ```X``` and ```y```.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# ## Step 9: Train a Multiple Linear Regression Model
# 
# Similar to what we did above, now let's create an OLS model for the multiple linear regression.

# In[ ]:


# Create the  LinearRegression model object 
model2 = LinearRegression()

# Fit the model to the training data 
model2.fit(X_train, y_train)

#  Make predictions on the test data 
prediction2 = model2.predict(X_test)


# Examine the model parameters.

# In[ ]:


print('Model Summary:\n')

# Print intercept (alpha)
print('Intercept:')
print('alpha = ' , model2.intercept_)

# Print weights
print('\nWeights:')
i = 0
for w in model2.coef_:
    print('w_',i+1,'= ', w, ' [ weight of ', features[i],']')
    i += 1


# ## Step 10: Evaluate the Model on the Test Set
# 
# Run the cell below to examine the metrics.

# In[ ]:


# Print mean squared error
print('\nModel Performance\n\nRMSE =   %.2f'
      % np.sqrt(mean_squared_error(y_test, prediction2)))
# The coefficient of determination: 1 is perfect prediction
print(' R^2 =   %.2f'
      % r2_score(y_test, prediction2))


# ## Step 11: Conclusions
# 
# Examine the output in the Model Summary.  Estimates of model parameters are now provided for all of the features, as well as the overall intercept.  Note that the estimates for the intercept and the `LogGDP` weight are different than was the case in the simple regression.  That is typical, since multiple regression accounts for relationships between each independent and dependent variable once all the other data relationships are taken into account.
# 
# We can see from the summary results that the `Support` and `Freedom` variables have the largest weights, indicating that, on average, an increase in those variables corresponds to an increase in `Happiness`. `Corruption` has a negative weight implying that, on average, a decrease in that feature corresponds to an increase in `Happiness`. These results fit with our common sense which is always important to verify.
# 
# We also see that our RMSE has decreased and our $R^2$ value has increased, both good indicators that adding more features has increased the accuracy and fit of the model (although it is important to note that adding variables will always increase $R^2$ and that the magnitude of the increase may differ depending on the variable)!
# 

# ## Deep Dive: Iterative Approach - Gradient Descent

# OLS is a non-iterative linear regression. If you have a regression problem that will benefit from using the iterative approach that uses the optimization algorithm Gradient Descent, scikit-learn makes it easy for you. You can simply use the `SGDRegressor` class in place of the `LinearRegression` class. (Note that `SGDRegressor` uses a type of Gradient Descent algorithm called Stochastic Gradient Descent). 
# 
# Essentially, you would replace the line ``model = LinearRegression()`` with the line ``model = SGDRegressor(loss='squared_loss', max_iter=1000, tol=1e-3, learning_rate='constant')``, but supplying the arguments of your choosing. You'll note that `SGDRegressor` allows you to specify which loss function to use, and the max number of iterations over your training data (epochs). It will also allow you to set hyperparameters, such as the `learning rate`. Once you train your model, you can evaluate the model's performance, and run the `SGDRegressor` again with different hyperparameter arguments.
# 
# You can consult the [scikit-learn documentation on SGDRegressor](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd) for more information. 
