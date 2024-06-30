#!/usr/bin/env python
# coding: utf-8

# # Get Summary Statistics Using Pandas describe() Method

# In this exercise, we will continue exploring the data and answering some initial questions about the dataset using the `describe()` data summarization method from the `Pandas` package.

# In[1]:


import pandas as pd
import numpy as np
import os 


# ### Load the Dataset 

# In[2]:


filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)


# ### Glance at the Dataset
# 

# In[3]:


df.head()


# ### Get the Dimensions of the Dataset

# In[4]:


df.shape


# ## Step 1: Compute Summary Statistics Using Pandas `describe()` Method

# The code cell below uses the Pandas DataFrame `describe()` method to get the summary statistics of the `df` DataFrame. It saves the resulting table as a new DataFrame named `df_summ`.

# In[5]:


df_summ = df.describe()
df_summ


# We can see that the `fnlwgt` variable is scaled very differently from others. What does this variable represent?<br>
# It is always a good idea to consult the data description before analyzing your data. This variable represents a weight of a given data point, which is the number of units in the target population that the data point represents. A weight is assigned to each observation depending on to which community or subgroup the represented person belongs. Within each state, people with similar demographic characteristics should have similar weights. 

# Recall that Pandas `describe()` ignores all non-numerical columns. This is why your summary table contains fewer columns than the original data. To fix this, the code cell below passes the `include = 'all'` parameter to the `describe()` method, and saves the results to DataFrame `df_summ_all`.

# In[6]:


df_summ_all = df.describe(include = 'all')
df_summ_all


# We could also use `describe()` to get the statistics of only a few selected columns of interest. 
# 
# The idea is to first filter the DataFrame, and then call `describe()` on the filtered object. 
# 
# How would you get a summary table for only the *age*, *education (numerical)*, and *hours per week* data?<br>
# First, create a Python list containing relevant column names. 
# Then, use the list to retrieve a subset of the DataFrame `df` with just these columns.
# Finally, apply `describe()` to that subset.
# 
# The code cell below follows these steps and saves the result to DataFrame `df_summ_selected`.

# In[7]:


describe_vars = ['age', 'education-num', 'hours-per-week']
df_summ_selected = df[describe_vars].describe()
df_summ_selected


# Going forward, we will use the first summary table `df_summ` to answer some of the questions that can help us explore and understand our (numerical) data better.  

# ## Step 2: Data Analytics Using Summary Statistics

# Let's print our summary data again:

# In[8]:


df_summ


# ### What is the 25th percentile of feature 'age'?

# The code cell below uses column and row indices to get a particular cell of the summary table that answers this initial data exploration question. Recall that you can call a value from a row named `r1` and a column named `c1` by using `loc[]`.

# In[9]:


age_25p = df_summ.loc['25%']['age']
print(f"The 25th percentile of the feature 'age' is {age_25p}")


# ### Which feature has the most variation?

# We will need to use both  `loc[]`  and a new method: `idxmax()`. Consult the online [documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.idxmax.html) for more information. The method `idxmax()` retrieves the index (or a name) of the location where the maximum value in a series was found. We need to first get a vector of `std` values, and then pass it to `idxmax()` to identify the name of a column which has the maximum value. We must specify `idxmax(axis = 1)` to indicate that the search for the highest value must occur *column-wise*.

# In[10]:


df_summ.loc['std'].idxmax(axis=1)


# Note: Many Pandas methods can be applied to both Series and DataFrame objects. The `idxmax()` method is one such method. Therefore, this could have been done in a different order: You can apply the `idxmax()` method to the DataFrame `df_summ` to find the name of the column that contains the max value *for all of the rows*, and then select only the row (`std`) of interest:

# In[11]:


df_summ.idxmax(axis = 1)['std']


# Use the same approach as the code cell above to answer the same question for the `mean` statistic: which feature in our data has the highest mean value? Save your result to variable `column_name`. Hint: Use the same code as in the code cell above, but change the column name.

# ### Graded Cell
# 
# The cell below will be graded. Remove the line "raise NotImplementedError()" before writing your code.

# In[14]:


# YOUR CODE HERE
column_name = df_summ.loc["mean"].idxmax(axis=1)


# ### Self-Check
# 
# Run the cell below to test the correctness of your code above before submitting for grading. Do not add code or delete code in the cell.

# In[15]:


# Run this self-test cell to check your code; 
# do not add code or delete code in this cell
from jn import testColumnName

try:
    p, err = testColumnName(column_name)
    print(err)
except Exception as e:
    print("Error!\n" + str(e))
    


# ### Do any features have negative values?

# The code cell below uses the appropriate row name, `loc[]`, and the `np.any()` function to get the `True`/`False` answer to the question.

# In[16]:


np.any(df_summ.loc['min'] < 0)


# ###  Which feature has the highest range?

# In the code cell below, write code to find the feature with the highest range. Follow the steps below:
# 1. Construct a vector of *differences* using `df_summ.loc[]` to find the difference between the `max` and `min` columns. Save the result to variable `column_ranges`.
# 2. Apply the `idxmax()` method to `column_ranges` to find the column with the maximum range. Save the result to variable `column_range_name`.
# 

# ### Graded Cell
# 
# The cell below will be graded. Remove the line "raise NotImplementedError()" before writing your code.

# In[22]:


# YOUR CODE HERE
column_ranges = df_summ.loc["max"] - df_summ.loc["min"]
column_range_name= column_ranges.idxmax(axis = 1)


# ### Self-Check
# 
# Run the cell below to test the correctness of your code above before submitting for grading. Do not add code or delete code in the cell.

# In[23]:


# Run this self-test cell to check your code; 
# do not add code or delete code in this cell
from jn import testRange
try:
    p, err = testRange(df, df_summ, column_ranges, column_range_name)
    print(err)
except Exception as e:
    print("Error!\n" + str(e))
    


# In[ ]:




