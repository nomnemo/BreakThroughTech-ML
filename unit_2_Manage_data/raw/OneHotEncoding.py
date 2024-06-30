#!/usr/bin/env python
# coding: utf-8

# # One-Hot Encoding 

# One-hot encoding is one of the most important feature transformation techniques that turns categorical values into binary representations. Machine learning algorithms operate on numerical inputs. Therefore, we have to transform categorical data into some form of numerical representation to prepare our data for modeling. One-hot encoding is an excellent candidate for this task because it is easy to understand and straightforward to implement. 
# 
# When we apply one-hot encoding to our categorical columns in our dataset, we create a new binary indicator column for every unique value in the original categorical column. 
# 
# Let's consider an example. Let's say we have a feature named `animal` that can have one of three possible values: `Dog`, `Cat` and `Dinosaur`. We would replace the column `animal` with three new columns, one for every potential value of `animal`: `Dog`, `Cat` and `Dinosaur`. Each new column would contain binary values. For example, for every row in which the original column `animal` had the value of `Dog`, the new column `Dog` would have the value of 1. For every row in which the original column `animal` did NOT have the value of `Dog`, the new column `Dog` would have the value of 0. Compare the original dataset below with the resulting dataset after one-hot encoding has been performed on the `animal` column.
# 
# <img src='onehotencoding3.png' width=600 height=600 align="left"/>

# In this demo, you will see how to use one hot encoding to transform string-valued, categorical features into binary values.

# ### Import Packages
# 
# Let's begin by loading the required packages:

# In[1]:


import pandas as pd
import numpy as np
import os 


# ### Load the Data Set

# In this demo, we will work with a dataset called "cell2cell." It is a dataset that contains information from a telecommunications company and can be used for customer churn prediction- that is, it can be used to predict whether a customer will remain with the company or not. 

# In[2]:


filename = os.path.join("data", "cell2cell.csv")
df = pd.read_csv(filename, header=0)


# #### Inspect the Data

# In[3]:


df.shape


# In[4]:


df.head()


# Our label will be the column `Churn`.

# ### Find the Columns (Features) That Contain String Values

# In[5]:


df.dtypes


# The code cell below finds all columns of type `object`, which corresponds to the string type.

# In[6]:


to_encode = list(df.select_dtypes(include=['object']).columns)
to_encode


# Below you will one-hot encode the columns using two different approaches.

# ## One-Hot Encode the Data Using NumPy

# There are five object-type columns in our DataFrame `df`. Lets inspect the possible number of values each column (feature) may have.

# In[7]:


df[to_encode].nunique()


# Notice that column `ServiceArea` has 747 potential values. This means we would have to create 747 new binary indicator columns - one column per unique value. That is too many!
# 
# Let's handle the special case of column `ServiceArea` first. Transforming this many categorical values would slow down the computation down the line. Instead, we will convert the top 10 most frequent values in column `ServiceArea`. 

# In[8]:


top_10_SA = list(df['ServiceArea'].value_counts().head(10).index)

top_10_SA


# Now that we have obtained the ten most frequent values for `ServiceArea`, let's transform DataFrame `df` to represent these values numerically.
# 
# 1. Create new columns to represent `ServiceArea`.
# 
#     * Instead of the original `ServiceArea` column, `df` must contain ten one-hot encoded columns: one column for every value in the top 10 most frequent service areas.
# 
#     * For example, there will be one column for `NYCBRO917`, one column for `HOUHOU281`, one column for `DALDAL214` and so on. We will name each column 'ServiceArea + '\_' + $<$service area value$>$'. For example, there will be a column named  `ServiceArea_NYCBRO917`.
# 
# 
# 2. Create values for each column.
# 
#     * Each column will have a value of either 0 or 1. 
# 
#     * 1 means that the row in question had that corresponding value present in the original `ServiceArea` column. 
# 
#     * For example, row 47 in DataFrame `df` originally had the value `DALDAL214` in column `ServiceArea`. After one-hot encoding, row 47 will have the value of 1 in new column `ServiceArea_DALDAL214`.
#     
# The code cell below accomplishes the task of creating ten one-hot encoding columns. 
# 

# In[9]:


for value in top_10_SA:
    
    ## Create columns and their values
    df['ServiceArea_'+ value] = np.where(df['ServiceArea']==value,1,0)
    
    
# Remove the original column from your DataFrame df
df.drop(columns = 'ServiceArea', inplace=True)


# Inspect DataFrame `df` and see the new columns and their values.

# In[10]:


df.head()


# In[11]:


df.columns


# Let's inspect column `ServiceArea_DALDAL214` in row 47. Remember, it should have a value of 1.

# In[12]:


df.loc[47]['ServiceArea_DALDAL214']


# ## One-Hot Encode the Data Using Pandas

# Now that we have successfully transformed the `ServiceArea` column, let us transform the rest of the data in the remaining columns.
# 
# We will perform the same method as above, but using a simpler approach. We will use the Pandas `pd.get_dummies()` function. This function transforms categorical values into binary ones. We often refer to a binary value that represents a categorical one as "dummy" value or variable. For more information on the `pd.get_dummies()` function, consult the online [documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html). 
# 
# Let's begin with the`Married` column. Let's inspect the values in `Married`. 

# In[13]:


df['Married']


# Note that each entry in the `Married` column contains one of two values - True or False. Therefore, we will replace the `Married` column with two new columns (one column per value).
# 
# In the code cell below, we are specifying which column to encode (`Married` column) as well as the prefix for the names of the new columns (`Married_`). Note that `pd.get_dummies()` returns a new DataFrame with the new one-hot encoded values. Run the cell below and inspect the resulting DataFrame with one-hot encoded values.

# In[14]:


df_Married = pd.get_dummies(df['Married'], prefix='Married_')
df_Married


# Since the `pd.get_dummies()` function returned a new DataFrame rather than making the changes to the original DataFrame `df`, let us add the new DataFrame `df_Married` to DataFrame `df`, and delete the original `Married` column from DataFrame `df`.
# 

# In[15]:


# Concatenate DataFrame df with the one-hot encoded DataFrame df_Married
df = df.join(df_Married)

# Remove the original 'Married' column from DataFrame df
df.drop(columns = 'Married', inplace=True)


# Let's inspect DataFrame `df` to see the changes that have been made. DataFrame `df` now contains columns `Married__True` and `Married__False` and no longer contains the `Married` column.

# In[16]:


df.columns


# Let's use the approach to one-hot encode the remain columns `CreditRating`, `PrizmCode` and `Occupation`.

# #### `CreditRating` Column

# We saw earlier that each row in the `CreditRating` column contains one of seven possible values. Therefore, we will replace the `CreditRating` column with seven new columns (one column per value).

# In[23]:


# Inspect the values in the column
df['CreditRating']


# In[21]:


# Use pd.get_dummies() to create a new DataFrame with the one-hot encoded values.
df_CreditRating = pd.get_dummies(df['CreditRating'], prefix='CreditRating_')
df_CreditRating


# In[20]:


# Concatenate DataFrame df with the one-hot encoded DataFrame df_CreditRating
df = df.join(df_CreditRating)

# Remove the original 'CreditRating' column from DataFrame df
df.drop(columns = 'CreditRating', inplace=True)


# #### `PrizmCode` Column

# Note that each row in the `PrizmCode` column contains one of four possible values. Therefore, we will replace the `PrizmCode` column with four new columns (one column per value).

# In[ ]:


# Inspect the values in the column
df['PrizmCode']


# In[ ]:


# Use pd.get_dummies() to create a new DataFrame with the one-hot encoded values.
df_PrizmCode = pd.get_dummies(df['PrizmCode'], prefix='PrizmCode_')
df_PrizmCode


# In[ ]:


# Concatenate DataFrame df with the one-hot encoded DataFrame df_PrizmCode
df = df.join(df_PrizmCode)

# Remove the original 'PrizmCode' column from DataFrame df
df.drop(columns = 'PrizmCode', inplace=True)


# #### `Occupation` Column

# Note that each row in the `Occupation` column contains one of eight possible values. Therefore, we will replace the `Occupation` column with eight new columns (one column per value).
# 

# In[ ]:


# Inspect the values in the column
df['Occupation']


# In[ ]:


# Use pd.get_dummies() to create a new DataFrame with the one-hot encoded values.
df_Occupation = pd.get_dummies(df['Occupation'], prefix='Occupation_')
df_Occupation


# In[ ]:


# Concatenate DataFrame df with the one-hot encoded DataFrame df_Occupation
df = df.join(df_Occupation)

# Remove the original 'Occupation' column from DataFrame df
df.drop(columns = 'Occupation', inplace=True)


# Let's now inspect DataFrame `df` and see how the columns have been transformed. Notice the new dimensions of `df`: we previously had 58 columns and now we have 84.

# In[ ]:


df.columns


# In[ ]:


df.shape

