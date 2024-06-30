#!/usr/bin/env python
# coding: utf-8

# # Bivariate Plotting With Seaborn

# In[1]:


import pandas as pd
import numpy as np
import os 


# ###  Load the Dataset

# In[2]:


filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)


# ### Glance at the Data

# In[3]:


df.head()


# ### Get the Dimensions of the Dataset

# In[4]:


df.shape


# ## Plot Multiple Variables Using `seaborn`

# Load `matplotlib` and `seaborn` packages

# In[5]:


import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme() # this line activates a signature aesthetic that makes seaborn plots look better


# ###  Filter the Dataset
# 
# We will work with the following subset of our data: we will keep only three numeric features, one categorical feature, and one label.

# The code cell below filters `df` into a new DataFrame `df_sub`, keeping only the following columns: features `age`, `capital-gain`, `hours-per-week`, `education`, and the label `income`.

# In[6]:


df_sub = df[['age', 'capital-gain', 'hours-per-week', 'education','income']].copy() 
# Don't be alarmed by the sudden use of .copy() here!
# Adding it is not necessary. We could remove it, but then we would get some warnings down the line.
# The code would still run correctly either way.


# ### Produce a Pairplot on the Numeric Features Using `seaborn`

# Plotting a histogram of a given column is a common way to understand the distribution of this feature in your dataset. The code below accomplishes this using the `pairplot()` function from the `seaborn` package.

# In[7]:


sns.pairplot(data=df_sub)


# The pairwise scatter plots do not make obvious any straightforward relationships between the variables. It seems that `capital-gain` is concentrated at 0, so the fact that this variable has little variability may explain why we did not see strong correlation with, say, `age`. <br>
# But can we conclude that the variables are independent by looking at the apparent lack of covariability between `age` and `hours-per-week`? We can make a new, better plot that will help establish this. In particular, we will modify the plot above to:
# 1. use two different colors based on the label value.
# 2. decrease the size of the points to de-clutter the display and better see if a 'tilt' characteristic of correlated features is emerging.
# 
# (Note: recall that the label in our data signifies whether the income for the observed individual is above $50K.)

# In[8]:


sns.pairplot(data=df_sub, hue = 'income', plot_kws={'s':3})


# This version of the plot looks much better, in that it passes the 'sanity check' by meeting some of our expectations: for example, in the top-right corner we see that zero hours per week worked is common for extremely high and extremely low ends of the age range. This makes sense!<br>
# It also makes sense that all of the 'outlier' points of very high capital gain are also points that have the '>50K' value of the label. 

# ### Produce a Bar Plot on the Categorical Feature 

# First, let's properly format our categorical feature `education` in `df_sub` by converting it to a `Pandas.Categorical` format. Do you remember how to do this? Run the cell below.

# In[9]:


cat_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', 
             '9th', '10th', '11th', '12th', 'HS-grad', 
             'Prof-school', 'Assoc-acdm', 'Assoc-voc', 
             'Some-college', 'Bachelors', 'Masters', 'Doctorate']

df_sub['education'] = pd.Categorical(df_sub['education'], cat_order)


# Your objective is now to plot a histogram of all levels of `education` on the x-axis, with the counts of occurrences being on the y-axis, with one additional detail: split every bar into two parts of different colors, depending on the value of `income`. 
# In other words, for every education level (for example, 'Bachelors'), the count bar should be part orange and part blue (these are the default colors), where the size of each part is the relative size of 'income >50K' to 'income<=50K' (among 'Bachelors').
# You will need to use `hue` and `multiple` parameters of the `seaborn.histplot()` function. 
# Inspect the code below to see how this is accomplished.

# In[10]:


fig1 = plt.figure(figsize=(13,7)) 
t1 = plt.xticks(rotation=45)


sns.histplot(data=df_sub, x='education', hue='income',  multiple='stack')


# In[11]:





# In[12]:


df_sub


# Another way to analyze the distribution between the two label classes for the data grouped by the education category is to use a `barplot()`:

# In[13]:


fig2 = plt.figure(figsize=(13,7)) 
t2 = plt.xticks(rotation=45)
sns.barplot(data = df_sub, x='education', y='income')


# Here, the y-axis represents the average class label for each educational category (that is, the average of all the 0 and 1 values encountered in a particular education group). 
# Note: the black lines represent the 95% confidence interval.
