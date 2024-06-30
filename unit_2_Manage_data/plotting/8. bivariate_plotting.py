import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()
filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)

"""
1. sns.pairplot() -> 
2. sns.barplot()
"""

### Get copy subset of the dataFrame
df_sub = df[['age', 'capital-gain', 'hours-per-week', 'education','income']].copy() 

# ### Produce a Pairplot on the Numeric Features Using `seaborn`
sns.pairplot(data=df_sub)

#### Produce
sns.pairplot(data=df_sub, hue = 'income', plot_kws={'s':3})

# ### Produce a Bar Plot on the Categorical Feature 
cat_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', 
             '9th', '10th', '11th', '12th', 'HS-grad', 
             'Prof-school', 'Assoc-acdm', 'Assoc-voc', 
             'Some-college', 'Bachelors', 'Masters', 'Doctorate']

df_sub['education'] = pd.Categorical(df_sub['education'], cat_order)
fig1 = plt.figure(figsize=(13,7)) 
t1 = plt.xticks(rotation=45)
sns.histplot(data=df_sub, x='education', hue='income',  multiple='stack')
df_sub['income'] = (df_sub['income'] =='>50K').astype(int)

fig1 = plt.figure(figsize=(13,7)) 
t1 = plt.xticks(rotation=45)

# Another way to analyze the distribution between the two label classes for the data grouped by the education category is to use a `barplot()`:
fig2 = plt.figure(figsize=(13,7)) 
t2 = plt.xticks(rotation=45)
sns.barplot(data = df_sub, x='education', y='income')
