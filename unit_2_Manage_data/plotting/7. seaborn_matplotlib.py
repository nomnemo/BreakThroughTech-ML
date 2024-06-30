import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme() 

"""
1. sns.histplot(dataDF, columnName)
2. plt.ylim() -> set the limits of the y-axis in a plot. 
3. plt.figure(figsize=(13,7)) -> creates a new figure, which is an empty canvas on which you can draw plots, (width, height)
4. plt.xticks(rotation=45) -> rotate the x-axis ticks labels
"""
filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)

#### 1) Produce Histogram of a Feature
sns.histplot(data=df, x="age")
sns.histplot(data=df, x="hours-per-week")
# Produce a Histogram of a Logarithm of the Same Feature
sns.histplot(data=np.log(df['age']))
sns.histplot(data=df, x="age", log_scale=True)

# Rescale the plot to zoom in/out 
sns.histplot(data=df, x="hours-per-week", log_scale = True)
plt.ylim(0, 600) # y-axis Limits: plt.ylim(0, 600) sets the y-axis to range from 0 to 600. As a result, the data points with y-values 700 and 900 will be outside the visible range of the plot and will not be displayed.

# ###  Produce a Bar Plot for a Categorical Feature
sns.histplot(data=df, x="education")

# Rotate the labels
fig1 = plt.figure(figsize=(13,7)) # this is to rescale an image so that it is easier to read
ax = sns.histplot(data=df, x="education")
t1 = plt.xticks(rotation=45)

# ## Deep Dive: Enforcing an Order of Categories
np.dtype(df['education'])
cat_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', 
             '9th', '10th', '11th', '12th', 'HS-grad', 
             'Prof-school', 'Assoc-acdm', 'Assoc-voc', 
             'Some-college', 'Bachelors', 'Masters', 'Doctorate']
df['education'] = pd.Categorical(df['education'], cat_order)

# replot 
fig2 = plt.figure(figsize=(13,7)) 
ax = sns.histplot(data=df, x="education")
t2 = plt.xticks(rotation=45)