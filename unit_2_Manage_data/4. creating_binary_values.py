import pandas as pd
import numpy as np
import os 

filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)
df.head()
df.describe(include='all')

"""
1. df.isnull()
2. df[col] = np.where(condition, values to fill in if True, in which column )
3. pd.Categorical(which column, whether ordered or not?, the order in form of list)
"""
#### 1) Group Columns into Binary Values Using np.where()

df['workclass'].unique()
df.head(30)

## Step 1: Create Group 1: Not-self-emp

# Since there are only two values for self-employment, we can simplify our code by writing 
# NOT self employed 

# get all examples (rows) in which the workclass feature (column) is not self-employed
# Note: the code below uses the Pandas logical operator ~ for NOT
columns_not_self_employed = ~(df['workclass'] == 'Self-emp-not-inc') & ~(df['workclass'] == 'Self-emp-inc')

#leave nan (null) in the dataset for now. Get all examples (rows) in which the workclass feature (column) is not null
columns_not_null = ~(df['workclass'].isnull())  

# create condition
condition = columns_not_self_employed & columns_not_null

# Use np.where() to change all of the workclass values that fulfill the specified condition to Not-self-emp
df['workclass'] = np.where(condition, 'Not-self-emp', df['workclass'])

# Inspect the data to see the changed values
df.head(30)
df['workclass'].unique()

## Step 2: Create Group 2: Self-emp
condition = (df["workclass"] == 'Self-emp-not-inc') | (df["workclass"] == 'Self-emp-inc')
df["workclass"] = np.where(condition, "Self-emp", df["workclass"])
df['workclass'].unique()

condition1 = (df['income'] == '>50K')
df['income'] = np.where(condition1, 'True', df['income'])
condition2 = (df['income'] == '<=50K')
df['income'] = np.where(condition2, 'False', df['income'])
df.head(30)

#### 2) Deep Dive: Categorical Variables
df['education'].unique()
df['education'].dtype
# dtype('O')

# First, create a correctly ordered list of category names:
edu = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '1th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
# Then, use the pd.Categorical() method to reassign the values of this column 
# as a new type with awareness of the order:
df['education'] = pd.Categorical(df['education'], ordered=True, categories=edu)
df['education'].dtype
# CategoricalDtype(categories=['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
#                   '1th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm',
#                   'Assoc-voc', 'Some-college', 'Bachelors', 'Masters',
#                   'Doctorate'],
#                  ordered=True)
