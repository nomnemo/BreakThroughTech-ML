#### 0) Improting Packages 
import pandas as pd
import numpy as np
import os 

#### 1) Loading Data 
filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename)

#### 2) Inspecting Data 
df.head()
nrows= np.shape(df)[0]
ncols= np.shape(df)[1]

#### 3) Random Sampling of Data 

## Random sampling  using NUMPY: => np.random.choice and => loc

"""
1. np.rancom.choice(source/list[1D structure], size[int], replace[boolean])
2. df.index -> Pandas.Index Class
    - Can be converted to Numpy
        - index_array = df.index.to_numpy()
    - To a Python list: 
        - index_list = df.index.tolist()
        - list(df.index)
3.  - df.loc[indices] -> gives rows 
    - df.loc[rowidx][colidx] -> element

"""
percentage = 0.3 # out of 1
indices = np.rancom.choice(df.index, size=percentage * nrows) 
df_subset = df.loc[indices]

## Random sampling  using Pandas: => df.sample(size[int])
df_subset = df.sample(int(percentage*nrows))

#### 4) Filtering DataFrame by Column Values 

## define conditon(s)
condition = df['workclass'] =='Private' 
df_private = df[condition]
num_rows_private = df_private.shape[0]

## get mean statistics
condition2 = df['sex_selfID']=='Female' 
df[condition2]['age'].mean()

list(df["relationship"].unique())
df[df["relationship"] == "Wife"]["age"].mean()

## combine multiple conditions
conditionA = df["workclass"]=="Local-gov"
conditionB = df["hours-per-week"] > 40
conditionC = conditionA & conditionB
df_local = df[conditionC]
final_rows = df_local.shape[0]

### EXAMPLE 2: 
percentage2 = 0.5
# obtain all rows in which the column 'native-country' contains a value
df_country_notnull = df[df['native-country'].notnull()]
# obtain the number of rows in df_country_notnull
num_rows_notnull = df_country_notnull.shape[0]
# obtain a 50% random sample of rows from df_country_notnull and save the indices of these rows
indices2 = np.random.choice(df_country_notnull.index, size=int(percentage2*num_rows_notnull), replace=False)
# using the row indices, save these row values to new DataFrame df_filtered
df_filtered = df_country_notnull.loc[indices2]

mean_age = np.mean(df_filtered["age"])