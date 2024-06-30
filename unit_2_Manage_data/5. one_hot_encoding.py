import pandas as pd
import numpy as np
import os 

"""
1. df.select_dtypes(include=['object'])
2. df[to_encode].nunique()
3. pd.get_dummies(df['CreditRating'], prefix='CreditRating_')
"""
#### load and inspect the data
filename = os.path.join("data", "cell2cell.csv")
df = pd.read_csv(filename, header=0)
df.shape
df.head()
df.dtypes

#### 1) get the names of columns of type string -> Object
to_encode = list(df.select_dtypes(include=['object']).columns)
to_encode

## for each string type column, count the unique values
df[to_encode].nunique()
        # ServiceArea     747
        # CreditRating      7
        # PrizmCode         4
        # Occupation        8
        # Married           2
        # dtype: int64

#### 2) ServiceArea COLUMN ====> too much distinct values

## since this column has so many distinct values, get the top10 ServiceAreas
top_10_SA = list(df['ServiceArea'].value_counts().head(10).index)
top_10_SA

## For each top distinct values, create separate columns and their values(binary)
for value in top_10_SA:
    df['ServiceArea_'+ value] = np.where(df['ServiceArea']==value,1,0)
    
## Remove the original column from your DataFrame df
df.drop(columns = 'ServiceArea', inplace=True)


# #### `CreditRating` Column =====> reasonable number of distinct values
df['CreditRating']
# Use pd.get_dummies() to create a new DataFrame with the one-hot encoded values.
df_CreditRating = pd.get_dummies(df['CreditRating'], prefix='CreditRating_')
df_CreditRating
# Concatenate DataFrame df with the one-hot encoded DataFrame df_CreditRating
df = df.join(df_CreditRating)
df.drop(columns = 'CreditRating', inplace=True)
# #### `PrizmCode` Column