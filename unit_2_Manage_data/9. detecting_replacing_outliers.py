import pandas as pd
import numpy as np
import os 
import scipy.stats as stats

filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)

"""
1. np.percentile(df, percent) => value
2. stats.mstats.winsorize(df[colname], limits = [upper, lower])
    => add a new column to DataFrame df. 
    The column will be named education-num-win and will contain
    the winsorized version of the education-num column,
    with the cutoff from the 'bottom' and the cutoff from 
    the 'top' both set at the 1% level.
3. stats.zscore(df[colname])
4. df.select_dtypes(include = numeric )=>  all numeric cols 
5. df.apply(function)
"""
### Step 1: compute n-th percentile of a given column 

## Using NumPy 
hpw_999 = np.percentile(df['hours-per-week'], 99.9)
hpw_999

### Step 2: add a column with the winsorized version of the original column 

df['education-num-win'] = stats.mstats.winsorize(df['education-num'], limits=[0.01, 0.01])
df.head(15)

### Computing Z-score 

##f or 1 value 
F_mean = 5.44
F_std = 7.7
value = 4 
value_zscore = (value-F_mean)/F_std
value_zscore

##for sample
F = [4, 6, 3, -3, 4, 5, 6, 7, 3 , 8, 1, 9, 1, 2, 2, 35, 4, 1]
value = F[0]

F_std = np.std(F)
F_mean = np.mean(F)
value_zscore = (value-F_mean)/F_std
value_zscore

## for all values of a feature vector => using NumPy
F_std = np.std(F)
F_mean = np.mean(F)
zscores = []
for value in F:
    z = (value-F_mean)/F_std
    zscores.append(z) 
zscores

## similar computation in a pythonic way 
F_std = np.std(F)
F_mean = np.mean(F)
zscores = [(value-F_mean)/F_std for value in F]
zscores

## for all values of a feature vector => Using Scipy 
zscores = stats.zscore(df['hours-per-week'])
zscores

## for all columns
df_zscores = df.select_dtypes(include=['number']).apply(stats.zscore)
df_zscores.head(10)