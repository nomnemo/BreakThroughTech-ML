import pandas as pd
import numpy as np
import os 

"""
# Note: Many Pandas methods can be applied to both Series and DataFrame 
        objects. The `idxmax()` method is one such method. 
        Therefore, this could have been done in a different order: 
1. df.idxmax(axis = 1)[colname]
    df[colname].idxmax(axis=1)
2. np.any(condition) => Boolean
3. 

"""
filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)


df_summ = df.describe()
df_summ

df_summ_all = df.describe(include = 'all')
df_summ_all

#### 1) Get specific column summaries 
describe_vars = ['age', 'education-num', 'hours-per-week']
df_summ_selected = df[describe_vars].describe()
df_summ_selected


#### 2) Answer rhetorical questions

# ### What is the 25th percentile of feature 'age'?
age_25p = df_summ.loc['25%']['age']
print(f"The 25th percentile of the feature 'age' is {age_25p}")

# ### Which feature has the most variation?
df_summ.loc['std'].idxmax(axis=1)
df_summ.idxmax(axis = 1)['std']

# which feature in our data has the highest mean value? 
column_name = df_summ.loc["mean"].idxmax(axis=1)

# ### Do any features have negative values?
np.any(df_summ.loc['min'] < 0)

# ###  Which feature has the highest range?
column_ranges = df_summ.loc["max"] - df_summ.loc["min"]
column_range_name= column_ranges.idxmax(axis = 1)