import pandas as pd
import numpy as np
import os 

filename = os.path.join(os.getcwd(), "data", "censusData_missingValues.csv")
df = pd.read_csv(filename, header=0)
nan_count = np.sum(df.isnull(), axis = 0)
nan_count

"""
1. df.isnull() -> 0,1 values
2. df.values.any() -> true if anything is true 
3. 
"""
### Step 1: Identify Missing Values Using Pandas isnull() Method
nan_count = np.sum(df.isnull(), axis = 0)
nan_count

condition = nan_count != 0 # look for all columns with missing values

### Step 2: get the columns with missing values
col_names = nan_count[condition].index # get the column names
print(col_names)

nan_cols = list(col_names) # convert column names to list
print(nan_cols)

### Step 3: Choose which values to fill 
nan_col_types = df[nan_cols].dtypes
nan_col_types
# look at the unique values

print(df['workclass'].unique())
print(df['occupation'].unique())
print(df['native-country'].unique())
### Step 4: Create dummy variables for the missing values 
df['age_na'] = df['age'].isnull()
df['hours-per-week_na'] = df['hours-per-week'].isnull()
df.head()

### Step 5: Fill Using fillna
# inspect the column's missing value records. 
df.loc[df['age'].isnull()]

# look at one row that contains a missing value for age
print("Row 654:  " + str(df['age'][654]))

# compute mean for all non null age values
mean_ages=df['age'].mean()
print("mean value for all age columns: " + str(mean_ages))

# fill all missing values with the mean
df['age'].fillna(value=mean_ages, inplace=True)

# look at one of the rows that contained a missing value for age. 
# It should now contain the mean
print("Row 654:  " + str(df['age'][654]))