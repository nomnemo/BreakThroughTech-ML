import pandas as pd
import numpy as np
import os 

#### Data Inspection
filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)
df.head()
df.shape

#### Random Sampling 
percentage = 0.3
num_rows = df.shape[0]
indices = np.random.choice(df.index, size = int(num_rows*percentage), replace = False)
df_subset = df.loc[indices]

#### Verifying Imbalances
"""
Functions: 
1. pd.unique(1D-structure)
2. np.sum
3. df.value_counts() -> counts unique values
4. df.groupby(listOfColumns[list])
5. df.size()
6. df.drop(labels=df_subset.index, axis=0, inplace=False)
7. df_subset.append(rows)
"""
## 1. Listing unique values of a column using Pandas unique() Method.
unique_ssID = pd.unique(df.sex_selfID)

## 2. Calculating the Proportion of Each Class
percent_female = np.sum(df_subset['sex_selfID']=='Female')/df_subset['sex_selfID'].shape[0]
percent_female

counts = df_subset['sex_selfID'].value_counts()
counts

num_examples = df_subset["race"].value_counts()
num_examples

## 3. Detecting group imbalance with respect to the label using Pandas groupby() Method.
df_subset.groupby(['sex_selfID', 'income']).size()
        # sex_selfID  income
        # Female      <=50K     611
        #             >50K       69
        # Non-Female  <=50K     991
        #             >50K      429
        # dtype: int64


## 4. Addressing imbalance: upsampling the underrepresented group
low_income_nonfemale, high_income_nonfemale = df_subset.groupby(['sex_selfID', 'income']).size()['Non-Female']
class_balance_nonfemale = high_income_nonfemale / low_income_nonfemale

low_income_female, high_income_female = df_subset.groupby(['sex_selfID', 'income']).size()['Female']

add_sample_size = int(class_balance_nonfemale*low_income_female - high_income_female)
add_sample_size # we need this many more points in (Female)&(>50K) group for balance

# Subset the original data: exclude entries that are already in our sample:
df_never_sampled = df.drop(labels=df_subset.index, axis=0, inplace=False)

# Filter that subset to include only the type of examples that we want to upsample: Females, higher income
condition = (df_never_sampled['income']=='>50K') & (df_never_sampled['sex_selfID']=='Female')
df_never_sampled_target = df_never_sampled[condition]

# Sample from the resulting set
size=min(add_sample_size, df_never_sampled_target.shape[0])
indices = np.random.choice(df_never_sampled_target.index, size=size, replace=False)

# Append the selected examples to our original sample
rows = df.loc[indices]
df_balanced_subset = df_subset.append(rows)
df_balanced_subset.head()

# Result
df_balanced_subset.groupby(['sex_selfID', 'income']).size()
        # sex_selfID  income
        # Female      <=50K      570
        #             >50K       239
        # Non-Female  <=50K     1025
        #             >50K       431
        # dtype: int64

