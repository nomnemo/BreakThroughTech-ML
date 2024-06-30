import pandas as pd
import numpy as np
import os 

filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename, header=0)
df.head()

#### 1. Summary statistics of the DataFrame's numeric columns by default.
df.describe()
# 	age	fnlwgt	education-num	capital-gain	capital-loss	hours-per-week
# count	7000.000000	7.000000e+03	7000.000000	7000.000000	7000.000000	7000.000000
# mean	38.596714	1.924335e+05	10.049857	1079.000429	84.970286	40.107143
# std	13.745594	1.063365e+05	2.580982	7011.160679	400.142351	12.323946
# min	17.000000	1.882700e+04	1.000000	0.000000	0.000000	1.000000
# 25%	28.000000	1.202478e+05	9.000000	0.000000	0.000000	40.000000
# 50%	37.000000	1.821170e+05	10.000000	0.000000	0.000000	40.000000
# 75%	47.000000	2.402370e+05	12.000000	0.000000	0.000000	45.000000
# max	90.000000	1.268339e+06	16.000000	99999.000000	4356.000000	99.000000

#### 2. Get the Data Types for all Columns using Pandas dtypes Property.
df.dtypes

# age                int64
# workclass         object
# fnlwgt             int64
# education         object
# education-num      int64
# marital-status    object
# occupation        object
# relationship      object
# race              object
# sex_selfID        object
# capital-gain       int64
# capital-loss       int64
# hours-per-week     int64
# native-country    object
# income            object
# dtype: object

####  3. Provides summary statistics for all columns, regardless of their data type
df.describe(include='all')

# 	age	workclass	fnlwgt	education	education-num	marital-status	occupation	relationship	race	sex_selfID	capital-gain	capital-loss	hours-per-week	native-country	income
# count	7000.000000	6625	7.000000e+03	7000	7000.000000	7000	6625	7000	7000	7000	7000.000000	7000.000000	7000.000000	6862	7000
# unique	NaN	7	NaN	16	NaN	7	14	6	5	2	NaN	NaN	NaN	40	2
# top	NaN	Private	NaN	HS-grad	NaN	Married-civ-spouse	Prof-specialty	Husband	White	Non-Female	NaN	NaN	NaN	United-States	<=50K
# freq	NaN	4879	NaN	2263	NaN	3277	911	2878	5990	4731	NaN	NaN	NaN	6233	5319
# mean	38.596714	NaN	1.924335e+05	NaN	10.049857	NaN	NaN	NaN	NaN	NaN	1079.000429	84.970286	40.107143	NaN	NaN
# std	13.745594	NaN	1.063365e+05	NaN	2.580982	NaN	NaN	NaN	NaN	NaN	7011.160679	400.142351	12.323946	NaN	NaN
# min	17.000000	NaN	1.882700e+04	NaN	1.000000	NaN	NaN	NaN	NaN	NaN	0.000000	0.000000	1.000000	NaN	NaN
# 25%	28.000000	NaN	1.202478e+05	NaN	9.000000	NaN	NaN	NaN	NaN	NaN	0.000000	0.000000	40.000000	NaN	NaN
# 50%	37.000000	NaN	1.821170e+05	NaN	10.000000	NaN	NaN	NaN	NaN	NaN	0.000000	0.000000	40.000000	NaN	NaN
# 75%	47.000000	NaN	2.402370e+05	NaN	12.000000	NaN	NaN	NaN	NaN	NaN	0.000000	0.000000	45.000000	NaN	NaN
# max	90.000000	NaN	1.268339e+06	NaN	16.000000	NaN	NaN	NaN	NaN	NaN	99999.000000	4356.000000	99.000000	NaN	NaN

#### 4. A More Detailed Way to Read Column Types using pd.api.types.infer_dtype()
types_dict = {}
for column in df.columns:
    types_dict[column] = pd.api.types.infer_dtype(df[column])

types_dict

# {'age': 'integer',
#  'capital-gain': 'integer',
#  'capital-loss': 'integer',
#  'education': 'string',
#  'education-num': 'integer',
#  'fnlwgt': 'integer',
#  'hours-per-week': 'integer',
#  'income': 'string',
#  'marital-status': 'string',
#  'native-country': 'string',
#  'occupation': 'string',
#  'race': 'string',
#  'relationship': 'string',
#  'sex_selfID': 'string',
#  'workclass': 'string'}