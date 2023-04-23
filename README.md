# Ex-06-Feature-Transformation
## AIM:

To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM:
## STEP 1:

Read the given Data
## STEP 2:

Clean the Data Set using Data Cleaning Process
## STEP 3:

Apply Feature Transformation techniques to all the features of the data set
## STEP 4:

Print the transformed features.

PROGRAM:

NAME:D.DHANUMALYA.

REGISTER NUMBER:212222230030.
Importing Libraries

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

## Reading CSV File

# READ CSV FILES
df=pd.read_csv("/content/Data_to_Transform.csv")
df

## Basic Process

## BASIC PROCESS:
```
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()
```

# BEFORE TRANSFORMATION:
# HIGHLY POSITIVE SKEW:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# HIGHLY NEGATIVE SKEW:
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

# MODERTE NEGATIVE SKEW
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

Log Transformation

# LOG TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

Reciprocal Transformation

# RECIPROCAL TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

Square Root Transformation

# SQUARE ROOT TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

Power Transformation

# POWER TRANSFORMATION
# MODERATE POSITIVE SKEW
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

Quantile Transformation

# QUANTILE TRANSFORMATION
# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
## OUTPUT:

![6 1](https://user-images.githubusercontent.com/118671457/233844081-3805c8a7-e81f-4c5b-90f5-6a662b155f95.png)


