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

## PROGRAM:

 NAME  :KOWSALYA M

 REGISTER NUMBER:212222230069.

## IMPORT LIBRARIES

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

```
##  READ CSV FILES
```
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
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

## BEFORE TRANSFORMATION:
## HIGHLY POSITIVE SKEW:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
## HIGHLY NEGATIVE SKEW:
```
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()
```
## MODERATE POSITIVE SKEW:
```
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```
## MODERTE NEGATIVE SKEW:
```
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()
```
## LOG TRANSFORMATION
## HIGHLY POSITIVE SKEW
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
## MODERATE POSITIVE SKEW:
```
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```

## RECIPROCAL TRANSFORMATION
## HIGHLY POSITIVE SKEW
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
## SQUARE ROOT TRANSFORMATION
## HIGHLY POSITIVE SKEW
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```

## POWER TRANSFORMATION
3# MODERATE POSITIVE SKEW
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
```
## MODERATE NEGATIVE SKEW
```
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```

## QUANTILE TRANSFORMATION
## MODERATE NEGATIVE SKEW
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
## OUTPUT:
![6 1](https://user-images.githubusercontent.com/118671457/233844081-3805c8a7-e81f-4c5b-90f5-6a662b155f95.png)
![6 2](https://user-images.githubusercontent.com/118671457/233844111-10e2c3da-66f3-4519-8c0f-e05aae928d9b.png)
![6 3](https://user-images.githubusercontent.com/118671457/233844131-2619c35d-c94c-4aca-8dd0-fded58148347.png)
![6 4](https://user-images.githubusercontent.com/118671457/233844136-a7df0bf2-0691-4438-960a-4cfe19902a96.png)
![6 5](https://user-images.githubusercontent.com/118671457/233844145-76b32fd3-de04-4138-b86a-9cd3db8048ce.png)
![6 6](https://user-images.githubusercontent.com/118671457/233844154-362e3c44-0282-47b7-b138-682f3d1cf20c.png)
![6 7](https://user-images.githubusercontent.com/118671457/233844162-d359877e-61bb-4e6a-9fb7-7e92c630cc81.png)
![6 8](https://user-images.githubusercontent.com/118671457/233844169-f6f536eb-dc1f-4b73-a5ea-90ac683ff482.png)
![6 9](https://user-images.githubusercontent.com/118671457/233844192-4c55a732-7623-42ec-986c-7df76268c4e1.png)
![6 10](https://user-images.githubusercontent.com/118671457/233844196-f83c1997-a2f4-45ac-80a6-026bf4c9b301.png)
![6 11](https://user-images.githubusercontent.com/118671457/233844209-5732dd93-4fe3-4bd2-aae2-fe0aae508b08.png)
![6 12](https://user-images.githubusercontent.com/118671457/233844215-c8fd2de3-88fe-4858-af13-d71a9f88fd20.png)
![6 13](https://user-images.githubusercontent.com/118671457/233844222-19ba6d68-314c-4e20-b332-48caa5de299f.png)
![6 14](https://user-images.githubusercontent.com/118671457/233844226-aafc2234-a96f-45ac-984c-531525f29ee9.png)
![6 15](https://user-images.githubusercontent.com/118671457/233844236-aac380e3-788d-44ec-aa51-c42456d3ce1f.png)

## RESULT:
Thus feature transformation is done for the given dataset.
