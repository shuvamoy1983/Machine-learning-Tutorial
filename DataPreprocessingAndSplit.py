# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset; 
dataset= pd.read_csv("/Users/shuvamoymondal/Desktop/Machine_Learning_Folder/Part_1_Data_Preprocessing/Data.csv")

## get indipendent variable columms which is also called Features(Levels).Means a property of your training data.
##If you put .values, it will convert colum to array format
X = dataset.iloc[: ,:-1].values
##get my Dependent column 
y= dataset.iloc[:,3].values

##Next to Observe any missing data Nan, replcae it with values. For that, we need Imputer class of skitlearn. 
## We will use sklearn.preprocessing import Imputer,axis=0 means , it will look for value of columns for NaN

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy='mean',axis=0)
## fit method is used to best match the curvature of given data points compute the closest match and transform is used to
## transform your data. We can use fit and transform separately or we can use fit_transform.
##imputer.fit(X[:,1:3])
X[:,1:3]=imputer.fit_transform(X[:,1:3])

###***********************Categoriacal dataset manipulation ********************#########
##In this dataset, we have two categorial columns and 2 numerical columns.Country and purchase coulmns are categorial
##columns and Average and Salary are numerical columns. But Machine learning algo only deal with numerical value
#So we need to convert Categorial value to numeric value. So we need encoding(Lavel Encoder) for categorial value.

from sklearn.preprocessing import LabelEncoder
lebelencoder= LabelEncoder()
X[:, 0]=lebelencoder.fit_transform(X[:, 0])
#What one hot encoding does is, it takes a column which has categorical data, which has been label encoded, 
#and then splits the column into multiple columns. 
#The numbers are replaced by 1s and 0s, depending on which column has what value
from sklearn.preprocessing import OneHotEncoder
onehotencoder_X=OneHotEncoder(categorical_features= [0])
X=onehotencoder_X.fit_transform(X).toarray()

lebelencoder_y= LabelEncoder()
y= lebelencoder_y.fit_transform(y)

###### ****************Splitting the dataset into train and test set.********************* ############

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train,y_test)= train_test_split(X,y, test_size=0.2, random_state=0)

####### **************** Standard Scalar****************** ###########
## StandardScaler. StandardScaler standardizes a feature by subtracting the mean and then scaling to 
## unit variance. Unit variance means dividing all the values by the standard deviation.
## It outputs something very close to normal distribution

## Why Scale, Standardize, or Normalize - Many machine learning algorithms perform better or converge faster when
####   features are on a relatively similar scale and/or close to normally distributed. 

from sklearn.preprocessing import StandardScaler
standardscalar = StandardScaler()
X_train= standardscalar.fit_transform(X_train)
X_test= standardscalar.fit_transform(X_test)
