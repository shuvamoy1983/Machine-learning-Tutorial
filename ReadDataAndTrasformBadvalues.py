import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset= pd.read_csv("/Users/shuvamoymondal/Desktop/Machine_Learning_Folder/Part_1_Data_Preprocessing/Data.csv")

## get indipendent variable columms which is also called Features.Means a property of your training data.
##If you put .values, it will convert colum to array format
X = dataset.iloc[: ,:-1].values
X
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
X[:,1:3]
