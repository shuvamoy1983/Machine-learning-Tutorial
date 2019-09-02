
##  How to select features and what are Benefits of performing feature selection before modeling your data?
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/shumondal/Downloads/AB_NYC_2019.csv")
data=data.fillna(0)
Numerical=data._get_numeric_data()
Categorial=data.iloc[:,:-1].select_dtypes('object').values
merge=pd.concat([Numerical, Categorial], sort=True)

imputer= Imputer(missing_values='NaN', strategy='mean',axis=0)
Categorial[:,0:6]=imputer.fit(Categorial[:,0:6])

