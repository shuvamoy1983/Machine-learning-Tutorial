import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#store the url in a variable
data = pd.read_csv("C:/Users/shumondal/Downloads/Mercedez/train.csv", usecols=['X1','X2','X3','X4','X5','X6'])
for i in data.columns:
    print(i,': ', len(data[i].unique()),'lebels')

    pd.get_dummies(data,drop_first=True).shape
    data.X2.value_counts().sort_values(ascending=False).head(20)
    top10=[X for X in data.X2.value_counts().sort_values(ascending=False).head(10).index]
    top10
    for label in top10:
        data[label] = np.where(data['X2']==label,1,0)
    data[['X2']+top10].head(40)


def  one_hot_top_x(df, variable,top_x_lebels):
    for label in top10:
        data[variable+'_'+label] = np.where(data['X2']==label,1,0)
    ##read the data again
data=pd.read_csv("C:/Users/shumondal/Downloads/Mercedez/train.csv", usecols=['X1','X2','X3','X4','X5','X6'])
one_hot_top_x(data,'X2',top10)

data.head()
