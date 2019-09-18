import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer

sns.set(style='white', context='notebook', palette='deep')

#store the url in a variable
data = pd.read_csv("C:/Users/shumondal/Downloads/adult.csv")
data.isnull().sum()
data= data.fillna(np.nan)

data['income'] = [0 if x== '<=50K' else 1 for x in data['income'] ]
X=data.drop('income',1)
y=data['income']

print(pd.get_dummies(X['education']))

for col_names in X.columns:
    if X[col_names].dtypes =='object':
        unique_cat= len(X[col_names].unique())
        print("Feature '{col_names} has '{unique_cat}' unique categories".format(col_names=col_names,unique_cat=unique_cat))

## check unique categories
print(X['native.country'].value_counts().sort_values(ascending=False))

X['native.country']=['United-States' if i=='United-States' else 'Other' for i in X['native.country'] ]

def dummy_df(df, list_col):
    for x in list_col:
        dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
        df=df.drop(x, 1)
        df=pd.concat([df,dummies],axis=1)
    return df

categorical_features = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
X=dummy_df(X,categorical_features)

X.isnull().sum().sort_values(ascending=False).head()
imp=Imputer(missing_values='NaN',strategy='median',axis=0)
imp.fit(X)
X=pd.DataFrame(data=imp.transform(X),columns=X.columns)

## now check again
X.isnull().sum().sort_values(ascending=False).head()

df= pd.concat([X,y.to_frame()])
imp.fit(df)
df=pd.DataFrame(data=imp.transform(df),columns=df.columns)



plt.figure(figsize=(12,10))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .10})

df.drop("Target", axis=1).apply(lambda x: x.corr(df.Target))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=101)
import sklearn.feature_selection
select=sklearn.feature_selection.SelectKBest(k=20)
selected_feature=select.fit(X_train,Y_train)
indices_selected= selected_feature.get_support(indices=True)
col_selected=[X.columns[i] for i in indices_selected]
X_train_Selected=X_train[col_selected]
X_test_Selected=X_test[col_selected]

model=LogisticRegression()
model.fit(X_train_Selected,Y_train)
y_hat = [x[1] for x in model.predict_proba(X_test_Selected)]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_test,y_hat))


