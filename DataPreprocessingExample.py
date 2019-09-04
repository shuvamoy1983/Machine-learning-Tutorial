import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#store the url in a variable
sales_data = pd.read_csv("C:/Users/shumondal/Downloads/data-science-class-master/WA_Fn-UseC_-Sales-Win-Loss.csv")

## get categorical and numnerical data if required
#sales_data_cate = sales_data.select_dtypes(include=['object']).copy()
#sales_data_Num=data._get_numeric_data()

##[col for col in df.columns if df[col].isnull().sum()>0]
#print(sales_data_Num.isnull().sum())
#sales_data_Num = sales_data_Num.fillna(sales_data_Num['reviews_per_month'].value_counts().index[0])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

## Dropping Opportunity result and opportunity number columns as we have to create target variable
cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
data = sales_data[cols]
target = sales_data['Opportunity Result']

from sklearn.feature_selection import SelectKBest, chi2
bestfeature= SelectKBest(score_func=chi2,k=10)
fit=bestfeature.fit(data,target)
dfscores=pd.DataFrame(fit.scores_)
dfcol=pd.DataFrame(data.columns)
## concat two dataframes for better visualization
featureScores=pd.concat([dfcol,dfscores],axis=1)
featureScores.columns=['Specs','Scores']
top10Col=featureScores.nlargest(10,'Scores')

from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(data,target)

featureImportant= pd.Series(model.feature_importances_,index=data.columns)
featureImportant.nlargest(10).plot('barh')
plot.show()

corrmat=sales_data.corr()
top_corr_feature=corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(sales_data[top_corr_feature].corr(),annot=True,cmap='RdYlGn')


from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

# set the background colour of the plot to white
sns.set(style="whitegrid", color_codes=True)
# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot
sns.countplot('Route To Market',data=sales_data,hue = 'Opportunity Result')
# Remove the top and down margin
sns.despine(offset=10, trim=True)
# display the plotplt.show()