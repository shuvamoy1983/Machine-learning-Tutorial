import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns

df_flights = pd.read_csv("C:/Users/shumondal/Downloads/flight/data/flights.csv")
cat_df_flights = df_flights.select_dtypes(include=['object']).copy()
## You can get the total number of missing values in the DataFrame by the following one liner code
print(cat_df_flights.isnull().values.sum())
## column-wise distribution of null values
print(cat_df_flights.isnull().sum())
cat_df_flights = cat_df_flights.fillna(cat_df_flights['tailnum'].value_counts().index[0])
## Test if any null present or not
print(cat_df_flights.isnull().values.sum())

## count of distinc categories
print(cat_df_flights['carrier'].value_counts().count())

carrier_count = cat_df_flights['carrier'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Carriers')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()