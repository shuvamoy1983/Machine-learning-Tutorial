import pandas as pd
import numpy as np

# Define the headers since the data does not have any
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?" )
## check the data type
df.dtypes

## Take dataframe for categorical val
obj_df=df.select_dtypes('object').copy()

## check null values either of this command
print(obj_df.isnull().sum())
obj_df[obj_df.isnull().any(axis=1)].T

## For the sake of simplicity, just fill in the value with the number 4 (since that is the most common value)
obj_df["num_doors"].value_counts()

## now replace the value with four
obj_df=obj_df.fillna({"num_doors": "four"})

# Approach #1 - Find and Replace
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}

#To convert the columns to numbers using replace :
obj_df.replace(cleanup_nums, inplace=True)
obj_df.head()

## One trick you can use in pandas is to convert a column to a category, then use those category values for your label encoding
obj_df["body_style"] = obj_df["body_style"].astype('category')

obj_df["body_style_cat"] = obj_df["body_style"].cat.codes

##  apply one hot encoder for drive wheels using get dummies
obj_df=pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()

import category_encoders as ce

# Get a new clean dataframe
obj_df = df.select_dtypes(include=['object']).copy()

# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=["engine_type"])
encoder.fit(obj_df, verbose=1)

# Only display the first 8 columns for brevity
obj_df=encoder.transform(obj_df).iloc[:,0:7]
