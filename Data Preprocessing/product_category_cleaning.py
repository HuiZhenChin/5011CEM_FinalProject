import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')

product_category= pd.read_csv("product_category_name_translation.csv")

print ("Product Category")
product_category.head()
product_category.info()

print ("Unique: ")
product_category.nunique()

print("Null Value: ")
product_category.isnull().sum()

print("Null Value Percentage: ")
(product_category.isnull().sum()/(len(product_category)))*100

