import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


product= pd.read_csv("olist_products_dataset.csv")

print ("Products")
product.head()
product.info()

print ("Unique: ")
product.nunique()

# remove those null values row
product.dropna(inplace=True)

print("Null Value: ")
product.isnull().sum()

print("Null Value Percentage: ")
(product.isnull().sum()/(len(product)))*100

