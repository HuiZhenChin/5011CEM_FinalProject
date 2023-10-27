import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')

seller= pd.read_csv("olist_sellers_dataset.csv")


print ("Seller")
seller.head()
seller.info()

print ("Unique: ")
seller.nunique()

print ("Unique: ")
for s in seller['seller_city'].unique():
   print(s)

print("Null Value: ")
seller.isnull().sum()

print("Null Value Percentage: ")
(seller.isnull().sum()/(len(seller)))*100
