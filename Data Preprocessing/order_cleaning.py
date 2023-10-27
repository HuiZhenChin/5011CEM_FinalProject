import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


order= pd.read_csv("olist_orders_dataset.csv")

print ("Orders")
order.head()
order.info()

print ("Unique: ")
order.nunique()

print("Null Value: ")
order.isnull().sum()

print("Null Value Percentage: ")
(order.isnull().sum()/(len(order)))*100

# remove those null values row
order.dropna(inplace=True)
