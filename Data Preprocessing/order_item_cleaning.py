import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')

order_item= pd.read_csv("olist_order_items_dataset.csv")

print ("Order Item")
order_item.info()

print ("Unique: ")
order_item['price'].unique()
order_item['freight_value'].unique()

print("Null Value: ")
order_item.isnull().sum()

print("Null Value Percentage: ")
(order_item.isnull().sum()/(len(order_item)))*100

order_item['price'] = order_item['price'].round(2)
order_item['freight_value'] = order_item['freight_value'].round(2)

print(order_item.head())
