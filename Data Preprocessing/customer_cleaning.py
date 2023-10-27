import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 

warnings.filterwarnings('ignore')

customer= pd.read_csv("olist_customers_dataset.csv")

print ("Customers")
customer.head()
customer.info()

print ("Unique: ")
customer.nunique()

print("Null Value: ")
customer.isnull().sum()

print("Null Value Percentage: ")
(customer.isnull().sum()/(len(customer)))*100
