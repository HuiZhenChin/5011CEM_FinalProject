import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')

review= pd.read_csv("olist_order_reviews_dataset.csv")

print ("Reviews")
review.head()
review.info()

print ("Unique: ")
review.nunique()

print("Null Value: ")
review.isnull().sum()

print("Null Value Percentage: ")
(review.isnull().sum()/(len(review)))*100

review['review_comment_title'].fillna("No Title", inplace=True)
review['review_comment_message'].fillna("No Message", inplace=True)

review.info()
