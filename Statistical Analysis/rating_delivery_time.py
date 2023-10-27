import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
import matplotlib.pyplot as plt 
warnings.filterwarnings('ignore')

review= pd.read_csv("olist_order_reviews_dataset.csv")
order= pd.read_csv("olist_orders_dataset.csv")

#-time based analysis(based on delivery)
#-in terms of delivery time

order.dropna(inplace=True)
review['review_comment_title'].fillna("No Title", inplace=True)
review['review_comment_message'].fillna("No Message", inplace=True)

# extract the used columns
merge_data = review.merge(order, on='order_id', how='inner')
merge_data['order_delivered_carrier_date'] = pd.to_datetime(merge_data['order_delivered_carrier_date'])
merge_data['order_delivered_customer_date'] = pd.to_datetime(merge_data['order_delivered_customer_date'])
merge_data['day_difference'] = (merge_data['order_delivered_customer_date'] - merge_data['order_delivered_carrier_date']).dt.days
rating = merge_data['review_score']

# create day categories
merge_data['day_category'] = pd.cut(merge_data['day_difference'], bins=range(0, 201, 10))

pivot_table = merge_data.pivot_table(values='review_score', index='day_category', aggfunc='mean')

# heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Average Rating'})
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Rating Score')
plt.title('Heatmap of Average Rating by Delivery Time in Olist')
plt.show()

# test the relationship
correlation = merge_data['day_difference'].corr(merge_data['review_score'])
print("Correlation between Delivery Time and Rating Score:", correlation)

# Correlation between Delivery Time and Rating Score: -0.2984135695972441
# Conclusion: not extremely strong relationship between these two variables
