import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
import matplotlib.pyplot as plt 

warnings.filterwarnings('ignore')

# read the csv file
review= pd.read_csv("olist_order_reviews_dataset.csv")
order= pd.read_csv("olist_orders_dataset.csv")
order_item= pd.read_csv("olist_order_items_dataset.csv")
product= pd.read_csv("olist_products_dataset.csv")
name= pd.read_csv("product_category_name_translation.csv")

# time based analysis(based on delivery)
# in terms of delivery time

order.dropna(inplace=True)
product.dropna(inplace=True)
review['review_comment_title'].fillna("No Title", inplace=True)
review['review_comment_message'].fillna("No Message", inplace=True)

# extract the used columns
# merge order with review using order id
merge_data = review.merge(order, on='order_id', how='inner')
# merge with order item
merge_data = merge_data.merge(order_item, on='order_id', how='inner')
# get seller deliver order date  
merge_data['order_delivered_carrier_date'] = pd.to_datetime(merge_data['order_delivered_carrier_date'])
# get customer receive order date
merge_data['order_delivered_customer_date'] = pd.to_datetime(merge_data['order_delivered_customer_date'])
# get the date difference
merge_data['day_difference'] = (merge_data['order_delivered_customer_date'] - merge_data['order_delivered_carrier_date']).dt.days
# add one new column in merge_data with rating
rating = merge_data['review_score']

# merge with the product DataFrame with product id
merged_data = merge_data.merge(product, on='product_id', how='inner')

# create day categories based on delivery time/ length, set range with 10 days 
merge_data['day_category'] = pd.cut(merge_data['day_difference'], bins=range(0, 201, 10))

pivot_table = merge_data.pivot_table(values='review_score', index='day_category', aggfunc='mean')

product_category = merged_data.groupby('product_category_name')['review_score'].mean().sort_values(ascending=False).reset_index()
# get translated product category name
merged_data_with_translation = product_category.merge(name, on='product_category_name', how='left')

merged_data = merged_data.merge(name, on="product_category_name", how="left")

# plot a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Average Rating'})
plt.xlabel('Rating Score')
plt.ylabel('Delivery Time (Days)')
plt.title('Heatmap of Average Rating by Delivery Time in Olist')
plt.show()

# test the relationship
correlation = merge_data['day_difference'].corr(merge_data['review_score'])
print("Correlation between Delivery Time and Rating Score:", correlation)

# Correlation between Delivery Time and Rating Score: -0.26726377054097505
# Conclusion: not extremely strong relationship between these two variables

