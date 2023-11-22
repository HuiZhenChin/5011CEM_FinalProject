import pandas as pd 
import numpy as np 
import warnings 
from matplotlib import cm
from matplotlib.colors import Normalize
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

# analyze the review score from 80 days to 160 days of delivery time
delivery_time_range = merged_data[(merged_data['day_difference'] >= 0) & (merged_data['day_difference'] <= 200)]
count_of_order = len(delivery_time_range)

# calculate how many order in each product category involved between 80 to 160 days
product_category_summary = delivery_time_range['product_category_name_english'].value_counts()
# display a table
category_summary_df = product_category_summary.reset_index()
category_summary_df.columns = ['Product Category', 'Count']

# sort the DataFrame by the count of orders in descending order
category_summary_df = category_summary_df.sort_values(by='Count', ascending=False)

# select the top N product categories to include in the graph (20 product categories only based on the table shown)
top_categories = 72

# select the top N product categories to include in the graph
top_category_names = category_summary_df.iloc[:top_categories, 0].tolist()

# input field
product_name = "bed_bath_table"
time_start = 0
time_end = 200

# filter the data for the selected top product category and the selected delivery time range
filtered_data = merged_data[
    (merged_data['product_category_name_english'].isin(top_category_names)) &
    (merged_data['day_difference'] >= time_start) &
    (merged_data['day_difference'] <= time_end) 
]

# create a column for review count
filtered_data['review_count'] = 1

# remove duplicate entries
filtered_data_no_duplicates = filtered_data.drop_duplicates(subset=['order_id', 'customer_id'])

# create a pivot table to show the customer count for each rating of each product category
pivot_table_customer_count = filtered_data_no_duplicates.groupby(['product_category_name_english', 'review_score'])['review_count'].count().unstack(fill_value=0)

# use the values from the table for plotting
customer_count = pivot_table_customer_count.values

# get the values of product category and ratings
product_categories = [str(category) for category in pivot_table_customer_count.index]
ratings = [str(rating) for rating in pivot_table_customer_count.columns]

product_data = filtered_data_no_duplicates[
    filtered_data_no_duplicates['product_category_name_english'] == product_name
]

# get the number of ratings and delivery times
ratings_product = product_data['review_score'].unique()
delivery_times_product = product_data['day_difference'].unique()

# create meshgrid for 3D bar graph
x_product, y_product = np.meshgrid(ratings_product, delivery_times_product)
z_product = product_data.groupby(['review_score', 'day_difference'])['review_count'].sum().unstack(fill_value=0).values

# set the bar width
dx_product = dy_product = 0.8

# set the graph size
fig_product = plt.figure(figsize=(12, 8))
ax_product = fig_product.add_subplot(111, projection='3d')

# assign colors to each rating
norm_product = Normalize(vmin=0, vmax=len(ratings_product))
colors_product = cm.Blues(norm_product(np.arange(len(ratings_product))))

# reshape colors to match the shape of z
colors_reshaped_product = colors_product.reshape(1, -1, 4)

# repeat the color array for each bar in the plot
colors_repeated_product = np.repeat(colors_reshaped_product, len(delivery_times_product), axis=1)

# plot the graph
bars_product = ax_product.bar3d(
    x_product.ravel(), 
    y_product.ravel(), 
    np.zeros_like(z_product).ravel(), 
    dx_product, 
    dy_product, 
    z_product.ravel(), 
    shade=True, 
    color=colors_repeated_product[0]
)

# set the axis title
ax_product.set_xlabel('Rating')
ax_product.set_ylabel('Delivery Time')
ax_product.set_zlabel('Customer Count')
ax_product.set_title("3D Bar Graph of Customer Count for " + product_name + " Product Category by Rating and Delivery Time", fontsize=14)

plt.show()
