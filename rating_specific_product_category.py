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

# analyze the review score from 0 days to 200 days of delivery time
delivery_time_range = merged_data[(merged_data['day_difference'] >= 0) & (merged_data['day_difference'] <= 200)]
count_of_order = len(delivery_time_range)

# calculate how many order in each product category involved between 0 to 200 days
product_category_summary = delivery_time_range['product_category_name_english'].value_counts()
# display a table
category_summary_df = product_category_summary.reset_index()
category_summary_df.columns = ['Product Category', 'Count']

# sort the DataFrame by the count of orders in descending order
category_summary_df = category_summary_df.sort_values(by='Count', ascending=False)

# select all the categories 
top_categories = 72

# select the top N product categories to include in the graph
top_category_names = category_summary_df.iloc[:top_categories, 0].tolist()

# create a numerical mapping for product category names
category_mapping = {category: i for i, category in enumerate(top_category_names)}

filtered_data = merged_data[
    (merged_data['product_category_name_english'].isin(top_category_names)) &
    (merged_data['day_difference'] >= 0) &
    (merged_data['day_difference'] <= 200) 
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

# input field
product_name = "computers_accessories"
time_start = 20
time_end = 80

# create a dataframe based on input data
specific_category_data = filtered_data[
    (filtered_data['product_category_name_english'] == product_name) &
    (filtered_data['day_difference'] >= time_start) &
    (filtered_data['day_difference'] <= time_end)
]

# group by review score and delivery time, accumulate customer count
specific_category_customer_count = specific_category_data.groupby(['review_score', 'day_difference']).size().reset_index(name='customer_count')

# assign colors based on delivery time
norm = Normalize(vmin=specific_category_customer_count['day_difference'].min(), vmax=specific_category_customer_count['day_difference'].max())
colors = cm.GnBu(norm(specific_category_customer_count['day_difference']))

# reshape colors to match the shape of z
colors_reshaped = colors.reshape(-1, 1, 4)  # Reshape to (num_colors, 1, 4)

# repeat the color array for each bar in the plot
colors_repeated = np.repeat(colors_reshaped, len(specific_category_customer_count['review_score']), axis=1)

if len(specific_category_data) > 0:
    # size of graph
    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    dx = dy = 1
    x = specific_category_customer_count['review_score']
    y = specific_category_customer_count['day_difference']
    z = specific_category_customer_count['customer_count']
    
    # plot graph
    bars = ax.bar3d(x, y, np.zeros_like(z), dx, dy, z, shade=True, color=colors_repeated[:, 0, :])
    
    # intervals for y-axis scaling
    if time_end - time_start <= 20:
        y_interval = 1
    else:
        y_interval = 10
    
    # set axis title
    ax.set_xticks(np.arange(1, len(ratings) + 1))  # Adjusted to start from 1
    ax.set_xticklabels(ratings)
    ax.set_xlabel('Rating')
    ax.set_yticks(np.arange(time_start, time_end + 1, y_interval))
    ax.set_ylabel('Delivery Time')
    ax.set_zlabel('Customer Count')
    ax.set_title(f'3D Bar Graph for {product_name} ({time_start} to {time_end} days)', fontsize= 26)
    
    plt.show()
    
    print("\nCustomer Count for Each Delivery Time for Each Rating:")
    print(specific_category_customer_count)

elif len(specific_category_data) == 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')  
    
    num_rows = 1
    plt.title(f'3D Bar Graph for {product_name} ({time_start} to {time_end} days)', fontsize=16, fontweight='bold', x=0.5)
    ax.text(0.5, 0.5, 'No orders within these days', ha='center', va='center', fontsize=16, color='black')
    plt.show()
