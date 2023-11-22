import pandas as pd 
import numpy as np 
import warnings 
from matplotlib import cm
from matplotlib.colors import Normalize
from googletrans import Translator 
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

# Select the top N product categories to include in the graph
top_category_names = category_summary_df.iloc[:top_categories, 0].tolist()

# Create a numerical mapping for product category names
category_mapping = {category: i for i, category in enumerate(top_category_names)}

# Filter the data for the selected top product categories and the selected delivery time range
filtered_data = merged_data[
    (merged_data['product_category_name_english'].isin(top_category_names)) &
    (merged_data['day_difference'] >= 0) &
    (merged_data['day_difference'] <= 200) 
]

# Create a column for review count
filtered_data['review_count'] = 1

# Remove duplicate entries
filtered_data_no_duplicates = filtered_data.drop_duplicates(subset=['order_id', 'customer_id'])

# Create a pivot table to show the customer count for each rating of each product category
pivot_table_customer_count = filtered_data_no_duplicates.groupby(['product_category_name_english', 'review_score'])['review_count'].count().unstack(fill_value=0)

# Use the values from the table for plotting
customer_count = pivot_table_customer_count.values

# Get the values of product category and ratings
product_categories = [str(category) for category in pivot_table_customer_count.index]
ratings = [str(rating) for rating in pivot_table_customer_count.columns]

# Filter categories with total customer count less than 1000
filtered_categories = [category for category in top_category_names if pivot_table_customer_count.loc[category].sum() >= 1000]

# Sort the product categories in y-axis according to alphabetical order
filtered_categories = sorted(filtered_categories)

# Create meshgrid for 3D bar graph
x, y = np.meshgrid(np.arange(len(ratings)), np.arange(len(filtered_categories)))
z = customer_count[pivot_table_customer_count.index.isin(filtered_categories), :]

# Bar width
dx = dy = 0.8

# Graph size
fig = plt.figure(figsize=(32, 37))
ax = fig.add_subplot(111, projection='3d')

# Assign colors to each product category
norm = Normalize(vmin=0, vmax=len(filtered_categories))
colors = cm.Blues(norm(np.arange(len(filtered_categories))))

# Reshape colors to match the shape of z
colors_reshaped = colors.reshape(1, -1, 4)

# Repeat the color array for each bar in the plot
colors_repeated = np.repeat(colors_reshaped, len(ratings), axis=1)

# Plot the 3D bar graph
# Swap x and y in the bar3d function
bars = ax.bar3d(y.ravel(), x.ravel(), np.zeros_like(z).ravel(), dy, dx, z.ravel(), shade=True, color=colors_repeated[0])

# Update y-axis tick labels and labels
ax.set_yticks(np.arange(len(ratings)))
ax.set_yticklabels(ratings)
ax.set_ylabel('Rating')

# Update x-axis tick labels and labels
ax.set_xticks(np.arange(len(filtered_categories)))
ax.set_xticklabels(np.arange(0, 21, 1))
ax.set_xlabel('Product Category')

# set axis title
ax.set_zlabel('Customer Count')
ax.set_title('3D Bar Graph of Customer Count (> 1000) by Rating and Product Category', fontsize=28)

legend_labels_with_numbers = [f"{i}: {category}" for i, category in enumerate(filtered_categories)]
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=cm.Blues(norm(i))) for i in range(len(filtered_categories))]
ax.legend(legend_handles, legend_labels_with_numbers, loc='upper left', bbox_to_anchor=(1.05, 0.5, 0.1, 0.5), title='Product Categories')

plt.show()

# Calculate total customer count for each category
total_customer_count = pivot_table_customer_count.loc[filtered_categories].sum(axis=1)

# Display the total customer count as a pivot table
print("\nTotal Customer Count by Product Category (Pivot Table):")
print(total_customer_count.reset_index(name='Total Customer Count'))



