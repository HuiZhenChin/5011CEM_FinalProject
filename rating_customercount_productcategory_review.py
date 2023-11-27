import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
import matplotlib.pyplot as plt 
from collections import Counter
from googletrans import Translator
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
delivery_time_range = merged_data[(merged_data['day_difference'] >= 80) & (merged_data['day_difference'] <= 160)]
count_of_order = len(delivery_time_range)

# calculate how many order in each product category involved between 80 to 160 days
product_category_summary = delivery_time_range['product_category_name_english'].value_counts()
# display a table
print("Product Category Summary for Orders with 80-160 Days Delivery Time:")
print(product_category_summary)
print(f"Number of Orders within the 80-150 days delivery time range: {count_of_order}\n")
print ("Total Count: ", count_of_order)


category_summary_df = product_category_summary.reset_index()
category_summary_df.columns = ['Product Category', 'Count']

# sort the DataFrame by the count of orders in descending order
category_summary_df = category_summary_df.sort_values(by='Count', ascending=False)

# select the top N product categories to include in the graph (20 product categories only based on the table shown)
top_categories = 20
top_category_names = category_summary_df.iloc[:top_categories, 0].tolist()

# filter the data for the selected top product categories and the selected delivery time range
filtered_data = merged_data[
    (merged_data['product_category_name_english'].isin(top_category_names)) &
    (merged_data['day_difference'] >= 80) &
    (merged_data['day_difference'] <= 160) &
    (merged_data['review_score'] >= 1) &  # filter for range of rating score
    (merged_data['review_score'] <= 5)  
]

# set the bar graph size
plt.rcParams["figure.figsize"] = [12.00, 8.00]
plt.rcParams["figure.autolayout"] = True

# plot in 3D bar graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# define the range of ratings from 1 to 5 and in integer format
x_values = np.arange(1, 6, 1)  

# create arrays to store the customer count for each rating score and product category
customer_count_by_rating = np.zeros((len(x_values), len(top_category_names)))

# populate the customer count arrays
# get the number of customer in each rating score for different product category
for i, rating in enumerate(x_values):
    for j, category in enumerate(top_category_names):
        rating_category_data = filtered_data[
            (filtered_data['review_score'] == rating) &
            (filtered_data['product_category_name_english'] == category)
        ]
        customer_count = len(rating_category_data)
        customer_count_by_rating[i, j] = customer_count

# create a numerical mapping for product category names
category_mapping = {category: i for i, category in enumerate(top_category_names)}

combined_reviews = []
combined_ratings = []

for rating_value in [1, 2, 3, 4, 5]:
    rating_reviews = filtered_data[filtered_data['review_score'] == rating_value]['review_comment_message']
    combined_reviews.extend(rating_reviews)
    combined_ratings.extend([rating_value] * len(rating_reviews))

# Translate combined review messages from Portuguese to English
translator = Translator()
translated_reviews = [translator.translate(msg, src='pt', dest='en').text if pd.notnull(msg) and msg.lower() != "no message" else "No Message" for msg in combined_reviews]

# Create a DataFrame with the translated reviews and their corresponding ratings
reviews_table = pd.DataFrame({'Rating': combined_ratings, 'Review Message': translated_reviews})

# Filter out rows with "No Message"
reviews_table = reviews_table[reviews_table['Review Message'] != "No Message"]

# Set the fixed number of rows to be 30
num_rows = 30

# Create a custom table-like display
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')  # Turn off axis labels and ticks

# Set the column titles to be black
ax.add_patch(plt.Rectangle((0, 0.95), 0.5, 0.1, color='white', ec='none'))
ax.text(0.25, 0.97, 'Rating', ha='left', color='black', weight='bold')
ax.add_patch(plt.Rectangle((0.5, 0.95), 0.5, 0.1, color='white', ec='none'))
ax.text(0.75, 0.97, 'Review Message', color='black', weight='bold')

# Set the color of the cells
for i in range(min(num_rows, len(reviews_table))):
    for j, col in enumerate(reviews_table.columns):
        # Set the text color to be black
        ax.text((2 * j + 1) / 4, 0.95 - i * 0.1, str(reviews_table.iloc[i, j]), va='center', color='black')

plt.title('Table of Review Messages and Ratings (1 to 5) (Delivery Time: 80 - 160 days)', fontsize=18, fontweight='bold', x=0.5)

plt.show()
