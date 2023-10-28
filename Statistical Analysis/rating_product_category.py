import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
import matplotlib.pyplot as plt 
warnings.filterwarnings('ignore')

review= pd.read_csv("olist_order_reviews_dataset.csv")
order= pd.read_csv("olist_order_items_dataset.csv")
product= pd.read_csv("olist_products_dataset.csv")
name= pd.read_csv("product_category_name_translation.csv")
# -product based analysis(based on product)
# -in terms of what product has higher rating 
# and what product has lower

# remove those null values row
product.dropna(inplace=True)
review['review_comment_title'].fillna("No Title", inplace=True)
review['review_comment_message'].fillna("No Message", inplace=True)

merge_data = review.merge(order, on='order_id', how='inner')
merged_data = merge_data.merge(product, on='product_id', how='inner')
merged_data = merge_data.merge(product, on='product_id', how='inner')

product_category_ratings = merged_data.groupby('product_category_name')['review_score'].mean().sort_values(ascending=False).reset_index()

merged_data_with_translation = product_category_ratings.merge(name, on='product_category_name', how='left')

merged_data_with_translation = merged_data_with_translation.sort_values(by='review_score', ascending=True)

# display the product categories with average ratings
print("Product Categories with Translation and Average Ratings:")
print(merged_data_with_translation[['product_category_name_english', 'review_score']])

plt.figure(figsize=(12, 8))

# countplot 
countplot = sns.countplot(data=merged_data, x='review_score', palette='viridis')

unique_categories = merged_data['product_category_name'].unique()
color_palette = sns.color_palette("viridis", n_colors=len(unique_categories))

# legend box with translated category names (excluding NaN)
legend_labels = []

# filter out NaN values when creating legend labels
for category, color in zip(merged_data_with_translation['product_category_name'], color_palette):
    translated_name = merged_data_with_translation.loc[merged_data_with_translation['product_category_name'] == category, 'product_category_name_english'].values[0]
    if pd.notna(translated_name):  
        legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=translated_name, markerfacecolor=color, markersize=10))

plt.legend(handles=legend_labels, title='Product Category', loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Products by Rating')

plt.show()

# finding: cds_dvds_musicals has highest rating score, security_and_services has lowest rating score