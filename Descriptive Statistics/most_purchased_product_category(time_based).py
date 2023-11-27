import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load datasets
orders = pd.read_csv("olist_orders_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
translations = pd.read_csv("product_category_name_translation.csv")

# Check for missing values
orders.isnull().sum()
order_items.isnull().sum()
products.isnull().sum()
translations.isnull().sum()

# Handle missing values
products['product_category_name'].fillna(products['product_category_name'].mode()[0], inplace=True)

# Merge datasets
order_items = pd.merge(order_items, products[['product_id', 'product_category_name']], on='product_id')
order_items = pd.merge(order_items, translations, on='product_category_name')
merged_data = pd.merge(orders[['order_id', 'order_purchase_timestamp']], order_items[['order_id', 'product_category_name_english']], on='order_id')

# Convert 'order_purchase_timestamp' to datetime
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

# Extract time-based features
merged_data['year_month'] = merged_data['order_purchase_timestamp'].dt.to_period('M')  # Convert to Period type

# Count purchases per product category per month
purchase_counts = merged_data.groupby(['year_month', 'product_category_name_english']).size().reset_index(name='purchase_count')

# Find the most purchased product category per month
most_purchased = purchase_counts.groupby('year_month')['purchase_count'].idxmax()
top_categories = purchase_counts.loc[most_purchased]

# Specify colors
category_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

# Sort the DataFrame by 'year_month' in ascending order
top_categories['year_month'] = top_categories['year_month'].astype(str)  # Convert to string for sorting
top_categories['year_month'] = pd.to_datetime(top_categories['year_month'])  # Convert to datetime for correct sorting
top_categories.sort_values(by='year_month', inplace=True)

# Plot the results using a scatter plot with specified colors
fig, ax = plt.subplots(figsize=(12, 8))

# Iterate through unique product categories
for i, category in enumerate(top_categories['product_category_name_english'].unique()):
    category_data = top_categories[top_categories['product_category_name_english'] == category]
    ax.scatter(category_data['year_month'], category_data['purchase_count'], label=category, alpha=0.7, color=category_colors[i])

ax.set_title('Most Purchased Product Category Over Time')
ax.set_xlabel('Year-Month')
ax.set_ylabel('Purchase Count')

# Convert 'year_month' to a Pandas PeriodIndex for formatting
formatted_ticks = top_categories['year_month'].dt.strftime('%b %Y')

ax.set_xticks(top_categories['year_month'].unique())
ax.set_xticklabels(formatted_ticks, rotation=45)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Store the most purchased product category, year, and month into a DataFrame
most_purchased_summary = top_categories[['year_month', 'product_category_name_english', 'purchase_count']].copy()
most_purchased_summary.rename(columns={'product_category_name_english': 'most_purchased_category'}, inplace=True)

# Reset the index to start from 0
most_purchased_summary.reset_index(drop=True, inplace=True)

# Save the summary to a CSV file
most_purchased_summary.to_csv("most_purchased_summary.csv", index=False)

# Print the updated summary
print("Most Purchased Product Category Summary:")
print(most_purchased_summary.to_string(index=False))
