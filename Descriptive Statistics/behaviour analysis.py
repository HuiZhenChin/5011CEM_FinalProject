import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
orders = pd.read_csv("C:/Users/JQgam/Desktop/olist_orders_dataset.csv")
order_items = pd.read_csv("C:/Users/JQgam/Desktop/olist_order_items_dataset.csv")
order_payments = pd.read_csv("C:/Users/JQgam/Desktop/olist_order_payments_dataset.csv")
products = pd.read_csv("C:/Users/JQgam/Desktop/olist_products_dataset.csv")
name = pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")

# Merge relevant tables
merged_data = pd.merge(orders, order_items, on='order_id')
merged_data = pd.merge(merged_data, order_payments, on='order_id')
merged_data = pd.merge(merged_data, products, on='product_id')
merged_data = pd.merge(merged_data, name, on='product_category_name', how='left')

# Convert 'order_purchase_timestamp' to datetime
merged_data['order_purchase_timestamp'] = pd.to_datetime(merged_data['order_purchase_timestamp'])

# Extract month and year from the purchase timestamp
merged_data['purchase_month'] = merged_data['order_purchase_timestamp'].dt.to_period('M')

# Function to analyze and plot based on product category
def analyze_and_plot_by_product(product_category):
    product_data = merged_data[merged_data['product_category_name_english'] == product_category]
    
    # Get highest and lowest prices
    highest_price_row = product_data.loc[product_data['payment_value'].idxmax()]
    lowest_price_row = product_data.loc[product_data['payment_value'].idxmin()]
    
    # Get payment methods for highest and lowest prices
    highest_price_method = highest_price_row['payment_type']
    lowest_price_method = lowest_price_row['payment_type']
    
    payment_counts = product_data['payment_type'].value_counts()

    if not payment_counts.empty:
        plt.bar(payment_counts.index, payment_counts.values)
        plt.xlabel('Payment Method')
        plt.ylabel('Customer/Order Count')
        
        # Include highest and lowest prices in the graph title
        plt.title(f'Customer Behavior Analysis for {product_category}')
        
        # Annotate the graph with the prices and payment methods
        plt.annotate(f'Highest Price: ${highest_price_row["payment_value"]:.2f} (Method: {highest_price_method})', 
                     xy=(0.5, 0.95), xycoords='axes fraction', ha='center')
        plt.annotate(f'Lowest Price: ${lowest_price_row["payment_value"]:.2f} (Method: {lowest_price_method})', 
                     xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
        
        plt.show()
