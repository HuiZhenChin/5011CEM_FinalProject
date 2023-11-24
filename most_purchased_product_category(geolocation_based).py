import pandas as pd
import os
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
warnings.filterwarnings('ignore')

customer= pd.read_csv("olist_customers_dataset.csv")
geolocation= pd.read_csv("olist_geolocation_dataset.csv")
order_item= pd.read_csv("olist_order_items_dataset.csv")
payment= pd.read_csv("olist_order_payments_dataset.csv")
review= pd.read_csv("olist_order_reviews_dataset.csv")
order= pd.read_csv("olist_orders_dataset.csv")
product= pd.read_csv("olist_products_dataset.csv")
seller= pd.read_csv("olist_sellers_dataset.csv")
category_translation= pd.read_csv("product_category_name_translation.csv")

# remove those null values row
order.dropna(inplace=True)

unique_states = []

# filter customer state
# Iterate through the 'customer' DataFrame and add unique city names to the list
for index, row in customer.iterrows():
    state = row['customer_state']
    if state not in unique_states:
        unique_states.append(state)

# print the unique states
print(unique_states)

# group the customers by 'customer_state' and count the number of customers in each state
state_counts = customer['customer_id'].groupby(customer['customer_state']).count()

# reset the index to make 'customer_state' a column again
state_counts = state_counts.reset_index()

# initialize a list to store (state, customer_count) tuples
state_count_list = []

# iterate through the 'state_counts' DataFrame and create the list of tuples
for index, row in state_counts.iterrows():
    state = row['customer_state']
    customer_count = row['customer_id']
    state_count_list.append((state, customer_count))

# initialize a dictionary to store state information, including customer count and customer IDs
state_info = {}

# iterate through the 'state_count_list' and populate the dictionary
for state, customer_count in state_count_list:
    state_info[state] = {
        'customer_count': customer_count,
        'customer_ids': customer[customer['customer_state'] == state]['customer_id'].tolist()
    }

# create a DataFrame from the state_info dictionary
state_info_df = pd.DataFrame.from_dict(state_info, orient='index')
state_info_df.reset_index(inplace=True)
state_info_df.columns = ['State', 'Customer Count', 'Customer IDs']

# save the DataFrame to a CSV file
state_info_df.to_csv('state_customer_counts.csv', index=False)

# storing orders related to a state to a file
# directory where the state CSV files are located
files_directory = "BigData"

# directory to save the product IDs CSV files
state_directory = "States"

# create the state-specific folder if it doesn't exist
os.makedirs(state_directory, exist_ok=True)

# loop through each state and filter orders related to each customer from the same state
for state, info in state_info.items():
    # Get the customer IDs for the current state
    customer_ids = info['customer_ids']
    
    # Filter orders related to the customer IDs in the current state
    state_orders = order[order['customer_id'].isin(customer_ids)]
    
    # Define the full file path for the current state
    csv_filepath = os.path.join(state_directory, f'orders_for_{state}.csv')
    
    # Save the state-specific orders to a separate CSV file in the specified folder
    state_orders.to_csv(csv_filepath, index=False)

print("State-specific orders saved to the 'States' folder.")

# check if any customer has more than one order in a state file
# get all CSV files in the directory
files = [file for file in os.listdir(state_directory) if file.startswith('orders_for_') and file.endswith('.csv')]

# initialize a list to store customer IDs with two order IDs for each state
customer_ids_with_two_orders = []

# loop through each state CSV file
for state_file in files:
    # read the CSV file for the current state
    state_orders = pd.read_csv(state_file)
    
    # group the orders by customer ID and count the number of orders for each customer
    customer_order_counts = state_orders['order_id'].groupby(state_orders['customer_id']).count()
    
    # filter customer IDs with two order IDs
    customer_ids_with_two_orders_state = customer_order_counts[customer_order_counts > 2].index.tolist()
    
    # append the results to the list
    customer_ids_with_two_orders.append((state_file, customer_ids_with_two_orders_state))

# print the results for each state
# result: No customer has two or more orders in a state
for state_file, customer_ids in customer_ids_with_two_orders:
    print(f"State: {state_file[len('orders_for_'): -len('.csv')]}")
    if customer_ids:
        print(f"Customer IDs with two or more orders: {customer_ids}")
    else:
        print("No customer IDs with two or more orders in this state.")
    print()

# storing Product ID of each Order ID into a file
# directory to save the data (Order Item ID and Product ID) CSV files
data_directory = 'ProductIDs'

# create the data directory if it doesn't exist
os.makedirs(data_directory, exist_ok=True)

# get all CSV files in the state files directory
state_files = [file for file in os.listdir(state_directory) if file.startswith('orders_for_') and file.endswith('.csv')]

# loop through each state CSV file
for state_file in state_files:
    state_name = state_file[len('orders_for_'):-len('.csv')]
    state_orders = pd.read_csv(os.path.join(state_directory, state_file))

    # get unique order IDs in the current state
    unique_order_ids = state_orders['order_id'].unique()

    # retrieve all order item IDs and product IDs from the 'olist_order_items_dataset' using the unique order IDs
    order_items_in_state = order_item[order_item['order_id'].isin(unique_order_ids)]

    # create a DataFrame with Order Item ID and Product ID
    data_df = pd.DataFrame({'Order Item ID': order_items_in_state['order_item_id'], 'Product ID': order_items_in_state['product_id']})

    # define the CSV file name for the data in the current state
    data_filename = os.path.join(data_directory, f'data_for_{state_name}.csv')

    # save the data to a separate CSV file for the current state
    data_df.to_csv(data_filename, index=False)

print("Data (Order Item ID and Product ID) for each state saved to separate CSV files in the 'ProductIDs' folder.")

# storing Order Item ID, Product ID and Product Category Name into a file
# directory to save the product category data CSV files
product_category_directory = 'ProductCategoryData'

# create the product category directory if it doesn't exist
os.makedirs(product_category_directory, exist_ok=True)

# get all CSV files in the state files directory
state_files = [file for file in os.listdir(state_directory) if file.startswith('orders_for_') and file.endswith('.csv')]

# loop through each state CSV file
for state_file in state_files:
    state_name = state_file[len('orders_for_'):-len('.csv')]
    state_orders = pd.read_csv(os.path.join(state_directory, state_file))

    # get unique order IDs in the current state
    unique_order_ids = state_orders['order_id'].unique()

    # retrieve all product IDs and product category names from 'olist_order_items_dataset' and 'olist_products_dataset'
    product_category_data = pd.merge(order_item[order_item['order_id'].isin(unique_order_ids)][['order_item_id', 'product_id']],
                                     product[['product_id', 'product_category_name']],
                                     on='product_id',
                                     how='left')

    # create a DataFrame with Product ID and Product Category Name
    data_df = pd.DataFrame({
        'Product ID': product_category_data['product_id'],
        'Product Category Name': product_category_data['product_category_name']
    })

    # define the CSV file name for the data in the current state
    data_filename = os.path.join(product_category_directory, f'product_category_name_{state_name}.csv')

    # save the data to a separate CSV file for the current state
    data_df.to_csv(data_filename, index=False)

print("Product Category Data (Product ID and Product Category Name) for each state saved to separate CSV files in the 'ProductCategoryData' folder.")

# find the most sold product category and store into a file by combining all states
# directory to save the results file
results_directory = 'C:/Users/Jolyn Peh/Documents/BigData/Results'

# create the results directory if it doesn't exist
os.makedirs(results_directory, exist_ok=True)

# initialize a dictionary to store the results
state_category_counts = {}

# get all CSV files in the product category data directory
category_files = [file for file in os.listdir(product_category_directory) if file.startswith('product_category_name_') and file.endswith('.csv')]

# loop through each product category data file
for category_file in category_files:
    state_name = category_file[len('product_category_name_'):-len('.csv')]
    category_data = pd.read_csv(os.path.join(product_category_directory, category_file))

    # find the most common product category name and its count in the current state
    most_common_category = category_data['Product Category Name'].mode().values[0]
    most_common_category_count = category_data['Product Category Name'].value_counts().max()

    # store the result in the dictionary
    state_category_counts[state_name] = {
        'Most Common Product Category': most_common_category,
        'Count': most_common_category_count
    }

# create a DataFrame from the dictionary
results_df = pd.DataFrame.from_dict(state_category_counts, orient='index')
results_df.reset_index(inplace=True)
results_df.columns = ['State', 'Most Common Product Category', 'Count']

# define the CSV file name for the results
results_filename = os.path.join(results_directory, 'most_common_category_results.csv')

# save the results to a single CSV file
results_df.to_csv(results_filename, index=False)

print("Most common product category results saved to 'most_common_category_results.csv'.")

# read most_common_product_category.csv
most_common_categories_df = pd.read_csv("Results/most_common_category_results.csv")

# extract unique product categories
unique_categories_list = most_common_categories_df['Most Common Product Category'].dropna().unique().tolist()

# create a DataFrame with the unique product categories and their English translations
unique_categories_df = pd.DataFrame({
    'Product Category Name': unique_categories_list,
    'Product Category English': unique_categories_list
})

# map the product categories to their English translations
unique_categories_df['Product Category English'] = unique_categories_df['Product Category Name'].map(
    category_translation.set_index('product_category_name')['product_category_name_english']
)

# save the DataFrame with both names to a new CSV file
unique_categories_df.to_csv("Results/unique_product_categories_with_english.csv", index=False)

# read unique_product_categories_with_english.csv
unique_categories_with_english = pd.read_csv("Results/unique_product_categories_with_english.csv")

# map the Product Category to Product Category English
category_mapping = unique_categories_with_english.set_index('Product Category Name')['Product Category English'].to_dict()

# update the Most Common Product Category in most_common_categories_df
most_common_categories_df['Most Common Product Category'] = most_common_categories_df['Most Common Product Category'].map(category_mapping)

# save the updated DataFrame to a new CSV file
most_common_categories_df.to_csv("Results/most_common_category_results_updated.csv", index=False)

# read most_common_category_results_updated.csv
most_common_updated = pd.read_csv("Results/most_common_category_results_updated.csv")

# extract unique non-null product categories
unique_categories_list = most_common_updated['Most Common Product Category'].dropna().unique().tolist()

# define the state_mapping
state_mapping = {
    'SP': 'São Paulo',
    'SC': 'Santa Catarina',
    'MG': 'Minas Gerais',
    'PR': 'Paraná',
    'RJ': 'Rio de Janeiro',
    'RS': 'Rio Grande do Sul',
    'PA': 'Pará',
    'GO': 'Goiás',
    'ES': 'Espírito Santo',
    'BA': 'Bahia',
    'MA': 'Maranhão',
    'MS': 'Mato Grosso do Sul',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'RN': 'Rio Grande do Norte',
    'PE': 'Pernambuco',
    'MT': 'Mato Grosso',
    'AM': 'Amazonas',
    'AP': 'Amapá',
    'AL': 'Alagoas',
    'RO': 'Rondônia',
    'PB': 'Paraíba',
    'TO': 'Tocantins',
    'PI': 'Piauí',
    'AC': 'Acre',
    'SE': 'Sergipe',
    'RR': 'Roraima'
}

# update the state names in the DataFrame
most_common_updated['State'] = most_common_updated['State'].map(state_mapping)

# save the updated DataFrame back to the CSV file
most_common_updated.to_csv("Results/most_common_category_results_updated.csv", index=False)

# read the GeoDataFrame with the geometry of each state
sns.set(style="whitegrid", palette="pastel", color_codes=True)
shp_path = "C:/Users/Jolyn Peh/Downloads/bra_adm_ibge_2020_shp/bra_admbnda_adm2_ibge_2020.shp"
gdf = gpd.read_file(shp_path)

# merge 'customer_count' data with 'gdf' based on the 'State' column
state_count = most_common_updated.groupby('State')['Count'].sum().reset_index()

# merge the 'customer_count' data with 'gdf' based on the 'ADM1_PT'
gdf = gdf.merge(state_count, left_on='ADM1_PT', right_on='State', how='left')

# plot the map
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
gdf.boundary.plot(ax=ax, linewidth=1, color='k')

# use the 'Most Common Product Category' column to define the color tone
gdf['Color'] = gdf['ADM1_PT'].map(most_common_updated.set_index('State')['Most Common Product Category'].to_dict())
gdf.plot(column='Color', ax=ax, legend=False, cmap='tab10')

# create a custom legend with dots of color for each product category
legend_labels = []
unique_categories = most_common_updated['Most Common Product Category'].dropna().unique()
num_categories = len(unique_categories)
colors = plt.cm.get_cmap('tab10', num_categories)

for category, color in zip(unique_categories, colors(range(num_categories))):
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=5))  # Adjust the markersize

# add the legend to the plot with extra space on the right and bottom
ax.legend(handles=legend_labels, title='Product Category', loc='upper right', bbox_to_anchor=(1.25, 1.0), borderaxespad=0.5, markerscale=2.0)  # Adjust the markerscale

# increase overall size of the graph
plt.subplots_adjust(right=0.85, bottom=0.15)

plt.title("Top Selling Product Category in Brazil")
plt.show()

# print summary
# extract only the 'State' column from most_common_category_results.csv
state_summary = most_common_categories_df[['State']]

# save the 'State' column to a new CSV file
state_summary.to_csv("Results/summary_results.csv", index=False)

# merge the 'State', 'Most Common Product Category', and 'Count' columns from most_common_category_results_updated.csv to the file
summary_results = pd.read_csv("Results/summary_results.csv")
summary_results['State Full Name'] = most_common_updated['State']
summary_results['Most Common Product Category'] = most_common_updated['Most Common Product Category']
summary_results['Product Category Count'] = most_common_updated['Count']

# save the updated summary_results to the same CSV file
summary_results.to_csv("Results/summary_results.csv", index=False)

# print the summary
print(summary_results.to_string())
