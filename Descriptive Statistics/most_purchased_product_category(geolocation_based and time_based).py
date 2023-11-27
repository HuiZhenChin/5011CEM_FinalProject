import pandas as pd
import os
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns 
import csv
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
# iterate through the 'customer' DataFrame and add unique city names to the list
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

# storing Order Purchase Timestamp and Product ID of each Order ID into a file
# directory to save the data (Order Purchase Timestamp and Product ID) CSV files
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

    # retrieve 'order_purchase_timestamp' and 'product_id' from the 'order_item' and 'product' DataFrames
    order_items_in_state = order_item[order_item['order_id'].isin(unique_order_ids)]
    
    # merge with 'product' DataFrame to get 'product_id'
    data_df = pd.merge(order_items_in_state[['order_id', 'product_id']], product[['product_id']], on='product_id', how='left')

    # drop duplicates to keep only unique combinations of 'order_id' and 'product_id'
    data_df.drop_duplicates(inplace=True)

    # add 'order_purchase_timestamp' from the 'state_orders' DataFrame
    data_df = pd.merge(data_df, state_orders[['order_id', 'order_purchase_timestamp']], on='order_id', how='left')

    # drop the 'order_id' column
    data_df.drop(columns=['order_id'], inplace=True)

    # define the CSV file name for the data in the current state
    data_filename = os.path.join(data_directory, f'data_for_{state_name}.csv')

    # save the data to a separate CSV file for the current state
    data_df.to_csv(data_filename, index=False)

print("Data (Order Purchase Timestamp and Product ID) for each state saved to separate CSV files in the 'ProductIDs' folder.")

# storing Order Purchase Timestamp, Product ID, and Product Category Name into a file
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

    # retrieve 'order_purchase_timestamp', 'product_id', and 'product_category_name' from 'order_item' and 'product' DataFrames
    product_category_data = pd.merge(
        order_item[order_item['order_id'].isin(unique_order_ids)][['order_id', 'order_item_id', 'product_id']],
        product[['product_id', 'product_category_name']],
        on='product_id',
        how='left'
    )

    # merge with 'state_orders' DataFrame to get 'order_purchase_timestamp'
    product_category_data = pd.merge(
        product_category_data,
        state_orders[['order_id', 'order_purchase_timestamp']],
        on='order_id',
        how='left'
    )

    # create a DataFrame with 'order_purchase_timestamp', 'product_id', and 'product_category_name'
    data_df = pd.DataFrame({
        'Order Purchase Timestamp': product_category_data['order_purchase_timestamp'],
        'Product ID': product_category_data['product_id'],
        'Product Category Name': product_category_data['product_category_name']
    })

    # define the CSV file name for the data in the current state
    data_filename = os.path.join(product_category_directory, f'product_category_data_{state_name}.csv')

    # save the data to a separate CSV file for the current state
    data_df.to_csv(data_filename, index=False)

print("Product Category Data (Order Purchase Timestamp, Product ID, and Product Category Name) for each state saved to separate CSV files in the 'ProductCategoryData' folder.")

# define directories
product_category_directory = 'ProductCategoryData'
monthly_product_category_directory = 'Monthly Product Category'

# create the product category directory if it doesn't exist
os.makedirs(product_category_directory, exist_ok=True)

# create the monthly product category directory if it doesn't exist
os.makedirs(monthly_product_category_directory, exist_ok=True)

# get all CSV files in the state files directory
state_files = [file for file in os.listdir(product_category_directory) if file.startswith('product_category_data_') and file.endswith('.csv')]

# loop through each state CSV file
for state_file in state_files:
    state_name = state_file[len('product_category_data_'):-len('.csv')]
    state_folder_path = os.path.join(monthly_product_category_directory, f'State_{state_name}')

    # Create state-specific folder inside Monthly Product Category
    os.makedirs(state_folder_path, exist_ok=True)

    # Read the state-specific product category data
    state_data = pd.read_csv(os.path.join(product_category_directory, state_file))

    # Convert 'Order Purchase Timestamp' to datetime format
    state_data['Order Purchase Timestamp'] = pd.to_datetime(state_data['Order Purchase Timestamp'])

    # Extract month and year from 'Order Purchase Timestamp'
    state_data['Month_Year'] = state_data['Order Purchase Timestamp'].dt.to_period('M')

    # Loop through each month and save data to a separate CSV file
    for month_year, month_data in state_data.groupby('Month_Year'):
        # Create a DataFrame for the current month
        month_data_df = pd.DataFrame({
            'Order Purchase Timestamp': month_data['Order Purchase Timestamp'],
            'Product ID': month_data['Product ID'],
            'Product Category Name': month_data['Product Category Name']
        })

        # Define the CSV file name for the current month in the current state
        month_filename = os.path.join(state_folder_path, f'{state_name}_monthly_data_{month_year}.csv')

        # Save the data to a separate CSV file for the current month and state
        month_data_df.to_csv(month_filename, index=False)

print("Monthly Product Category Data saved to separate folders for each state.")

# define directories
monthly_product_category_directory = 'Monthly Product Category'
final_directory = 'Final'

# create the final directory if it doesn't exist
os.makedirs(final_directory, exist_ok=True)

# initialize a dictionary to store the results for each month
monthly_results = {}

# get all state folders in the monthly product category directory
state_folders = [folder for folder in os.listdir(monthly_product_category_directory) if os.path.isdir(os.path.join(monthly_product_category_directory, folder))]

# loop through each state folder
for state_folder in state_folders:
    state_name = state_folder[len('State_'):]

    # get all monthly data files for the current state
    state_files = [file for file in os.listdir(os.path.join(monthly_product_category_directory, state_folder)) if file.endswith('.csv')]

    # loop through each monthly data file
    for state_file in state_files:
        month_year = state_file.split('_')[-1][:-4]

        # read the monthly data for the current state and month
        month_data = pd.read_csv(os.path.join(monthly_product_category_directory, state_folder, state_file))

        # find the most common product category name and its count
        most_common_category = month_data['Product Category Name'].mode().values[0]
        most_common_category_count = month_data['Product Category Name'].value_counts().max()

        # store the result in the dictionary
        if month_year not in monthly_results:
            monthly_results[month_year] = []

        monthly_results[month_year].append({
            'State': state_name,
            'Most Common Product Category': most_common_category,
            'Count': most_common_category_count
        })

# loop through each month and save data to a separate CSV file
for month_year, month_data in monthly_results.items():
    # create a DataFrame for the current month
    month_results_df = pd.DataFrame(month_data)

    # define the CSV file name for the final monthly data
    final_filename = os.path.join(final_directory, f'final_monthly_data_{month_year}.csv')

    # save the data to a separate CSV file for the current month
    month_results_df.to_csv(final_filename, index=False)

print("Final Monthly Product Category Data saved to separate files for each month.")

# define directories
monthly_product_category_directory = 'Monthly Product Category'
final_directory = 'Final'
translation_file = 'product_category_name_translation.csv'

# load the product category name translation data
translation_data = pd.read_csv(translation_file)

# create the final directory if it doesn't exist
os.makedirs(final_directory, exist_ok=True)

# initialize a dictionary to store the results for each month
monthly_results = {}

# get all state folders in the monthly product category directory
state_folders = [folder for folder in os.listdir(monthly_product_category_directory) if os.path.isdir(os.path.join(monthly_product_category_directory, folder))]

# loop through each state folder
for state_folder in state_folders:
    state_name = state_folder[len('State_'):]
    
    # get all monthly data files for the current state
    state_files = [file for file in os.listdir(os.path.join(monthly_product_category_directory, state_folder)) if file.endswith('.csv')]
    
    # loop through each monthly data file
    for state_file in state_files:
        month_year = state_file.split('_')[-1][:-4]
        
        # read the monthly data for the current state and month
        month_data = pd.read_csv(os.path.join(monthly_product_category_directory, state_folder, state_file))
        
        # merge with translation data to get English product category names
        month_data = pd.merge(month_data, translation_data, how='left', left_on='Product Category Name', right_on='product_category_name')
        
        # drop unnecessary columns
        month_data.drop(['Product Category Name', 'product_category_name'], axis=1, inplace=True)
        
        # rename the column to 'Product Category Name'
        month_data.rename(columns={'product_category_name_english': 'Product Category Name'}, inplace=True)

        # find the most common product category name and its count
        most_common_category = month_data['Product Category Name'].mode().values[0]
        most_common_category_count = month_data['Product Category Name'].value_counts().max()
        
        # store the result in the dictionary
        if month_year not in monthly_results:
            monthly_results[month_year] = []

        monthly_results[month_year].append({
            'State': state_name,
            'Most Common Product Category': most_common_category,
            'Count': most_common_category_count
        })

# loop through each month and save data to a separate CSV file
for month_year, month_data in monthly_results.items():
    # create a DataFrame for the current month
    month_results_df = pd.DataFrame(month_data)

    # define the CSV file name for the final monthly data
    final_filename = os.path.join(final_directory, f'final_monthly_data_{month_year}.csv')

    # save the data to a separate CSV file for the current month
    month_results_df.to_csv(final_filename, index=False)

print("Final Monthly Product Category Data saved to separate files for each month with translated product category names.")

# define the state mapping
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

# loop through each month and update state name in the final files
for month_year, month_data in monthly_results.items():
    # create a DataFrame for the current month
    month_results_df = pd.DataFrame(month_data)

    # update state names according to the mapping
    month_results_df['State'] = month_results_df['State'].map(state_mapping)

    # define the CSV file name for the final monthly data
    final_filename = os.path.join(final_directory, f'final_monthly_data_{month_year}.csv')

    # save the data to a separate CSV file for the current month
    month_results_df.to_csv(final_filename, index=False)

print("Final Monthly Product Category Data updated with state names according to the mapping.")

# prompt user for input (e.g., '2016-09')
user_input = input("Enter the month and year (in YYYY-MM format): ")

# define the file name based on user input
file_to_search = f'final_monthly_data_{user_input}.csv'

# read the GeoDataFrame with the geometry of each state
sns.set(style="whitegrid", palette="pastel", color_codes=True)
shp_path = "C:/Users/Jolyn Peh/Downloads/bra_adm_ibge_2020_shp/bra_admbnda_adm2_ibge_2020.shp"
gdf = gpd.read_file(shp_path)

# check if the file exists
file_path = os.path.join(final_directory, file_to_search)
if os.path.isfile(file_path):
    # read the final monthly data for the user input
    choropleth_data = pd.read_csv(file_path)

    # merge choropleth data with GeoDataFrame based on state abbreviations
    gdf_merged = gdf.merge(choropleth_data, left_on='ADM1_PT', right_on='State', how='left')

    # plot the map
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    gdf.boundary.plot(ax=ax, linewidth=1, color='k')

    # use the 'Most Common Product Category' column to define the color tone
    gdf_merged.plot(column='Most Common Product Category', ax=ax, legend=False, cmap='tab10')

    # create a custom legend with dots of color for each product category
    legend_labels = []
    unique_categories = choropleth_data['Most Common Product Category'].dropna().unique()
    num_categories = len(unique_categories)
    colors = plt.cm.get_cmap('tab10', num_categories)

    for category, color in zip(unique_categories, colors(range(num_categories))):
        legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=5)) 
        
    # add the legend to the plot with extra space on the right and bottom
    ax.legend(handles=legend_labels, title='Product Category', loc='upper right', bbox_to_anchor=(1.35, 1.0), borderaxespad=0.5, markerscale=2.0) 

    # increase overall size of the graph
    plt.subplots_adjust(right=0.85, bottom=0.15)

    plt.title(f"Top Selling Product Category in Brazil - {user_input}")
    plt.show()
    
    # read the final monthly data for the user input into a DataFrame
    user_input_data = pd.read_csv(file_path)

    # print the summary
    print("\nSummary Data:")
    print(user_input_data.to_string())
    
else:
    print(f"No data available for {user_input}. Please check your input.")
