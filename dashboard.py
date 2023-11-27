import sys
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#graph lian
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from sklearn.metrics import r2_score

# graph chin
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from googletrans import Translator
from textblob import TextBlob
from matplotlib.lines import Line2D

#delay imports
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

#graph peh
import sys
import os
import seaborn as sns 
import geopandas as gpd

class InputField(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        # Prediction Analysis [Product Sales Prediction for next month]
        self.Prediction_analysis_product_selection = QComboBox(self)
        self.Prediction_analysis_product_selection.addItems(self.product_categories())
        self.Prediction_analysis_product_selection.currentIndexChanged.connect(self.update_product_prediction_graph)

        # Geo Plot[Most Purchased Product Category]
        self.Most_purchased_product_category_selection = QComboBox(self)
        self.Most_purchased_product_category_selection.addItem("Default")
        self.Most_purchased_product_category_selection.addItems(["2016-09", "2016-10", "2016-12", "2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01", "2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08"])
        self.Most_purchased_product_category_selection.currentIndexChanged.connect(self.update_geo_plot)

        # 3D bar plot[sentiment analysis]
        self.Sentiment_analysis_product_selection = QComboBox(self)
        self.Sentiment_analysis_product_selection.addItem("Default")
        self.Sentiment_analysis_product_selection.addItems(self.product_categories())
        self.Sentiment_analysis_product_selection.currentIndexChanged.connect(self.update_spesific_3d_bar_plot)

        self.Sentiment_analysis_time_field_1 = QLineEdit(self)
        self.Sentiment_analysis_time_field_1.setText("0")
        self.Sentiment_analysis_time_field_1.editingFinished.connect(self.update_spesific_3d_bar_plot)
        self.Sentiment_analysis_time_field_1.setEnabled(False)

        self.Sentiment_analysis_time_field_2 = QLineEdit(self)
        self.Sentiment_analysis_time_field_2.setText("200")
        self.Sentiment_analysis_time_field_2.editingFinished.connect(self.update_spesific_3d_bar_plot)
        self.Sentiment_analysis_time_field_2.setEnabled(False)

        # Comments graph[Comment Analysis]
        self.Comment_analysis_product_selection = QComboBox(self)
        self.Comment_analysis_product_selection.addItems(self.product_categories())
        self.Comment_analysis_product_selection.currentIndexChanged.connect(self.update_comments_plot)

        self.Comment_analysis_rating_selection = QComboBox(self)
        self.Comment_analysis_rating_selection.addItems(["1","2","3","4","5"])
        self.Comment_analysis_rating_selection.currentIndexChanged.connect(self.update_comments_plot)

        # Purchasing Behaviour Graph[Purchasing Behaviour Analysis]
        self.Purchasing_behaviour_product_selection = QComboBox(self)
        self.Purchasing_behaviour_product_selection.addItems(self.product_categories())
        self.Purchasing_behaviour_product_selection.currentIndexChanged.connect(self.update_barplot2D_purchasing_behaviour)

        layout = QHBoxLayout(self)
        # Prediction Analysis [Product Sales Prediction for next month]
        layout.addWidget(self.Prediction_analysis_product_selection)

        # Geo Plot[Most Purchased Product Category]
        layout.addWidget(self.Most_purchased_product_category_selection)
        self.hide_input_b()

        # 3D bar plot[sentiment analysis]
        layout.addWidget(self.Sentiment_analysis_product_selection)
        layout.addWidget(self.Sentiment_analysis_time_field_1)
        layout.addWidget(self.Sentiment_analysis_time_field_2)
        self.hide_input_c()

        # Comments graph[Comment Analysis]
        layout.addWidget(self.Comment_analysis_product_selection)
        layout.addWidget(self.Comment_analysis_rating_selection)
        self.hide_input_d()

        # Purchasing Behaviour Graph[Purchasing Behaviour Analysis]
        layout.addWidget(self.Purchasing_behaviour_product_selection)
        self.hide_input_e()

    def product_categories(self):
        product_categories = pd.unique(pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")[
                                           'product_category_name_english'])
        return product_categories

    def hide_input_a(self):
        self.Prediction_analysis_product_selection.hide()

    def hide_input_b(self):
        self.Most_purchased_product_category_selection.hide()

    def hide_input_c(self):
        self.Sentiment_analysis_product_selection.hide()
        self.Sentiment_analysis_time_field_1.hide()
        self.Sentiment_analysis_time_field_2.hide()

    def hide_input_d(self):
        self.Comment_analysis_product_selection.hide()
        self.Comment_analysis_rating_selection.hide()

    def hide_input_e(self):
        self.Purchasing_behaviour_product_selection.hide()

    def show_input_a(self):
        self.Prediction_analysis_product_selection.show()
        product = self.Prediction_analysis_product_selection.currentText()
        self.hide_input_b()
        self.hide_input_c()
        self.hide_input_d()
        self.hide_input_e()
        self.main_window.graph_widget.productpredictiongraph(product)

    def show_input_b(self):
        self.Most_purchased_product_category_selection.show()
        self.hide_input_a()
        self.hide_input_c()
        self.hide_input_d()
        self.hide_input_e()
        self.main_window.graph_widget.geoplot()

    def show_input_c(self):
        self.Sentiment_analysis_product_selection.show()
        self.Sentiment_analysis_time_field_1.show()
        self.Sentiment_analysis_time_field_2.show()
        self.hide_input_a()
        self.hide_input_b()
        self.hide_input_d()
        self.hide_input_e()
        self.main_window.graph_widget.bar3dgraph()

    def show_input_d(self):
        self.Comment_analysis_product_selection.show()
        self.Comment_analysis_rating_selection.show()
        product = self.Comment_analysis_product_selection.currentText()
        rating = self.Comment_analysis_rating_selection.currentText()
        self.hide_input_a()
        self.hide_input_b()
        self.hide_input_c()
        self.hide_input_e()
        self.main_window.graph_widget.commentsgraph(product,int(rating))

    def show_input_e(self):
        self.Purchasing_behaviour_product_selection.show()
        product = self.Purchasing_behaviour_product_selection.currentText()
        self.hide_input_a()
        self.hide_input_b()
        self.hide_input_c()
        self.hide_input_d()
        self.main_window.graph_widget.barplot2Dpurchasingbehaviour(product)

    def update_product_prediction_graph(self):
        product = self.Prediction_analysis_product_selection.currentText()
        self.main_window.graph_widget.productpredictiongraph(product)

    def update_geo_plot(self):
        time = self.Most_purchased_product_category_selection.currentText()
        if time == "Default":
            self.main_window.graph_widget.geoplot()
        else:
            self.main_window.graph_widget.geoplotspecific(time)

    def update_spesific_3d_bar_plot(self):
        try:
            product = self.Sentiment_analysis_product_selection.currentText()
            # Get the values from the text edit fields
            value_1 = int(self.Sentiment_analysis_time_field_1.text())
            value_2 = int(self.Sentiment_analysis_time_field_2.text())

            if product == "Default":
                self.main_window.graph_widget.bar3dgraph()
                self.Sentiment_analysis_time_field_1.setText("0")
                self.Sentiment_analysis_time_field_2.setText("200")
                self.Sentiment_analysis_time_field_1.setEnabled(False)
                self.Sentiment_analysis_time_field_2.setEnabled(False)
            else:
                if self.Sentiment_analysis_time_field_1.isEnabled() and self.Sentiment_analysis_time_field_2.isEnabled():
                    # Validate the range and relationship between the values
                    if 0 <= value_1 <= 200 and 0 <= value_2 <= 200 and value_1 <= value_2:
                        self.main_window.graph_widget.bar3dgraphspecific(product, value_1, value_2)
                    else:
                        QMessageBox.warning(self, "Invalid Input", "Please ensure 0 <= Value 1 <= Value 2 <= 200.")
                        self.main_window.graph_widget.bar3dgraphspecific(product, 0, 200)
                else:
                    self.Sentiment_analysis_time_field_1.setEnabled(True)
                    self.Sentiment_analysis_time_field_2.setEnabled(True)
                    self.main_window.graph_widget.bar3dgraphspecific(product, 0, 200)
        except ValueError:
            # Handle the case where the input is not a valid integer
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers.")

    def update_comments_plot(self):
        product = self.Comment_analysis_product_selection.currentText()
        rating = self.Comment_analysis_rating_selection.currentText()
        self.main_window.graph_widget.commentsgraph(product, rating)

    def update_barplot2D_purchasing_behaviour(self):
        product = self.Purchasing_behaviour_product_selection.currentText()
        self.main_window.graph_widget.barplot2Dpurchasingbehaviour(product)

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = plt.figure()
        self.layout = QVBoxLayout()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    #miss lian
    def productpredictiongraph(self, product_name):
        self.figure.clear()  # Clear the previous plot
        ax = self.figure.add_subplot(111)  # 2D plot
        try:
            # read the monthly sales data from Final directory
            monthly_sales_data_2017_09 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2017-09.csv")
            monthly_sales_data_2017_10 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2017-10.csv")
            monthly_sales_data_2017_11 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2017-11.csv")
            monthly_sales_data_2017_12 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2017-12.csv")
            monthly_sales_data_2018_01 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-01.csv")
            monthly_sales_data_2018_02 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-02.csv")
            monthly_sales_data_2018_03 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-03.csv")
            monthly_sales_data_2018_04 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-04.csv")
            monthly_sales_data_2018_05 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-05.csv")
            monthly_sales_data_2018_06 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-06.csv")
            monthly_sales_data_2018_07 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-07.csv")
            monthly_sales_data_2018_08 = pd.read_csv("C:/Users/JQgam/Desktop/Final/final_monthly_data_2018-08.csv")

            # combine the monthly data into a single DataFrame
            monthly_sales_data = pd.concat([
                monthly_sales_data_2017_09,
                monthly_sales_data_2017_10,
                monthly_sales_data_2017_11,
                monthly_sales_data_2017_12,
                monthly_sales_data_2018_01,
                monthly_sales_data_2018_02,
                monthly_sales_data_2018_03,
                monthly_sales_data_2018_04,
                monthly_sales_data_2018_05,
                monthly_sales_data_2018_06,
                monthly_sales_data_2018_07,
                monthly_sales_data_2018_08
            ], ignore_index=True)

            # input field
            target_category = product_name

            # filter data for the target product category
            target_data = monthly_sales_data[monthly_sales_data['Most Common Product Category'] == target_category]

            # sort data by the state
            target_data = target_data.sort_values(by=['State'])

            # calculate the average sales increase from the previous record for each state
            target_data['Sales Increase'] = target_data.groupby('State')['Count'].diff()

            # calculate the average sales increase excluding the first record (which has NaN in 'Sales Increase')
            average_sales_increase_by_state = target_data.groupby('State')['Sales Increase'].mean()

            # fill any potential NaN values with 0
            average_sales_increase_by_state = average_sales_increase_by_state.fillna(0)

            # create a DataFrame for predicted next month's sales in each state
            predicted_next_month_sales_by_state = pd.DataFrame({
                'State': average_sales_increase_by_state.index,
                'Predicted Sales': target_data.groupby('State')['Count'].last() + average_sales_increase_by_state.values
            })

            # display the predicted sales for next month
            print("Predicted Sales for September 2018:")
            print(predicted_next_month_sales_by_state)

            # Feature Engineering
            # use State
            X = target_data[['State']]
            y = target_data['Count']

            # encode categorical variables
            X_encoded = pd.get_dummies(X, columns=['State'], drop_first=True)

            # initialize and train a Random Forest Regressor
            model = RandomForestRegressor()
            model.fit(X_encoded, y)

            # make predictions for the target data
            X_encoded_pred = pd.get_dummies(predicted_next_month_sales_by_state[['State']], columns=['State'], drop_first=True)
            predicted_next_month_sales_by_state['Predicted Sales ML'] = model.predict(X_encoded_pred)

            # plot the growth for each state (Previous Data)
            for state in predicted_next_month_sales_by_state['State']:
                state_data = target_data[target_data['State'] == state]
                ax.plot(state_data['State'], state_data['Count'], label=f'{state} Actual', marker='o')

            # plot the predicted sales using the machine learning model (Predicted Data)
            ax.plot(predicted_next_month_sales_by_state['State'], predicted_next_month_sales_by_state['Predicted Sales ML'],
                    label='Predicted Sales (Machine Learning)', marker='o', linestyle='--', color='green')

            # axis title
            ax.set_title(f'Summary Line Graph of Growth for {target_category}')
            ax.set_xlabel('State')
            ax.set_ylabel('Sales Count')
            ax.legend()
            ax.grid(True)

            # show the accuracy table
            print ("Accuracy Table")
            print (" ")
            mse = mean_squared_error(target_data['Count'], model.predict(X_encoded))
            mae = mean_absolute_error(target_data['Count'], model.predict(X_encoded))
            r2 = r2_score(target_data['Count'], model.predict(X_encoded))

            summary_table = pd.DataFrame({
                'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'R-squared'],
                'Value': [mse, mae, r2]
            })

            print(summary_table)

            self.canvas.draw()

        except:
            ax.axis('off')  
            
            num_rows = 1
            ax.set_title(f'Summary Line Graph of Growth for {target_category}', fontsize=16, fontweight='bold', x=0.5)
            ax.text(0.5, 0.5, 'No prediction/previous data for selected category', ha='center', va='center', fontsize=16, color='black')

            self.canvas.draw()


    #miss peh
    def geoplotdataprocessing(self):
        customer= pd.read_csv("C:/Users/JQgam/Desktop/olist_customers_dataset.csv")
        geolocation= pd.read_csv("C:/Users/JQgam/Desktop/olist_geolocation_dataset.csv")
        order_item= pd.read_csv("C:/Users/JQgam/Desktop/olist_order_items_dataset.csv")
        payment= pd.read_csv("C:/Users/JQgam/Desktop/olist_order_payments_dataset.csv")
        review= pd.read_csv("C:/Users/JQgam/Desktop/olist_order_reviews_dataset.csv")
        order= pd.read_csv("C:/Users/JQgam/Desktop/olist_orders_dataset.csv")
        product= pd.read_csv("C:/Users/JQgam/Desktop/olist_products_dataset.csv")
        seller= pd.read_csv("C:/Users/JQgam/Desktop/olist_sellers_dataset.csv")
        category_translation= pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")

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
        state_info_df.to_csv('C:/Users/JQgam/Desktop/state_customer_counts.csv', index=False)

        # storing orders related to a state to a file
        # directory where the state CSV files are located
        files_directory = "C:/Users/JQgam/Desktop"

        # directory to save the product IDs CSV files
        state_directory = "C:/Users/JQgam/Desktop/States"

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

        # storing Product ID of each Order ID into a file
        # directory to save the data (Order Item ID and Product ID) CSV files
        data_directory = 'C:/Users/JQgam/Desktop/ProductIDs'

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
        product_category_directory = 'C:/Users/JQgam/Desktop/ProductCategoryData'

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
        product_category_directory = 'C:/Users/JQgam/Desktop/ProductCategoryData'
        monthly_product_category_directory = 'C:/Users/JQgam/Desktop/Monthly Product Category'

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
        monthly_product_category_directory = 'C:/Users/JQgam/Desktop/Monthly Product Category'
        final_directory = 'C:/Users/JQgam/Desktop/Final'

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
        monthly_product_category_directory = 'C:/Users/JQgam/Desktop/Monthly Product Category'
        final_directory = 'C:/Users/JQgam/Desktop/Final'
        translation_file = 'C:/Users/JQgam/Desktop/product_category_name_translation.csv'

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

        # find the most sold product category and store into a file by combining all states
        # directory to save the results file
        results_directory = 'C:/Users/JQgam/Desktop/Results'

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
        print(results_df.head())
        results_df.reset_index(inplace=True)
        results_df.columns = ['State', 'Most Common Product Category', 'Count']

        # define the CSV file name for the results
        results_filename = os.path.join(results_directory, 'most_common_category_results.csv')

        # save the results to a single CSV file
        results_df.to_csv(results_filename, index=False)

    def geoplot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        category_translation= pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")

        # read most_common_product_category.csv
        most_common_categories_df = pd.read_csv("C:/Users/JQgam/Desktop/most_common_category_results.csv")

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
        unique_categories_df.to_csv("C:/Users/JQgam/Desktop/unique_product_categories_with_english.csv", index=False)

        # read unique_product_categories_with_english.csv
        unique_categories_with_english = pd.read_csv("C:/Users/JQgam/Desktop/unique_product_categories_with_english.csv")

        # map the Product Category to Product Category English
        category_mapping = unique_categories_with_english.set_index('Product Category Name')['Product Category English'].to_dict()

        # update the Most Common Product Category in most_common_categories_df
        most_common_categories_df['Most Common Product Category'] = most_common_categories_df['Most Common Product Category'].map(category_mapping)

        # save the updated DataFrame to a new CSV file
        most_common_categories_df.to_csv("C:/Users/JQgam/Desktop/most_common_category_results_updated.csv", index=False)

        # read most_common_category_results_updated.csv
        most_common_updated = pd.read_csv("C:/Users/JQgam/Desktop/most_common_category_results_updated.csv")

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
        most_common_updated.to_csv("C:/Users/JQgam/Desktop/most_common_category_results_updated.csv", index=False)

        # read the GeoDataFrame with the geometry of each state
        sns.set(style="whitegrid", palette="pastel", color_codes=True)
        shp_path = "C:/Users/JQgam/Desktop/bra_adm_ibge_2020_shp/bra_admbnda_adm2_ibge_2020.shp"
        gdf = gpd.read_file(shp_path)

        # merge 'customer_count' data with 'gdf' based on the 'State' column
        state_count = most_common_updated.groupby('State')['Count'].sum().reset_index()

        # merge the 'customer_count' data with 'gdf' based on the 'ADM1_PT'
        gdf = gdf.merge(state_count, left_on='ADM1_PT', right_on='State', how='left')

        # plot the map using the existing ax
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

        self.canvas.draw()

        # print summary
        # extract only the 'State' column from most_common_category_results.csv
        state_summary = most_common_categories_df[['State']]

        # save the 'State' column to a new CSV file
        state_summary.to_csv("C:/Users/JQgam/Desktop/summary_results.csv", index=False)

        # merge the 'State', 'Most Common Product Category', and 'Count' columns from most_common_category_results_updated.csv to the file
        summary_results = pd.read_csv("C:/Users/JQgam/Desktop/summary_results.csv")
        summary_results['State Full Name'] = most_common_updated['State']
        summary_results['Most Common Product Category'] = most_common_updated['Most Common Product Category']
        summary_results['Product Category Count'] = most_common_updated['Count']

        # save the updated summary_results to the same CSV file
        summary_results.to_csv("C:/Users/JQgam/Desktop/summary_results.csv", index=False)

        # print the summary
        print(summary_results.to_string())

    def geoplotspecific(self, time):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        monthly_product_category_directory = 'C:/Users/JQgam/Desktop/Monthly Product Category'
        final_directory = 'C:/Users/JQgam/Desktop/Final'
        translation_file = 'C:/Users/JQgam/Desktop/product_category_name_translation.csv'

        # load the product category name translation data
        translation_data = pd.read_csv(translation_file)

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

        # define the file name based on user input
        file_to_search = f'final_monthly_data_{time}.csv'

        # read the GeoDataFrame with the geometry of each state
        sns.set(style="whitegrid", palette="pastel", color_codes=True)
        shp_path = "C:/Users/JQgam/Desktop/bra_adm_ibge_2020_shp/bra_admbnda_adm2_ibge_2020.shp"
        gdf = gpd.read_file(shp_path)

        # check if the file exists
        file_path = os.path.join(final_directory, file_to_search)
        if os.path.isfile(file_path):
            # read the final monthly data for the user input
            choropleth_data = pd.read_csv(file_path)

            # merge choropleth data with GeoDataFrame based on state abbreviations
            gdf_merged = gdf.merge(choropleth_data, left_on='ADM1_PT', right_on='State', how='left')

            # plot the map
            gdf.boundary.plot(ax=ax, linewidth=1, color='k')

            # use the 'Most Common Product Category' column to define the color tone
            gdf_merged.plot(column='Most Common Product Category', ax=ax, legend=False, cmap='tab10')

            # create a custom legend with dots of color for each product category
            legend_labels = []
            unique_categories = choropleth_data['Most Common Product Category'].dropna().unique()
            num_categories = len(unique_categories)
            colors = plt.cm.get_cmap('tab10', num_categories)

            for category, color in zip(unique_categories, colors(range(num_categories))):
                legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=5))  # Adjust the markersize

            # add the legend to the plot with extra space on the right and bottom
            ax.legend(handles=legend_labels, title='Product Category', loc='upper right', bbox_to_anchor=(1.35, 1.0), borderaxespad=0.5, markerscale=2.0)  # Adjust the markerscale

            # increase overall size of the graph
            plt.subplots_adjust(right=0.85, bottom=0.15)

            plt.title(f"Top Selling Product Category in Brazil - {time}")

            self.canvas.draw()

            # read the final monthly data for the user input into a DataFrame
            user_input_data = pd.read_csv(file_path)

            # print the summary
            print("\nSummary Data:")
            print(user_input_data.to_string())

        else:
            print(f"No data available for {time}. Please check your input.")

    #miss chin
    def bar3dgraph(self):
        self.figure.clear()

        review = pd.read_csv("C:/Users/JQgam/Desktop/olist_order_reviews_dataset.csv")
        order = pd.read_csv("C:/Users/JQgam/Desktop/olist_orders_dataset.csv")
        order_item = pd.read_csv("C:/Users/JQgam/Desktop/olist_order_items_dataset.csv")
        product = pd.read_csv("C:/Users/JQgam/Desktop/olist_products_dataset.csv")
        name = pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")

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
        merge_data['day_difference'] = (
                    merge_data['order_delivered_customer_date'] - merge_data['order_delivered_carrier_date']).dt.days
        # add one new column in merge_data with rating
        rating = merge_data['review_score']

        # merge with the product DataFrame with product id
        merged_data = merge_data.merge(product, on='product_id', how='inner')

        # create day categories based on delivery time/ length, set range with 10 days
        merge_data['day_category'] = pd.cut(merge_data['day_difference'], bins=range(0, 201, 10))
        pivot_table = merge_data.pivot_table(values='review_score', index='day_category', aggfunc='mean')
        product_category = merged_data.groupby('product_category_name')['review_score'].mean().sort_values(
            ascending=False).reset_index()

        # get translated product category name
        merged_data_with_translation = product_category.merge(name, on='product_category_name', how='left')

        merged_data = merged_data.merge(name, on="product_category_name", how="left")

        # analyze the review score from 80 days to 160 days of delivery time
        delivery_time_range = merged_data[(merged_data['day_difference'] >= 0) & (merged_data['day_difference'] <= 200)]
        count_of_order = len(delivery_time_range)

        # calculate how many order in each product category involved between 80 to 160 days
        product_category_summary = delivery_time_range['product_category_name_english'].value_counts()
        # display a table
        print("Product Category Summary for Orders with 0-200 Days Delivery Time:")
        print(product_category_summary)
        print(f"Number of Orders within the 0-200 days delivery time range: {count_of_order}\n")
        print("Total Count: ", count_of_order)

        category_summary_df = product_category_summary.reset_index()
        category_summary_df.columns = ['Product Category', 'Count']

        # sort the DataFrame by the count of orders in descending order
        category_summary_df = category_summary_df.sort_values(by='Count', ascending=False)

        # select the top N product categories to include in the graph (20 product categories only based on the table shown)
        top_categories = 21
        top_category_names = category_summary_df.iloc[:top_categories, 0].tolist()
        # create a numerical mapping for product category names
        category_mapping = {category: i for i, category in enumerate(top_category_names)}

        # filter the data for the selected top product categories and the selected delivery time range
        filtered_data = merged_data[
            (merged_data['product_category_name_english'].isin(top_category_names)) &
            (merged_data['day_difference'] >= 0) &
            (merged_data['day_difference'] <= 200)

            ]

        filtered_data['review_count'] = 1

        # create a pivot table to show the customer count for each rating of each product category
        pivot_table_customer_count = filtered_data.groupby(['product_category_name_english', 'review_score'])[
            'review_count'].count().unstack(fill_value=0)

        # use the value from table for plotting
        customer_count = pivot_table_customer_count.values

        # get the value of product category and ratings
        product_categories = [str(category) for category in pivot_table_customer_count.index]
        ratings = [str(rating) for rating in pivot_table_customer_count.columns]

        # filter categories with total customer count less than 1000
        filtered_categories = [category for category in top_category_names if
                               pivot_table_customer_count.loc[category].sum() >= 1000]
        x, y = np.meshgrid(np.arange(len(ratings)), np.arange(len(filtered_categories)))
        z = customer_count[pivot_table_customer_count.index.isin(filtered_categories), :]

        # sort the product categories in y-axis according to alphabetical order
        filtered_categories = sorted(filtered_categories)

        x, y = np.meshgrid(np.arange(len(ratings)), np.arange(len(filtered_categories)))
        z = customer_count[pivot_table_customer_count.index.isin(filtered_categories), :]

        ax = self.figure.add_subplot(111, projection='3d')  # 3D plot

        dx = dy = 0.8
        dz = z.flatten()

        # assign colors to each product category
        norm = Normalize(vmin=0, vmax=len(filtered_categories))
        colors = cm.Blues(norm(np.arange(len(filtered_categories))))

        # reshape colors to match the shape of z
        colors_reshaped = colors.reshape(1, -1, 4)

        # repeat the color array for each bar in the plot
        colors_repeated = np.repeat(colors_reshaped, len(ratings), axis=1)

        ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z).ravel(), dx, dy, z.ravel(), shade=True,
                 color=colors_repeated[0])

        # set y-axis labels
        ax.set_yticks(np.arange(len(filtered_categories)))
        ax.set_yticklabels(np.arange(0, 21, 1))

        # set axis title
        ax.set_xticks(np.arange(len(ratings)))
        ax.set_xticklabels(ratings)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Product Category')
        ax.set_zlabel('Customer Count')
        ax.set_title('3D Bar Graph of Customer Count (> 1000) by Rating and Product Category', fontsize=28)

        # plot the legend
        legend_labels_with_numbers = [f"{i}: {category}" for i, category in enumerate(filtered_categories)]

        # assign with specified colors
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=cm.Blues(norm(i))) for i in range(len(filtered_categories))]

        # display the legend with an adjusted position and width
        ax.legend(legend_handles, legend_labels_with_numbers, loc='upper left', bbox_to_anchor=(1.05, 0.5, 0.1, 0.5),
                  title='Product Categories')

        self.canvas.draw()

    def bar3dgraphspecific(self, product_name, time_start, time_end):
        self.figure.clear()

        review = pd.read_csv("C:/Users/JQgam/Desktop/olist_order_reviews_dataset.csv")
        order = pd.read_csv("C:/Users/JQgam/Desktop/olist_orders_dataset.csv")
        order_item = pd.read_csv("C:/Users/JQgam/Desktop/olist_order_items_dataset.csv")
        product = pd.read_csv("C:/Users/JQgam/Desktop/olist_products_dataset.csv")
        name = pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")

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
        merge_data['day_difference'] = (
                    merge_data['order_delivered_customer_date'] - merge_data['order_delivered_carrier_date']).dt.days
        # add one new column in merge_data with rating
        rating = merge_data['review_score']

        # merge with the product DataFrame with product id
        merged_data = merge_data.merge(product, on='product_id', how='inner')
        # create day categories based on delivery time/ length, set range with 10 days
        merge_data['day_category'] = pd.cut(merge_data['day_difference'], bins=range(0, 201, 10))
        pivot_table = merge_data.pivot_table(values='review_score', index='day_category', aggfunc='mean')
        product_category = merged_data.groupby('product_category_name')['review_score'].mean().sort_values(
            ascending=False).reset_index()

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
        pivot_table_customer_count = \
        filtered_data_no_duplicates.groupby(['product_category_name_english', 'review_score'])[
            'review_count'].count().unstack(fill_value=0)

        # use the values from the table for plotting
        customer_count = pivot_table_customer_count.values

        # get the values of product category and ratings
        product_categories = [str(category) for category in pivot_table_customer_count.index]
        ratings = [str(rating) for rating in pivot_table_customer_count.columns]

        # create a dataframe based on input data
        specific_category_data = filtered_data[
            (filtered_data['product_category_name_english'] == product_name) &
            (filtered_data['day_difference'] >= time_start) &
            (filtered_data['day_difference'] <= time_end)
            ]

        # group by review score and delivery time, accumulate customer count
        specific_category_customer_count = specific_category_data.groupby(
            ['review_score', 'day_difference']).size().reset_index(name='customer_count')

        # assign colors based on delivery time
        norm = Normalize(vmin=specific_category_customer_count['day_difference'].min(),
                         vmax=specific_category_customer_count['day_difference'].max())
        colors = cm.GnBu(norm(specific_category_customer_count['day_difference']))

        # reshape colors to match the shape of z
        colors_reshaped = colors.reshape(-1, 1, 4)  # Reshape to (num_colors, 1, 4)

        # repeat the color array for each bar in the plot
        colors_repeated = np.repeat(colors_reshaped, len(specific_category_customer_count['review_score']), axis=1)

        if len(specific_category_data) > 0:
            ax = self.figure.add_subplot(111, projection='3d')  # 3D plot
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
            ax.set_title(f'3D Bar Graph for {product_name} ({time_start} to {time_end} days)', fontsize=26)

        elif len(specific_category_data) == 0:
            ax = self.figure.add_subplot(111)
            ax.axis('off')

            num_rows = 1
            plt.title(f'3D Bar Graph for {product_name} ({time_start} to {time_end} days', fontsize=16,
                      fontweight='bold', x=0.5)
            ax.text(0.5, 0.5, 'No orders within these days', ha='center', va='center', fontsize=16, color='black')

        self.canvas.draw()

    #miss chin functions
    def translate_to_english(self, comment):
        translator = Translator()
        translated = translator.translate(comment, src='pt', dest='en')
        return translated.text

    def assign_sentiment_label(self, comment):
        # translate the comment to English
        translated_comment = self.translate_to_english(comment)
        
        # check for specific negative keywords
        negative_keywords = ['not arrived', "unfortunately", "didn't like" ,'nothing', 'wrong', 'missing', 'lower','not received', 'delayed', 'late', 'theft', 'not come']
        
        positive_keywords = ["on time", "earlier", "faster","on schedule" ,"as agreed" ,"ahead ","before", "recommend", "trustworthy", "promised", "confidence", "well", "without problem", "10", "like", "less than 24 hours", "correct", "as advertised", "arrived in record time"]
        
        # convert the translated comment to lowercase for case-insensitive matching
        comment_lower = translated_comment.lower()
        
        # check if any negative keyword is present in the comment
        # if yes, assign negative label
        if any(keyword in comment_lower for keyword in negative_keywords):
            return 'negative'
        
        elif any(keyword in comment_lower for keyword in positive_keywords):
            return 'positive'
        
        # use TextBlob to assign labels (polarity) for reviews
        analysis = TextBlob(translated_comment)
        sentiment = analysis.sentiment.polarity
        
        # assign labels based on sentiment
        if sentiment > 0:
            return 'positive'
        elif sentiment == 0:
            return 'neutral'
        else:
            return 'negative'
        
    def assign_category(self, comment):
        # translate the comment to English
        translated_comment = self.translate_to_english(comment)

        # check for specific keywords related to delivery time, product quality, wrong product, and missing product
        delivery_keywords = ['not arrived', 'did not arrive', "didn't arrive", "didn't receive", 'delayed', 'late', 'not come', 'not received', "haven't received", "haven't arrived"]
        quality_keywords = ['poor', 'bad', 'terrible', 'expected', "didn't like", 'did not receive', 'horrible', 'not what I expected', 'thin', 'not very good', 'torn', 'not meet expectations']
        wrong_product_keywords = ['wrong', 'not what I ordered', 'different', 'color', 'description']
        missing_product_keywords = ['missing', 'one', 'theft', '1', 'did not deliver']

        # convert the translated comment to lowercase for case-insensitive matching
        comment_lower = translated_comment.lower()

        # check if any keywords related to delivery time 
        if any(keyword in comment_lower for keyword in delivery_keywords):
            return 'Delivery Time'
        
        # check if any keywords related to product quality 
        elif any(keyword in comment_lower for keyword in quality_keywords):
            return 'Product Quality'
        
        # check if any keywords related to receiving the wrong product 
        elif any(keyword in comment_lower for keyword in wrong_product_keywords):
            return 'Wrong Product'
        
        # check if any keywords related to missing products
        elif any(keyword in comment_lower for keyword in missing_product_keywords):
            return 'Missing Product'
        
        # if none of the keywords matched, return Others
        return 'Other'
    
    def commentsgraph(self, product_name, rating_var):
        self.figure.clear()
        ax = self.figure.add_subplot(111)  # 2D plot
        # read the csv file
        review= pd.read_csv("C:/Users/JQgam/Desktop/olist_order_reviews_dataset.csv")
        order= pd.read_csv("C:/Users/JQgam/Desktop/olist_orders_dataset.csv")
        order_item= pd.read_csv("C:/Users/JQgam/Desktop/olist_order_items_dataset.csv")
        product= pd.read_csv("C:/Users/JQgam/Desktop/olist_products_dataset.csv")
        name= pd.read_csv("C:/Users/JQgam/Desktop/product_category_name_translation.csv")


        # remove those null values rows
        order.dropna(inplace=True)
        product.dropna(inplace=True)
        review.dropna(inplace=True)

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

        # Create a numerical mapping for product category names
        category_mapping = {category: i for i, category in enumerate(top_category_names)}

        # input field
        selected_ratings = [int(rating_var)]

        # filter the data for the selected top product categories and the selected delivery time range
        filtered_data = merged_data[
            (merged_data['product_category_name_english'] == product_name) &
            (merged_data['day_difference'] >= 0) &
            (merged_data['day_difference'] <= 200) &
            (merged_data['review_score'].isin(selected_ratings))
        ]

        # lists
        # store reviews
        combined_reviews = []
        # store ratings
        combined_ratings = []
        # store labels
        sentiment_labels = []
        # store translated reviews
        translated_reviews = []

        for rating_value in selected_ratings:
            rating_reviews = filtered_data[filtered_data['review_score'] == rating_value]['review_comment_message']
            
            for review in rating_reviews:
                # get sentiment label for each review
                sentiment_label = self.assign_sentiment_label(review)
                
                # translate each review
                translated_review = self.translate_to_english(review)
                translated_reviews.append(translated_review)
                
                # append the results to respective lists
                combined_reviews.append(translated_review)
                combined_ratings.append(rating_value)
                sentiment_labels.append(sentiment_label)

        # create reviews_table DataFrame
        reviews_table = pd.DataFrame({'Rating': combined_ratings, 'Review Message': combined_reviews, 'Sentiment': sentiment_labels})

        # filter out rows with "No Message"
        reviews_table = reviews_table[reviews_table['Review Message'] != "No Message"] 

        category_labels = []
        for index, row in reviews_table.iterrows():
            review = row['Review Message']
            sentiment_label = row['Sentiment']
            
            # block the category label for positive reviews
            if sentiment_label == 'positive':
                category_labels.append('Positive')
            else:
                category_label = self.assign_category(review)
                category_labels.append(category_label)

        # add the category column to the reviews_table DataFrame
        reviews_table['Category'] = category_labels

        # drop duplicated reviews
        reviews_table = reviews_table.drop_duplicates(subset='Review Message')

        # create a new table to store those negative and neutral reviews
        filtered_reviews_table = reviews_table[reviews_table['Sentiment'].isin(['negative', 'neutral'])]

        # if there are no negative or neutral comments for a category, include positive comments in the table
        positive_reviews_table = reviews_table[reviews_table['Sentiment'] == 'positive']

        # if there is no negative and neutral reviews, show the positive reviews

        positive_reviews_table = positive_reviews_table.drop(['Sentiment', 'Category'], axis=1)

        # combine positive and filtered reviews into a single table
        combined_reviews_table = pd.concat([filtered_reviews_table, positive_reviews_table])

        # save the combined reviews table (positive and filtered) to a text file
        combined_reviews_table_file_name = f'{product_name}_combined_reviews_table.txt'
        combined_reviews_table.to_csv(combined_reviews_table_file_name, index=False, sep='\t')
        print(f"Combined Reviews table saved to '{combined_reviews_table_file_name}'")

        # plot the bar chart only if there are negative or neutral comments
        if len(filtered_reviews_table) > 0:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            category_counts = filtered_reviews_table['Category'].value_counts()
            
            # colour for each category bar
            category_colors = {
                'Delivery Time': '#C1E3FA',
                'Product Quality': '#FFDDA7',
                'Wrong Product': '#B0F5E1',
                'Missing Product': '#FFC3BD',
                'Other': '#9DA9B5'
            }

            # create a DataFrame with all categories and their counts (filling with zeros where necessary)
            all_categories = pd.DataFrame({'Category': list(category_colors.keys())})
            # rename Category to Category
            all_categories_counts = category_counts.reset_index().rename(columns={'index': 'Category', 'Category': 'Category'})
            # rename the 'count' column to 'Count'
            all_categories_counts = all_categories_counts.rename(columns={'count': 'Count'})

            # merge the DataFrames for 'all_categories' and 'all_categories_counts' on 'Category'
            all_categories = pd.merge(all_categories, all_categories_counts.reset_index(), on='Category',
                                    how='left').fillna(0)

            # sort the DataFrame by the count of orders in descending order
            all_categories = all_categories.sort_values(by='Count', ascending=False)
            
            # select the top N categories to include in the graph
            top_categories = 5
            top_categories_df = all_categories.head(top_categories)
            
            # plot bar chart
            ax.bar(top_categories_df['Category'], top_categories_df['Count'], color=[category_colors[category] for category in top_categories_df['Category']])
            
            # set title
            ax.set_title(f'Category Counts of Ratings {selected_ratings} for {product_name}')
            ax.set_xlabel('Review Comment Category')
            ax.set_xticks(range(len(top_categories_df['Category'])))
            ax.set_xticklabels(top_categories_df['Category'], rotation=360, ha='center', fontsize="8")
            ax.set_ylabel('Count')
            
            # display the count values on top of each bar
            for i, count in enumerate(top_categories_df['Count']):
                ax.text(i, count + 0.1, str(int(count)), ha='center', fontsize="8")
            
            # create legend handles and labels
            legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors[category], markersize=8) for category in top_categories_df['Category']]
            legend_labels = top_categories_df['Category']
            
            # display the legend outside the plot
            ax.legend(legend_handles, legend_labels, title='Category Counts', title_fontsize='8', bbox_to_anchor=(1.05, 1), loc='upper left',  fontsize='8')
            
            self.canvas.draw()

        # display the table only if there are positive comments
        elif len(positive_reviews_table) > 0:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            #fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')  

            # set the column titles to be black
            ax.add_patch(plt.Rectangle((0, 0.95), 0.5, 0.1, color='white', ec='none'))
            ax.text(0.10, 0.97, 'Rating', ha='left', color='black', weight='bold')
            ax.add_patch(plt.Rectangle((0.5, 0.95), 0.5, 0.1, color='white', ec='none'))
            ax.text(0.55, 0.97, 'Review Message', color='black', weight='bold')

            num_rows = len(positive_reviews_table)  # display all positive reviews

            # set the color of the cells
            for i in range(num_rows):
                for j, col in enumerate(positive_reviews_table.columns):
                    # set the text color to be black
                    ax.text((2 * j + 1) / 8, 0.90 - i * 0.1, str(positive_reviews_table.iloc[i, j]), va='center', color='black')

            ax.set_title(f'Table of Positive Review Messages and Ratings {selected_ratings} of {product_name}', fontsize=16, fontweight='bold', x=0.5)

            self.canvas.draw()
            
        # if there is no reviews at all for certain ratings
        elif len(filtered_reviews_table) == 0 or len(positive_reviews_table) == 0:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            #, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')  
            
            num_rows = 1
            ax.set_title(f'Table of Review Messages and Ratings {selected_ratings} of {product_name}', fontsize=16, fontweight='bold', x=0.5)
            ax.text(0.5, 0.5, 'No comments for the selected ratings', ha='center', va='center', fontsize=16, color='black')

            self.canvas.draw()

    #jq
    def barplot2Dpurchasingbehaviour(self, product_category):
        self.figure.clear()  
        ax = self.figure.add_subplot(111) 

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
        product_data = merged_data[merged_data['product_category_name_english'] == product_category]
        
        # Get highest and lowest prices
        highest_price_row = product_data.loc[product_data['payment_value'].idxmax()]
        lowest_price_row = product_data.loc[product_data['payment_value'].idxmin()]
        
        # Get payment methods for highest and lowest prices
        highest_price_method = highest_price_row['payment_type']
        lowest_price_method = lowest_price_row['payment_type']
        
        payment_counts = product_data['payment_type'].value_counts()

        if not payment_counts.empty:
            ax.bar(payment_counts.index, payment_counts.values)
            ax.set_xlabel('Payment Method')
            ax.set_ylabel('Customer/Order Count')
            
            # Include highest and lowest prices in the graph title
            ax.set_title(f'Customer Behavior Analysis for {product_category}')
            
            # Annotate the graph with the prices and payment methods
            ax.annotate(f'Highest Price: ${highest_price_row["payment_value"]:.2f} (Method: {highest_price_method})', 
                        xy=(0.5, 0.95), xycoords='axes fraction', ha='center')
            ax.annotate(f'Lowest Price: ${lowest_price_row["payment_value"]:.2f} (Method: {lowest_price_method})', 
                        xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
            
            self.canvas.draw()
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Big Data Analysis of Olist")

        self.title1 = QLabel("OLIST")
        self.title1.setStyleSheet("QLabel{font-size: 20pt;color:white;}")

        self.title2 = QLabel("Data Dashboard")
        self.title2.setStyleSheet("QLabel{font-size: 10pt;color:white;}")

        self.button_a = QPushButton("Product Sales Prediction", self)
        self.button_a.setFixedHeight(50)
        self.button_a.setStyleSheet("QPushButton{border-style: outset;color: #ffffff;} QPushButton::hover{background-color : #11469c;}")

        self.button_b = QPushButton("Most Purchased Product Category", self)
        self.button_b.setFixedHeight(50)
        self.button_b.setStyleSheet("QPushButton{border-style: outset;color: #ffffff;} QPushButton::hover{background-color : #11469c;}")

        self.button_c = QPushButton("Customer Sentiment", self)
        self.button_c.setFixedHeight(50)
        self.button_c.setStyleSheet("QPushButton{border-style: outset;color: #ffffff;} QPushButton::hover{background-color : #11469c;}")

        self.button_d = QPushButton("Comment Analysis", self)
        self.button_d.setFixedHeight(50)
        self.button_d.setStyleSheet("QPushButton{border-style: outset;color: #ffffff;} QPushButton::hover{background-color : #11469c;}")

        self.button_e = QPushButton("Purchasing Behaviour", self)
        self.button_e.setFixedHeight(50)
        self.button_e.setStyleSheet("QPushButton{border-style: outset;color: #ffffff;} QPushButton::hover{background-color : #11469c;}")

        layoutright = QVBoxLayout(self)
        layoutright.addWidget(self.title1)
        layoutright.addWidget(self.title2)

        picd = QByteArray.fromBase64(b'iVBORw0KGgoAAAANSUhEUgAABLAAAASwCAIAAABkQySYAAAgAElEQVR4AezdwUrj3tsH8PcC5g68gvECXLh249Ld7Fy6mbXgDegNOBcw4N7FbGfl0tUMgoMgDIqCCEJRGBCEvvzpS97+23pM2zQ5J8/nx/AjNmmbfJ6Tk3xP0/R/Pn3+5R8BAgQIECBAgAABAgQIBBT4n4DbbJMJECBAgAABAgQIECBA4NPnXwKhD0gJECBAgAABAgQIECAQVEAgDFp4wyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAg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BBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIECBAQCAVCAgQIECBAgAABAgQIBBUQCIMW3lgIAQIECBAgQIAAAQIEBEKBkAABAgQIECBAgAABAkEFBMKghTcWQoAAAQIECBAgQIAAAYFQICRAgAABAgQIECBAgEBQAYEwaOGNhRAgQIAAAQIECBAgQEAgFAgJECBAgAABAgQIECAQVEAgDFp4YyEECBAgQIAAAQIE/rcdOyYAAABgENS/tUGkArscAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCAMEwuwAAATwSURBVKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECglAQEiBAgAABAgQIECBAYCogCKfD+0IIECBAgAABAgQIECAgCAUhAQIECBAgQIAAAQIEpgKCcDq8L4QAAQIECBAgQIAAAQKCUBASIECAAAECBAgQIEBgKiAIp8P7QggQIECAAAECBAgQICAIBSEBAgQIECBAgAABAgSmAoJwOrwvhAABAgQIECBAgAABAoJQEBIgQIAAAQIECBAgQGAqIAinw/tCCBAgQIAAAQIECBAgIAgFIQECBAgQIECAAAECBKYCgnA6vC+EAAECBAgQIECAAAECAaMejAi/UxIpAAAADmVYSWZNTQAqAAAACAAAAAAAAADSU5MAAAAASUVORK5CYII=')
        t_img = QPixmap(QImage.fromData(picd))
        self.setWindowIcon(QIcon(t_img))

        logo_widget = QLabel()
        logo_widget.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        logo_widget.setFixedSize(60, 60)
        logo = QPixmap(QImage.fromData(picd))
        logo = logo.scaled(logo_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_widget.setPixmap(logo)

        layouttop = QHBoxLayout(self)
        layouttop.addWidget(logo_widget)
        layouttop.addLayout(layoutright)

        layout = QVBoxLayout(self)
        layout.addLayout(layouttop)
        layout.addWidget(self.button_a)
        layout.addWidget(self.button_b)
        layout.addWidget(self.button_c)
        layout.addWidget(self.button_d)
        layout.addWidget(self.button_e)
        layout.addStretch()

        self.button_a.clicked.connect(self.show_content_a)
        self.button_b.clicked.connect(self.show_content_b)
        self.button_c.clicked.connect(self.show_content_c)
        self.button_d.clicked.connect(self.show_content_d)
        self.button_e.clicked.connect(self.show_content_e)

        self.side_menu = QWidget()
        self.side_menu.setLayout(layout)
        self.side_menu.setStyleSheet("background-color: #031C44;")

        self.input_field = InputField(self)

        self.content_widget = QWidget(self)

        self.graph_widget = MatplotlibWidget(self)
        
        self.graph_widget.geoplotdataprocessing()
        self.graph_widget.productpredictiongraph("health_beauty")
        
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.addWidget(self.input_field, 1)
        self.content_layout.addWidget(self.graph_widget, 15)
        self.content_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.side_menu, 2)
        main_layout.addWidget(self.content_widget, 9)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def show_content_a(self):
        self.input_field.show_input_a()

    def show_content_b(self):
        self.input_field.show_input_b()

    def show_content_c(self):
        self.input_field.show_input_c()

    def show_content_d(self):
        self.input_field.show_input_d()

    def show_content_e(self):
        self.input_field.show_input_e()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()
    sys.exit(app.exec_())
