import sys
from PySide6.QtCore import *
from PySide6.QtWidgets import *
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

class SideMenuWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.button_a = QPushButton("Product Sales Prediction", self)
        self.button_b = QPushButton("Most Purchased Product Category", self)
        self.button_c = QPushButton("Customer Sentiment", self)
        self.button_d = QPushButton("Comment Analysis", self)
        self.button_e = QPushButton("Purchasing Behaviour", self)

        layout = QVBoxLayout(self)
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

    def show_content_a(self):
        self.main_window.input_field.show_input_a()

    def show_content_b(self):
        self.main_window.input_field.show_input_b()

    def show_content_c(self):
        self.main_window.input_field.show_input_c()

    def show_content_d(self):
        self.main_window.input_field.show_input_d()

    def show_content_e(self):
        self.main_window.input_field.show_input_e()

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
        
    # assign category label for negative and neutral review
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

        self.side_menu = SideMenuWidget(self)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()
    sys.exit(app.exec_())
