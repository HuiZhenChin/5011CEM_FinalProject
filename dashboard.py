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