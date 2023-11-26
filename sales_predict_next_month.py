import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# read the monthly sales data from Final directory
monthly_sales_data_2017_09 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2017-09.csv")
monthly_sales_data_2017_10 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2017-10.csv")
monthly_sales_data_2017_11 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2017-11.csv")
monthly_sales_data_2017_12 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2017-12.csv")
monthly_sales_data_2018_01 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-01.csv")
monthly_sales_data_2018_02 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-02.csv")
monthly_sales_data_2018_03 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-03.csv")
monthly_sales_data_2018_04 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-04.csv")
monthly_sales_data_2018_05 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-05.csv")
monthly_sales_data_2018_06 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-06.csv")
monthly_sales_data_2018_07 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-07.csv")
monthly_sales_data_2018_08 = pd.read_csv("D:/INTI Degree/DEG YEAR 2 INTI-Sem 5/Big Data Programming Project/Statistical Coding Project/Final/final_monthly_data_2018-08.csv")

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
target_category = "health_beauty"

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

# plot the summary line graph of growth
plt.figure(figsize=(15, 10))

# plot the growth for each state (Previous Data)
for state in predicted_next_month_sales_by_state['State']:
    state_data = target_data[target_data['State'] == state]
    plt.plot(state_data['State'], state_data['Count'], label=f'{state} Actual', marker='o')

# plot the predicted sales using the machine learning model (Predicted Data)
plt.plot(predicted_next_month_sales_by_state['State'], predicted_next_month_sales_by_state['Predicted Sales ML'],
         label='Predicted Sales', marker='o', linestyle='--', color='black')

# axis title
plt.title(f'Summary Line Graph of Growth for {target_category}')
plt.xlabel('State')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel('Sales Count')
plt.legend()
plt.grid(True)
plt.show()

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