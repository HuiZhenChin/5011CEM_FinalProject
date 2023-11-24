import pandas as pd
from googletrans import Translator
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np 
import warnings

warnings.filterwarnings('ignore')

# read the csv file
review= pd.read_csv("olist_order_reviews_dataset.csv")
order= pd.read_csv("olist_orders_dataset.csv")
order_item= pd.read_csv("olist_order_items_dataset.csv")
product= pd.read_csv("olist_products_dataset.csv")
name= pd.read_csv("product_category_name_translation.csv")


# remove those null values rows
order.dropna(inplace=True)
product.dropna(inplace=True)
review.dropna(inplace=True)

# quality-based analysis (based on product)
# in terms of products with lower rating, check for products with bad quality

def translate_to_english(comment):
    translator = Translator()
    translated = translator.translate(comment, src='pt', dest='en')
    return translated.text

def assign_sentiment_label(comment):
    # translate the comment to English
    translated_comment = translate_to_english(comment)
    
    # check for specific negative keywords
    negative_keywords = ['not arrived', "didn't like" ,'nothing', 'wrong', 'missing', 'lower','not received', 'delayed', 'late', 'theft', 'not come']
    
    positive_keywords = ["on time", "before"]
    
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
def assign_category(comment):
    # translate the comment to English
    translated_comment = translate_to_english(comment)

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
product_name = "health_beauty"
selected_ratings = [5]

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
        sentiment_label = assign_sentiment_label(review)
        
        # translate each review
        translated_review = translate_to_english(review)
        translated_reviews.append(translated_review)
        
        # append the results to respective lists
        combined_reviews.append(translated_review)
        combined_ratings.append(rating_value)
        sentiment_labels.append(sentiment_label)

# create reviews_table DataFrame
reviews_table = pd.DataFrame({'Rating': combined_ratings, 'Review Message': combined_reviews, 'Sentiment': sentiment_labels})

# filter out rows with "No Message"
reviews_table = reviews_table[reviews_table['Review Message'] != "No Message"]

# assign category label to each rating
category_labels = []
for index, row in reviews_table.iterrows():
    review = row['Review Message']
    category_label = assign_category(review)
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
if len(filtered_reviews_table) == 0:
    filtered_reviews_table = positive_reviews_table

# combine positive and filtered reviews into a single table
combined_reviews_table = pd.concat([filtered_reviews_table, positive_reviews_table])
combined_reviews_table = combined_reviews_table.drop(['Sentiment', 'Category'], axis=1)

# save the combined reviews table (positive and filtered) to a text file
combined_reviews_table_file_name = f'{product_name}_combined_reviews_table.txt'
combined_reviews_table.to_csv(combined_reviews_table_file_name, index=False, sep='\t')
print(f"Combined Reviews table saved to '{combined_reviews_table_file_name}'")

# plot the bar chart only if there are negative or neutral comments
if len(filtered_reviews_table) > 0:
    category_counts = filtered_reviews_table['Category'].value_counts()

    # colour for each category bar
    category_colors = {
        'Delivery Time': '#C1E3FA',
        'Product Quality': '#FFDDA7',
        'Wrong Product': '#B0F5E1',
        'Missing Product': '#FFC3BD',
        'Other': '#9DA9B5'
    }

    # plot bar chart
    plt.figure(figsize=(13, 8))
    ax = category_counts.plot(kind='bar', color=[category_colors[category] for category in category_counts.index])
  
    # set title
    plt.title(f'Category Counts of Ratings {selected_ratings} for {product_name}')
    plt.xlabel('Review Comment Category')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=360, ha='center')
    plt.ylabel('Count')

    # display the count values on top of each bar
    for i, count in enumerate(category_counts):
        plt.text(i, count + 0.1, str(count), ha='center')

    # create legend handles and labels
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors[category], markersize=10) for category in category_counts.index]
    legend_labels = category_counts.index

    # display the legend outside the plot
    ax.legend(legend_handles, legend_labels, title='Category Counts', title_fontsize='12', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

# display the table only if there are positive comments
elif len(positive_reviews_table) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
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

    plt.title(f'Table of Positive Review Messages and Ratings {selected_ratings} of {product_name}', fontsize=16, fontweight='bold', x=0.5)

    plt.show()
    
# if there is no reviews at all for certain ratings
elif len(filtered_reviews_table) == 0 or len(positive_reviews_table) == 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')  
    
    num_rows = 1
    plt.title(f'Table of Review Messages and Ratings {selected_ratings} of {product_name}', fontsize=16, fontweight='bold', x=0.5)
    ax.text(0.5, 0.5, 'No comments for the selected ratings', ha='center', va='center', fontsize=16, color='black')
    plt.show()