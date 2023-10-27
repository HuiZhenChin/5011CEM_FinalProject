import pandas as pd 
import numpy as np 
import seaborn as sns 
import warnings 
import unidecode
import re
warnings.filterwarnings('ignore')

geolocation= pd.read_csv("olist_geolocation_dataset.csv")

print ("Geolocation")
geolocation.head()
geolocation.info()

print ("Unique: ")
geolocation.nunique()

print("Null Value: ")
geolocation.isnull().sum()

print("Null Value Percentage: ")
(geolocation.isnull().sum()/(len(geolocation)))*100

#---------------------------------------
# remove all the city symbols
# view how many city in dataset
unique_city = geolocation['geolocation_city'].unique()
print(unique_city)

geolocation['geolocation_city'] = geolocation['geolocation_city'].apply(lambda x: unidecode.unidecode(x))
geolocation['geolocation_city'] = geolocation['geolocation_city'].str.replace('[^\w\s]', ' ', regex=True)

   
# function to remove symbols and convert to lowercase
def clean_city_name(city_name):
    # Remove symbols, diacritics, and convert to lowercase
    cleaned_name = re.sub(r'[^A-Za-z0-9\s]', '', city_name).lower()
    return cleaned_name

# clean_city_name function to the 'geolocation_city' column
geolocation['geolocation_city'] = geolocation['geolocation_city'].apply(clean_city_name)

# display the cleaned city
print("Unique City in the Dataset:")
for city in geolocation['geolocation_city'].unique():
   print(city)