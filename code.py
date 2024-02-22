import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Zomato Bangalore Restaurants")

df = pd.read_csv('zomato.csv')

st.subheader("About the Dataset")
st.write("The first five rows of the dataset is as follows:")
st.write(df.head())

st.write("The statistical summary of the dataset is as follows:")
st.write(df.describe())

st.write(f"The dataset have {df.shape[0]} rows and {df.shape[1]} columns.")

st.subheader("Data Cleaning")

st.write("Calculating percentage of null values so that we can deal with them accordingly")
df.isna().sum().sort_values(ascending=False)*100/len(df)

st.write("Removing columns that are having a greater percentage of null values and the ones that are not useful in this Data Analysis. The updated dataset is as follows:")
df.drop(["Unnamed: 0","address","phone","menu_item","dish_liked"], axis=1, inplace=True)
st.write(df.head())

st.subheader("Dropping Duplicates")
df.drop_duplicates(inplace=True)
st.write(f"The number of rows after dropping duplicated ones are {df.shape[0]}")

st.subheader("Cleaning Rate Column")
st.write("The rate column consists of the followng values")
st.write(df['rate'].unique())
st.write("We can see that in the rate column all the values are in fractions and some of them are in string format. I want the rating to be a simple float value like 3.1,4.1 e.t.c. So I am replacing all string values with null values as we donot have a data for that and converting all fractions to floats. After converting string to null values we will fill those null values with the mean.")

def cleaning_rate(n):
    if (n == 'NEW' or n == '-'):
        return np.nan
    else:
        n = str(n).split('/')
        n = n[0]
        return float(n)
    
df['rate'] = df['rate'].apply(cleaning_rate)
st.write(df['rate'].head())
df['rate'].fillna(df['rate'].mean(),inplace = True)

def handlecomma(n):
    n = str(n)
    if ',' in n:
        n = n.replace(',','')
        return float(n)
    else:
        return float(n)
    
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].apply(handlecomma)

st.subheader("Dropping null values")
st.write('Now dropping all the rows with null values.')
df.dropna(inplace=True)
st.write(df.isnull().sum())

st.write('We can see that there are two columns that are providing us with the location of the restaurants. So we will keep one and drop the other')
df.drop(['listed_in(city)'],axis=1,inplace=True)
st.write(df.columns)

st.subheader("Cleaning Restaurant Type Column")
st.write('The value count of the column is as follows:')
st.write(df['rest_type'].value_counts())
st.write('We can see that this column have so many unique values but there are few values which are containing very less number of restaurants.So I will make a cluster of all these values which are having less number of restaurants and name them as Other. The maximum value count is 19000 approx so all the values which have a count of less than 1000 will be included in the clustered value. The values having a count of less than 1000 that will be included in the cluster are as follows:')
rest_type = df['rest_type'].value_counts()
rest_type_less_than_1000 = rest_type[rest_type<1000]
st.write(rest_type_less_than_1000)

def deal_with_rest_type(n):
    if n in rest_type_less_than_1000:
        return 'Others'
    else:
        return n
    
st.write("The updated value count of the column is as follows")    
df['rest_type'] = df['rest_type'].apply(deal_with_rest_type)
st.write(df['rest_type'].value_counts())

st.subheader('Cleaning Location Column')
st.write('The value counts of the column is as follows')
st.write(df['location'].value_counts())
st.write('Here in this column we can also see that there are values which are containing very less number of restaurants. So again a cluster named as Others will be made for this column as well. The values that will form a cluster are as follows:')
location = df['location'].value_counts()
location_less_than_300 = location[location<300]
st.write(location_less_than_300)

def deal_with_location(n):
    if n in location_less_than_300:
        return 'Others'
    else:
        return n
    
st.write("The updated value count of the column is as follows")    
df['location'] = df['location'].apply(deal_with_location)
st.write(df['location'].value_counts())

st.subheader('Cleaning Cuisines Column')
st.write('The value counts of the column is as follows')
st.write(df['cuisines'].value_counts())
st.write('Here in this column we can also see that there are values which are containing very less number of restaurants. So again a cluster named as Others will be made for this column as well. The values that will form a cluster are as follows:')
cuisines = df['cuisines'].value_counts()
cuisines_less_than_100 = cuisines[cuisines<100]
st.write(cuisines_less_than_100)

def deal_with_cuisines(n):
    if n in cuisines_less_than_100:
        return 'Others'
    else:
        return n
    
st.write("The updated value count of the column is as follows")    
df['cuisines'] = df['cuisines'].apply(deal_with_cuisines)
st.write(df['cuisines'].value_counts())

st.subheader("Exploratory Data Analysis")
st.write("The data is cleaned. Now let's move forward with the visualization.")

st.subheader("Count Plot for Locations")
loc_count  = plt.figure(figsize = (16,10))
sns.countplot(x = 'location', data = df)
plt.xticks(rotation=90)
plt.title('Location Count')
st.pyplot(loc_count)

st.write("Using this countplot we can conclude that the maximum number of restaurants are in the BTM area so it is not a good idea to open a restaurant in this very saturated area. ")

st.subheader("Visualizing Online Order")
st.write("Now we will analyze that how many restaurants are offering online order facility")
online_order = plt.figure(figsize=(6,6))
sns.countplot(x='online_order', data= df)
plt.title('Online Order Count')
st.pyplot(online_order)

st.subheader("Visualizing Book Table")
st.write("Now we will analyze that how many restaurants are offering booking or reserving a table facility")
book_table = plt.figure(figsize=(6,6))
sns.countplot(x='book_table', data= df)
plt.title('Book Table Count')
st.pyplot(book_table)

st.subheader("Visualizing Online Order vs Rate.")
oo_rate = plt.figure(figsize=(6,6))
sns.boxplot(x='online_order',y='rate', data= df)
st.pyplot(oo_rate)

st.subheader("Visualizing Book Table vs Rate.")
bt_rate = plt.figure(figsize=(6,6))
sns.boxplot(x='book_table',y='rate', data= df)
st.pyplot(bt_rate)

st.subheader("Heat Map.")
a = df.iloc[:,[3,4,8]]
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6,6))
sns.heatmap(a.corr(), annot=True, linewidths=0.1, fmt=".2f", ax = ax_heatmap)
st.pyplot(fig_heatmap)
st.write("A heatmap showing how different variables are correlated with eachother.")






































