import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import *





# header of web application
st.title('Choose your food')
st.header('Using the data collected from OpenFoodFacts')



df = pd.read_csv('/Users/marius/Documents/GitHub/OC-projects/oc_3/data/df_app.csv')


options_categories_food = st.sidebar.multiselect(
     'Choose foods categories you prefer',
     df.my_categoty.unique().tolist(),
     ['sweet', 'melange'])





option_var_100g = st.sidebar.selectbox(
     'Choose the nutrition you prefer',
     (get_cols_100g(df)),
     key = 'var1_100g')

option_var2_100g = st.sidebar.selectbox(
     'Choose the nutrition you prefer',
     (get_cols_100g(df)),
     key = 'var2_100g')


df_categories =  get_pandas_catVar_numVar(df, catVar='my_categoty', numVar=option_var_100g)


fig = plt.figure(figsize=(10, 4))
sns.boxplot(x=option_var_100g, y="my_categoty", data=df[df['my_categoty'].isin(options_categories_food)], orient = 'h', showfliers = False);
st.pyplot(fig)



fig=plt.figure(figsize=(10,4));
sns.scatterplot(data=df, x=option_var_100g, y=option_var2_100g)
plt.title('Interaction of fat on energy', fontsize=20);
plt.xlabel('Fat', fontsize=15);
plt.ylabel('Energy', fontsize=15);
st.pyplot(fig)
