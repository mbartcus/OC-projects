import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import *


st.markdown(
    """
    <style>
    .reportview-container {
        background: url("url_goes_here")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True
)


# header of web application
st.title('Choose your food')
st.header('Using the data collected from OpenFoodFacts')



df = pd.read_csv('/Users/marius/Documents/GitHub/OC-projects/oc_3/data/df_app.csv')


options_categories_food = st.sidebar.multiselect(
     'Choose foods categories you prefer',
     df.my_categoty.unique().tolist(),
     ['sweet', 'melange'])

st.write('You selected:', options_categories_food)





option_var_100g = st.sidebar.selectbox(
     'Choose the nutrition you prefer',
     (get_cols_100g(df)))

st.write('You selected:', option_var_100g)



df_categories =  get_pandas_catVar_numVar(df, catVar='my_categoty', numVar=option_var_100g)

st.dataframe(df_categories.head())

fig = plt.figure(figsize=(10, 4))
sns.boxplot(x=option_var_100g, y="my_categoty", data=df[df['my_categoty'].isin(options_categories_food)], orient = 'h', showfliers = False);
st.pyplot(fig)
