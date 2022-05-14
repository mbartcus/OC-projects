import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import *
from sklearn.linear_model import LinearRegression



st.set_page_config(layout="wide")


# header of web application
st.title('Choose your food')
st.header('Using the data collected from OpenFoodFacts')



df = pd.read_csv('/Users/marius/Documents/GitHub/OC-projects/oc_3/data/df_app.csv')


options_categories_food = st.sidebar.multiselect(
     'Choose foods categories you prefer',
     df.my_categoty.unique().tolist(),
     ['sweet', 'melange'],
     key = 'categories_food')

options_nutrition_grade = st.sidebar.multiselect(
     'Choose nutrition grade',
     df.nutrition_grade_fr.unique().tolist(),
     ['a', 'b'],
     key = 'nutrition_grade')



option_var_100g = st.sidebar.selectbox(
     'Choose the nutrition you prefer',
     (get_cols_100g(df)),
     key = 'var1_100g')

option_var2_100g = st.sidebar.selectbox(
     'Choose the nutrition you prefer',
     (get_cols_100g(df)),
     key = 'var2_100g')

df_selected = df[df['my_categoty'].isin(options_categories_food) & df.nutrition_grade_fr.isin(options_nutrition_grade) ]

df_categories =  get_pandas_catVar_numVar(df_selected, catVar='my_categoty', numVar=option_var_100g)


col1, col2= st.columns(2)

with col1:
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(x=option_var_100g, y="my_categoty", data=df_selected, orient = 'h', showfliers = False);
    plt.title("Categories Food distribution over {0}".format(option_var_100g), fontsize=20)
    plt.xlabel('{0}'.format('Food categories'), fontsize=15);
    plt.ylabel('{0}'.format(option_var_100g), fontsize=15);
    st.pyplot(fig)

with col2:
    fig=plt.figure(figsize=(10,4));
    sns.scatterplot(data=df_selected, x=option_var_100g, y=option_var2_100g, hue="nutrition_grade_fr")
    plt.title('Interaction of {0} on {1}'.format(option_var_100g, option_var2_100g), fontsize=20)
    plt.xlabel('{0}'.format(option_var_100g), fontsize=15);
    plt.ylabel('{0}'.format(option_var2_100g), fontsize=15);
    st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    fig=plt.figure(figsize=(10,4));
    sns.scatterplot(data=df_selected, x=option_var_100g, y=option_var2_100g, hue="my_categoty")
    plt.title('Interaction of {0} on {1}'.format(option_var_100g, option_var2_100g), fontsize=20)
    plt.xlabel('{0}'.format(option_var_100g), fontsize=15);
    plt.ylabel('{0}'.format(option_var2_100g), fontsize=15);
    st.pyplot(fig)



with col4:
    x=df_selected['nutrition-score-fr_100g']
    y=df_selected['nutrition-score-uk_100g']

    fig = plt.figure(figsize=(10,4));
    sns.scatterplot(x,
                    y,
                    hue = df_selected['my_categoty'],
                    legend='full',
                    s=30);

    plt.title('Nutri score UK vs FR', fontsize=20);
    plt.xlabel('nutri score fr 100g', fontsize=15);
    plt.ylabel('nutri score uk 100g', fontsize=15);


    #linear regression
    x = np.array(x).reshape(-1, 1);
    y = np.array(y).reshape(-1, 1);

    reg = LinearRegression();
    model = reg.fit(x, y);
    plt.plot(x, model.predict(x),color='k');
    st.pyplot(fig)
