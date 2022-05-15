import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import *
from sklearn.linear_model import LinearRegression



st.set_page_config(layout="wide")

df = pd.read_csv('/Users/marius/Documents/GitHub/OC-projects/oc_3/data/df_app.csv')

with st.sidebar.container():
    options_categories_food = st.multiselect(
         'Choose foods categories you prefer',
         df.my_categoty.unique().tolist(),
         ['sweet', 'melange', 'fats', 'beverage'],
         key = 'categories_food')

    options_nutrition_grade = st.multiselect(
         'Choose nutrition grade',
         df.nutrition_grade_fr.unique().tolist(),
         ['a', 'b'],
         key = 'nutrition_grade')



    option_var_100g = st.selectbox(
         'Choose the nutrition you prefer',
         (get_cols_100g(df)),
         key = 'var1_100g')

    option_var2_100g = st.selectbox(
         'Choose the nutrition you prefer',
         (get_cols_100g(df)),
         key = 'var2_100g')




df_selected = df[df['my_categoty'].isin(options_categories_food) &
                df.nutrition_grade_fr.isin(options_nutrition_grade)]



energy_max = int(np.round(df_selected['energy_100g'].max()))
additives_max =  int(np.round(df_selected['additives_n'].max()))

with st.sidebar.expander("See more ..."):
    energy_selected = st.slider(
         'Select energy',
         0, energy_max, energy_max)

    sugar_selected = st.slider(
         'Select sugar',
         0.0, 100.0, (25.0, 75.0))

    salt_selected = st.slider(
         'Select salt',
         0.0, 100.0, (25.0, 75.0))

    additives_select = st.slider(
         'Select additives',
         0, additives_max, additives_max)

#st.dataframe(df_selected[df_selected.sugar_100g >= sugar_selected[0]])


df_food = df_selected[
                (df_selected.energy_100g <= energy_selected) &
                (df_selected.sugars_100g >= sugar_selected[0]) &
                (df_selected.sugars_100g <= sugar_selected[1]) &
                (df_selected.salt_100g >= salt_selected[0]) &
                (df_selected.salt_100g <= salt_selected[1]) &
                (df_selected.additives_n <= additives_select)
                ]

# header of web application

st.title('Choose your food')
st.header('Using the data collected from OpenFoodFacts')


'''
#### Each of us has it's own preferences of food. Someone are interested in meat/fish, other are looking for cheese, some are looking for some drinks.
But the hole variety of theese foods with lots of nutrition facts makes difficult to make a good selection.
 - We can have an active life so the need of energy foods is higher then if we have an sitting proffesion.
 - Some desises can impose/limit to eat some foods (ex: sugar/salt/additives limitter). Sugars higher blood pressure, inflammation, weight gain, diabetes, and fatty liver disease — are all linked to an increased risk for heart attack and stroke.

In my data analyse project I propose to analyse the nutrition food facts, and porpose an application to help the user understand the food nutrition facts, and make a decision
on the desired food.
'''

fig_words = plot_words(df_selected, 'my_categoty')
st.pyplot(fig_words)

df_categories =  get_pandas_catVar_numVar(df_selected, catVar='my_categoty', numVar=option_var_100g)




col1, col2= st.columns(2)

with col1:
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(x=option_var_100g, y="my_categoty", data=df_selected, orient = 'h', showfliers = False);
    plt.title("Categories Food distribution over {0}".format(option_var_100g), fontsize=20)
    plt.xlabel('{0}'.format(option_var_100g), fontsize=15);
    plt.ylabel('{0}'.format('Food categories'), fontsize=15);
    st.pyplot(fig)

with col2:
    fig=plt.figure(figsize=(10,4));
    sns.scatterplot(data=df_selected, x=option_var_100g, y=option_var2_100g, hue="nutrition_grade_fr")
    plt.title('{0} / {1} by nutrition grade'.format(option_var_100g, option_var2_100g), fontsize=20)
    plt.xlabel('{0}'.format(option_var_100g), fontsize=15);
    plt.ylabel('{0}'.format(option_var2_100g), fontsize=15);
    st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    fig=plt.figure(figsize=(10,4));
    sns.scatterplot(data=df_selected, x=option_var_100g, y=option_var2_100g, hue="my_categoty")
    plt.title('{0} / {1} by category'.format(option_var_100g, option_var2_100g), fontsize=20)
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


# propose 3 foods for each category having energy sugar and salt selected

with st.expander("Proposals"):
    st.dataframe(df_food.head())

#    for index, row in df_food.iterrows():
#        st.write('Product: {0} from {1} with {2} additives, {3} sugar and {4} salt in 100g'.format(row["product_name"], row["countries_fr"], int(row['additives_n']), row['sugars_100g'], row['sugars_100g']))
