import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
import os

import requests


#response = requests.get("http://127.0.0.1:5000/")
#print(response.json())



st.title('Sentiment analyse')
st.subheader('Air Paradis')

with st.form(key='tweet_form', clear_on_submit=True):
    my_tweet = st.text_input('Tweet something here:')

    submit_button = st.form_submit_button('Submit')

if submit_button:
    with st.spinner('Wait for it...'):
        st.info(f'Your tweet is :  {my_tweet}')#, icon='ℹ️')

        result = requests.get("http://127.0.0.1:5000/api/tweet/", params={"my_tweet": my_tweet}).json()

        pos = float(result['Positive'])
        neg = float(result['Negative'])

        if pos>neg:
            st.success('This is a {:.2f}% positive tweet :thumbsup:'.format(pos*100))
        else:
            st.error('This is a {:.2f}% negative tweet :thumbsdown:'.format(neg*100))
