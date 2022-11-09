import streamlit as st
import requests




st.title('Books recommandation')
st.subheader('Air Paradis')

with st.form(key='tweet_form', clear_on_submit=True):
    my_tweet = st.text_input('Tweet something here:')

    submit_button = st.form_submit_button('Submit')

if submit_button:
    with st.spinner('Wait for it...'):
        st.info(f'Your tweet is :  {my_tweet}')

        result = requests.get("https://sentimentanalyseapi.herokuapp.com/api", params={"my_tweet": my_tweet}).json()

        sentiment = result['sentiment']
        prob = float(result['prob'])

        st.success('This is a {0}% {1} tweet'.format(round(prob*100, 2), sentiment))
