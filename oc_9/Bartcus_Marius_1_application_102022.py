import streamlit as st
import requests

st.title('Books recommandation')
st.subheader('Collaborative Recommandation System')

with st.form(key='recommandation_form', clear_on_submit=True):
    user_id = st.text_input('Select user')
    submit_button = st.form_submit_button('Submit')

if submit_button:
    with st.spinner('Wait for it...'):
        st.info(f'Your user is :  {user_id}')
        article_score = requests.get("https://oc9.azurewebsites.net/api/HttpTriggerRecommand", params={"clientId":"blobs_extension", "user_id": user_id}).json()
        st.text(type(article_score))
        st.json(article_score)
