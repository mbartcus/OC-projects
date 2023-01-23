import streamlit as st
import requests
import io
import numpy as np
import zlib
import matplotlib.image
import os
import my_classes as mc

# ## HELPERS

def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

st.title('Image segmentation')
st.subheader('Self-driven car')

LOCAL =  'http://127.0.0.1:5000/api'
LOCAL_DOCKER = 'http://172.17.0.2:5000/api'
SERVER_GOOGLERUN = 'https://autcar-img-dyu56zxquq-uc.a.run.app/api'

APP = SERVER_GOOGLERUN

x_test_dir = 'images.txt'
y_test_dir = 'mask.txt'

dataset = mc.Dataset(
    x_test_dir,
    y_test_dir
)

with st.form(key='recommandation_form', clear_on_submit=True):
    option = st.selectbox(
    'Select the image id',
    (0, 1, 2))



    submit_button = st.form_submit_button('Submit')

if submit_button:
    with st.spinner('Wait for it...'):

        st.write('You selected the image id:', option)
        image_origin, mask_origin = dataset[option]

        resp = requests.get(APP, params={"image_id": option}, headers={'Content-Type': 'application/octet-stream'})

        pr_mask = uncompress_nparr(resp.content) # predicted mask

        col1, col2, col3 = st.columns(3)

        with col1:
           st.image(image_origin.squeeze(), caption='Original image');

        with col2:
           matplotlib.image.imsave('or_mask.png', np.argmax(mask_origin, 2))
           st.image('or_mask.png', caption='Original mask');
           os.remove('or_mask.png')

        with col3:
           matplotlib.image.imsave('mask.png', pr_mask)
           st.image('mask.png', caption='Predicted mask');
           os.remove('mask.png')
