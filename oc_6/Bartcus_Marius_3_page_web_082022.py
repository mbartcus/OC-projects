import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score




st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

df_nlp_review = pd.read_pickle("/Users/bartcus/Documents/GitHub/OC-projects/oc_6/data/processed/df_nlp_review.pkl.gz")
X_shift_std = pd.read_pickle("/Users/bartcus/Documents/GitHub/OC-projects/oc_6/data/processed/X_shift_std.pkl.gz")
X_orb_std = pd.read_pickle("/Users/bartcus/Documents/GitHub/OC-projects/oc_6/data/processed/X_orb_std.pkl.gz")
df_photos_features_vgg = pd.read_pickle("/Users/bartcus/Documents/GitHub/OC-projects/oc_6/data/processed/df_photos_features_vgg.pkl.gz")
labels_photos = pd.read_pickle("/Users/bartcus/Documents/GitHub/OC-projects/oc_6/data/processed/labels_photos.pkl.gz")

with st.sidebar.container():
    n_topics = st.selectbox(
         'Choose number of topics',
         ([5, 10]),
         key = 'nr_topics')

    no_top_words = st.selectbox(
         'Choose the number of top words',
         ([3, 5, 10]),
         key = 'no_top_words')

    explained_by_pca = st.slider(
     'Select data explained by pca',
     0, 100, 40)

    features_option = st.selectbox(
     'Select the type of features to use',
     ('ORB', 'SHIFT', 'VGG'))

    vizualize_option = st.selectbox(
     'Select the type of vizualization',
     ('PCA', 'TSNE'))


lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='batch',
            #learning_offset=50.,
            random_state=1)

# Fitter sur les données
lda.fit(df_nlp_review)

if features_option == 'ORB':
    X = X_orb_std
elif features_option == 'SHIFT':
    X = X_shift_std
elif features_option == 'VGG':
    X = np.array([vgg_feature for vgg_feature in df_photos_features_vgg.vgg_features])
    sc = StandardScaler().fit(X) #MinMaxScaler().fit(vgg_features) #  # RobustScaler().fit(vgg_features)
    X = sc.transform(X)
    X = pd.DataFrame(X, columns = ['vgg_{0}'.format(i) for i in range(X.shape[1])])

from sklearn.decomposition import PCA
pca_features_ratio = {}

is_not_selected=True
pca = PCA(random_state=10)

#st.dataframe(X)

pca.fit(X)

for k in range(min(X.shape[0],X.shape[1])):
    s = np.cumsum(pca.explained_variance_ratio_)[k]
    pca_features_ratio[k] = s
    if ((s>=(explained_by_pca/100)) & is_not_selected):
        n_comp = k
        is_not_selected = False




from sklearn import decomposition
pca = decomposition.PCA(n_components=n_comp, random_state=10)
pca.fit(X)

X_projected = pca.transform(X)
X_projected = pd.DataFrame(X_projected, columns = ['F{0}'.format(i) for i in range(n_comp)])


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=1)
T = tsne.fit_transform(X_projected)
T = pd.DataFrame(T, columns=['T1', 'T2'])




# header of web application

st.title('Avis Restau')
st.header('Topic analysis')


f_topics = plot_top_words(
    lda, list(df_nlp_review.columns), no_top_words, n_topics, "Topics in LDA model" # (Frobenius norm)
)
st.pyplot(f_topics)

st.header('Photos clustering')
st.text('Number of PCA components is {0}'.format(n_comp))

col1, col2= st.columns(2)
with col1:
    if n_comp>=2:
        if vizualize_option=='PCA':
            N = 3
            to_plot = X_projected.iloc[: , :N]
            to_plot.loc[:,'label'] = labels_photos
            #to_plot.loc[:,'label'] = to_plot.label.astype(np.float)
            #st.dataframe(to_plot)
            fig=plt.figure(figsize=(10,4));
            sns.scatterplot(data=to_plot, x='F0', y='F1', hue='label', palette=sns.color_palette("Set1", to_plot.label.nunique()))
            st.pyplot(fig)
            del to_plot
        elif vizualize_option=='TSNE':
            T.loc[:,'label'] = labels_photos
            #T.loc[:,'label'] = T.label.astype(np.float)
            fig=plt.figure(figsize=(10,4));
            sns.scatterplot(data=T, x='T1', y='T2', hue='label', palette=sns.color_palette("Set1", T.label.nunique()))
            st.pyplot(fig)
            T = T.drop(columns=['label'])

with col2:
    if vizualize_option=='PCA':
        kmeans = KMeans(n_clusters=5, random_state=1).fit(X_projected)
        N = 3
        to_plot = X_projected.iloc[: , :N]
        to_plot.loc[:,'label'] = kmeans.labels_
        fig=plt.figure(figsize=(10,4));
        sns.scatterplot(data=to_plot, x='F0', y='F1', hue='label', palette=sns.color_palette("Set1", to_plot.label.nunique()))
        st.pyplot(fig)
        st.text('ARI = {0}'.format(adjusted_rand_score(labels_photos, kmeans.labels_)))
        del to_plot
    elif vizualize_option=='TSNE':
        kmeans = KMeans(n_clusters=5, random_state=1).fit(T)
        T.loc[:,'label'] = kmeans.labels_
        fig=plt.figure(figsize=(10,4));
        sns.scatterplot(data=T, x='T1', y='T2', hue='label', palette=sns.color_palette("Set1", T.label.nunique()))
        st.pyplot(fig)
        T = T.drop(columns=['label'])
        st.text('ARI = {0}'.format(adjusted_rand_score(labels_photos, kmeans.labels_)))

ari_metric  = {'ari_shift_pca': 0.12443246667924797,
 'ari_shift_tsne': 0.10121979928279368,
 'ari_orb_pca': 0.0441680585783754,
 'ari_orb_tsne': 0.042284790160085434,
 'ari_vgg_tsne': 0.6369244758074947,
 'ari_vgg_pca': 0.49708985447392773}
ari_metric = pd.DataFrame.from_dict(ari_metric, orient='index').reset_index()
ari_metric.columns=['Method', 'ARI']

fig_ari=plt.figure(figsize=(10,4));
sns.barplot(data=ari_metric, y="ARI", x='Method')
st.pyplot(fig_ari)
