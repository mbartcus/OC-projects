from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import re
from wordcloud import WordCloud, STOPWORDS

# afficher tout le dataset
def plot_data(df):
    '''
    Visualize the hole dataset in order to see the missing values
    Input:
        - df: DataFrame to plot
    '''
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.isna(), cbar = False)
    plt.title('Entire dataset',fontsize=25)
    plt.xlabel('Variables',fontsize=15)
    plt.ylabel('Observations',fontsize=15)

def get_pandas_catVar_numVar(df, catVar, numVar):
    modalities = list(df[catVar].value_counts().index)

    groupes= {}
    for m in modalities:
        if m not in groupes:
            groupes[m] = 0;
        groupes[m] = list(df[df[catVar] == m][numVar]);



    labels, data = [*zip(*groupes.items())];  # 'transpose' items to parallel key, value lists

    # or backwards compatable
    labels, data = groupes.keys(), groupes.values();

    data = pd.DataFrame(data);
    data = data.T;
    data.columns = labels;

    return data

def plot_words(df, col):
    #take not nan values
    df = df[~df[col].isna()]

    # Remove punctuation

    df[col].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase

    df[col].map(lambda x: x.lower())

    # Print out the first rows of papers
    df[col].head()


    # Join the different processed titles together.
    long_string = ','.join(list(df[col].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
    wordcloud.generate(long_string)# Visualize the word cloud
    plt.figure( figsize=(15,10) )
    plt.imshow(wordcloud)
    plt.show()

def compute_words_freq(df, var, sep=None):
    var_new = var + '_new'
    # compute function most common wolrd
    if sep is None:
        # make counting for each word
        df[var_new] = df[var].str.lower().str.replace('[^\w\s]','')
        df_freq = df[var_new].str.split(expand=True).stack().value_counts().reset_index()
    else:
        # make counting for each sequances of worlds separated by sep for example ','
        df[var_new] = df[var].str.lower()
        df_freq = df[var_new].str.split(sep,expand=True).stack().value_counts().reset_index()
    df_freq.columns = ['Word', 'Frequency']

    df.drop([var_new], inplace=True, axis=1)
    return df_freq

def print_columns(df):
    '''
    prints all columns
    '''
    i=0
    for col in df.columns:
        i+=1
        print('{0}:{1}'.format(i,col))

def get_cols_100g(df):
    cols = []
    for c in df.columns:
        if c.endswith('_100g'):
            cols.append(c)
    return cols


class DensityTypes(Enum):
    Density = 1,
    Boxplot = 2

def plot_density(df, columns = np.NaN, dt = DensityTypes.Density):
    '''
    Used to plot density for a dataframe columns
    Input:
        df - the dataframe
        columns - a list of columns if it is nan than all the columns in the dataframe are selected
    '''
    if columns is np.NaN:
        columns = df.select_dtypes(include=np.number).columns

    fig, axes = plt.subplots(round(len(columns)/2+.1), 2, figsize=(30, 15), constrained_layout=True);

    on_col=0
    on_line=0
    for index, col in enumerate(columns):
        if (dt == DensityTypes.Density):
            sns.distplot(df[col], label=col, ax=axes[on_line, on_col%2], bins=100);
        elif (dt == DensityTypes.Boxplot):
            sns.boxplot(df[col], ax=axes[on_line, on_col%2]);
        #sns.histplot(df[col], kde=True, stat="density", linewidth=0, ax=axes[on_line, on_col%2])
        #axes[on_line, on_col%2].set_title('{0} distribution'.format(col,fontsize=25));
        axes[on_line, on_col%2].set_xlabel(col, fontsize=15);
        axes[on_line, on_col%2].set_ylabel('Density', fontsize=15);
        if on_col%2 == 1: on_line+=1
        on_col+=1;
    plt.show();

def plot_correlation(df):
    corr = df.corr()
    # Fill redundant values: diagonal and upper half with NaNs
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan
    return (corr
     .style
     .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
     .highlight_null(null_color='#f1f1f1')  # Color NaNs grey
    )

def get_values_of_interest(df, col):
    '''
    Used to eliminate outliers
    Input:
        - df: DataFrame
        - col: the column to analyse

    computes
        min, max, med = Q2,
        Q1-first quartile, Q3 - third quartile
        IQ = Q3 - Q1
    returns Q1-1/5*IQ<(o1, o2)<Q1+1/5*IQ
    '''
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQ = Q3 - Q1
    o1 = max(0,Q1-1.5*IQ)
    o2 = Q3+1.5*IQ
    return (o1,o2)

def compute_energy(proteins, carbohydrates, fat):
    # 1g of fat is 39 kJ and 1g of carbohydrates or proteins is 17 kJ of energy .
    return 17*proteins + 17*carbohydrates + 39*fat


def eta_squared(df, var1, var2):
    """
    compute the correlation of a categorical variable and numerical variable
    Input:
        - df: the dataframe
        - x: categorical variable
        - y: numerical variable
    """
    X = df[var1]
    Y = df[var2]
    moyenne_y = Y.mean()
    classes = []
    for classe in X.unique():
        yi_classe = Y[X==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})

    SCT = sum([(yj-moyenne_y)**2 for yj in Y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
