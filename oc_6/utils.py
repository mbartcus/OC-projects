from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from contextlib import contextmanager

import re
from wordcloud import WordCloud, STOPWORDS
import time
import gc

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

from sklearn.base import is_classifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    fbeta_score,
    make_scorer
)

import pickle


from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
import heapq

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Function to calculate missing values by column# Funct
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def plot_nan_in_pourcent_from_data(df):
    # verifié les valeurs manquants en affichant le pourcentage
    dd = df.isna().mean().sort_values(ascending=True)*100
    #plt.figure(figsize=(15, 10));
    fig = plt.figure(figsize=(15, 10));
    axes = sns.barplot(x=dd.values, y=dd.index, data=dd);
    axes.set_xticks([]);
    axes.set_yticks([0, 20, 40, 60, 80, 100]);
    plt.title('NaN Values on entire dataset',fontsize=25);
    plt.xlabel('Variables',fontsize=15);
    plt.ylabel('% of NaN values',fontsize=15);
    del dd;

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

def plot_words(df, col, height = 15, wieght = 10):
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
    fig = plt.figure( figsize=(height, wieght), frameon=False )
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.box(False)
    plt.imshow(wordcloud)
    return fig


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

    fig, axes = plt.subplots(nrows = round(len(columns)/2+.1), ncols = 2, figsize=(30, 15), constrained_layout=True);
    on_col=0
    on_line=0
    for index, col in enumerate(columns):
        if (dt == DensityTypes.Density):
            sns.distplot(a = df[col], label=col, ax=axes[on_line, on_col%2], bins=100);
        elif (dt == DensityTypes.Boxplot):
            sns.boxplot(data = df[col], ax=axes[on_line, on_col%2], orient = 'h');
        #sns.histplot(df[col], kde=True, stat="density", linewidth=0, ax=axes[on_line, on_col%2])
        #axes[on_line, on_col%2].set_title('{0} distribution'.format(col,fontsize=25));
        axes[on_line, on_col%2].set_xlabel(col, fontsize=15);
        axes[on_line, on_col%2].set_ylabel('Density', fontsize=15);
        if on_col%2 == 1: on_line+=1
        on_col+=1;

    plt.show();

    
def plot_count_col(df, col, label_col=None, top=None, show_val = False, on_x=True):
    plt.figure(figsize=(15,8))
    
    if on_x:
        if top==None:
            ax = sns.barplot(y=df[col].value_counts(normalize=True), x=df[col].value_counts(normalize=True).index, data=df);
        else:
            ax = sns.barplot(y=df[col].value_counts(normalize=True)[:top], x=df[col].value_counts(normalize=True).index[:top], data=df);
    else:
        if top==None:
            ax = sns.barplot(x=df[col].value_counts(normalize=True), y=df[col].value_counts(normalize=True).index, data=df);
        else:
            ax = sns.barplot(x=df[col].value_counts(normalize=True)[:top], y=df[col].value_counts(normalize=True).index[:top], data=df);
    
    if label_col==None:
        label_col=col
    if show_val:
        show_values(ax, space=0.01)
    
    plt.title('Counting in {0}'.format(label_col), fontsize=20);
    if on_x:   
        plt.ylabel('% of {0}'.format(label_col), fontsize=15);
        plt.xlabel('{0}'.format(label_col), fontsize=15);
    else:
        plt.xlabel('% of {0}'.format(label_col), fontsize=15);
        plt.ylabel('{0}'.format(label_col), fontsize=15);
    plt.show();
                            
                            
                            
def plot_correlation(df, col =  np.nan):
    if np.isnan(col):
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
    else:
        '''
        correlation for one variable
        '''
        corr = df.corr()[col].sort_values()
        return corr


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




def show_values(axs, orient="v", space=.01):
    """
    # You can use the following function to display the values on a seaborn barplot:
    inspired from : https://www.statology.org/seaborn-barplot-show-values/
    """
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center", fontsize=15)
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left", fontsize=15)

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def encode_categorical_variables(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    le = LabelEncoder()
    le_count = 0


    for col in categorical_columns:
        if len(list(df[col].unique())) <= 2:
            # Train on the training data
            le.fit(df[col])
            # Transform both training and testing data
            df[col] = le.transform(df[col])


    # one-hot encoding of categorical variables
    # Use dummies if > 2 values in the categorical variable
    df = pd.get_dummies(df, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns


def get_correlations(df, var):
    # gets the correlations between all the variables sorting by the variable var
    return df.corr().abs().sort_values(var, ascending=False, axis=0).sort_values(var, ascending=False, axis=1)


def high_decorelation(df_correlation, var, corr_min_threshold = 0.01):
    highly_decorrelation = {}
    for col in df_correlation.columns:
        if col != var and (pd.isnull(df_correlation[col][var]) or abs(df_correlation[col][var]) < corr_min_threshold):
                highly_decorrelation[col] = df_correlation[col][var]

    return highly_decorrelation


def high_correlation(df_correlation, corr_max_threshold = 0.9):
    highly_correlated = pd.DataFrame(columns=["pair", "correlation"])
    for i in range(len(df_correlation.columns)):
        for j in range(i + 1, len(df_correlation.columns)):
            if df_correlation.iloc[i, j] > corr_max_threshold:
                # variables are highly correlated
                if df_correlation.iloc[0, i] > df_correlation.iloc[0, j]:
                    # first variable is more correlated with the variable => we want to keep it
                    keep_index = i
                    drop_index = j
                else:
                    keep_index = j
                    drop_index = i

                highly_correlated.loc[df_correlation.columns[drop_index]] = {
                    "pair": df_correlation.columns[keep_index],
                    "correlation": df_correlation.iloc[i, j],
                }

    highly_correlated.sort_values(by="correlation", ascending=False)
    return highly_correlated

def remove_columns_regarding_correlation(df, df_corr, var='TARGET', hdc_make=True, hc_make=True):
    if hdc_make:
        hdc = high_decorelation(df_corr, var='TARGET', corr_min_threshold = 0.01).keys()
        cols_to_delete = list(hdc)
        df = df.drop(columns=cols_to_delete,
                        #inplace=True,
                        errors="ignore"
                     )
    if hc_make:
        hc = high_correlation(df_corr, corr_max_threshold = 0.9)
        df = df.drop(columns=hc.index,
                      #inplace=True,
                      errors="ignore",
        )
    return df


def process_encode_and_joining(df_train, df_previous_application, df_bureau):
    with timer("Encoding datasets..."):
        df_train, new_columns = encode_categorical_variables(df_train, nan_as_category = True)
        df_previous_application, new_columns_application = encode_categorical_variables(df_previous_application, nan_as_category = True)
        df_bureau, new_columns_bureau = encode_categorical_variables(df_bureau, nan_as_category = True)

    with timer("Processing joining dataframes and creation of features...."):
        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['sum']
            }
        cat_aggregations = {}
        for cat in new_columns_bureau: cat_aggregations[cat] = ['mean']
        bureau_agg = df_bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }

        cat_aggregations = {}
        for cat in new_columns_application: cat_aggregations[cat] = ['mean']
        prev_agg = df_previous_application.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

        df_train = df_train.join(bureau_agg, how='left', on='SK_ID_CURR')
        df_train = df_train.join(prev_agg, how='left', on='SK_ID_CURR')

        df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        del df_previous_application, df_bureau, bureau_agg, prev_agg
        gc.collect()

    return df_train





def get_best_classifier(X_train, y_train, X_test, y_test, estimator, params = {}, beta_param=2, verbose=0):
    """Runs cross validation to find the best estimator hyper-parameters.
    Args:
        X_train (pd.DataFrame): training data
        y_train (pd.Series): training labels
        X_test (pd.DataFrame): testing data
        y_test (pd.Series): testing labels
        estimator (ClassifierMixin): Classifier
        params (dict[str, list[Union[str, float, int, bool]]], optional):
            hyper-parameters range for cross validation. Defaults to {}.
    Raises:
        ValueError: Error if estimator is not a classifier
    Returns:
        dict[str, Any]: Classifier optimization results.
    """
    if not is_classifier(estimator):
        logging.error(f"{estimator} is not a classifier.")
        raise ValueError(f"{estimator} is not a classifier.")

    ftwo_scorer = make_scorer(fbeta_score, beta=beta_param)

    clf = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,

        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring=ftwo_scorer, # "f1",
        verbose=verbose,
        n_jobs=-1
    ).fit(
        X=X_train,
        y=y_train,
    )

    t0 = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t0

    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_pred_proba = clf.decision_function(X_test)
    else:
        y_pred_proba = y_pred

    return {
        "classifier": clf,
        "model": clf.best_estimator_,
        "params": clf.best_params_,
        "score": clf.best_score_,
        "predict_time": predict_time,
        "cv_results_": clf.cv_results_,
        "best_index_": clf.best_index_,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fbeta": fbeta_score(y_test, y_pred, beta=beta_param),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "precision_recall_curve": precision_recall_curve(y_test, y_pred_proba),
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        "roc_curve": roc_curve(y_test, y_pred_proba),
    }





def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                         ax=None):
    '''
    Thanks to https://github.com/DTrimarchi10/confusion_matrix

    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    if ax is None:
        plt.figure(figsize=figsize)

    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, ax=ax, annot_kws={"size": 18})
    if ax is None:
        if xyplotlabels:
            plt.ylabel('True label', size=15)
            plt.xlabel('Predicted label' + stats_text, size=15)
        else:
            plt.xlabel(stats_text, size=15)

        if title:
            plt.title(title, size=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    else:
        if xyplotlabels:
            ax.set_ylabel('True label', size=15)
            ax.set_xlabel('Predicted label' + stats_text, size=15)
        else:
            ax.set_xlabel(stats_text, size=15)

        if title:
            ax.set_title(title, size=20)
            
        ax.set_xticklabels(ax.get_xticks(), size=15)
        ax.set_yticklabels(ax.get_yticks(), size=15)



def plot_result_stats(model_res, label=None, title_fig=None):
    #"f1": f1_score(y_test, y_pred),
    #"accuracy": accuracy_score(y_test, y_pred),
    #"precision": precision_score(y_test, y_pred),
    #"recall": recall_score(y_test, y_pred),
    #"average_precision": average_precision_score(y_test, y_pred_proba),
    #"precision_recall_curve": precision_recall_curve(y_test, y_pred_proba),
    #"roc_auc_score": roc_auc_score(y_test, y_pred_proba),
    #"roc_curve": roc_curve(y_test, y_pred_proba),
    cf_matrix = model_res['confusion_matrix']
    fpr, tpr, _ = model_res['roc_curve'];
    lr_precision, lr_recall, _ = model_res['precision_recall_curve'];


    labels = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['0', '1']


    fig, axes = plt.subplots(1, 3, figsize=(30, 10));
    fig.suptitle(title_fig, fontsize=30)

    make_confusion_matrix(cf_matrix,
                          group_names=labels,
                          categories=categories,
                          figsize = (15,10),
                          cmap = 'inferno',
                          ax=axes[0]);

    sns.lineplot(fpr, tpr, label=label, linewidth = 1.5, ax=axes[1]);
    axes[1].set_title('ROC curve', fontsize=25)
    axes[1].set_xlabel('False Positive Rate', fontsize=20);
    axes[1].set_ylabel('True Positive Rate', fontsize=20);

    sns.lineplot(lr_recall, lr_precision, label=label, linewidth = 1.5, ax=axes[2]);
    axes[2].set_title('Precision Recall Curve', fontsize=25)
    axes[2].set_xlabel('Recall', fontsize=20);
    axes[2].set_ylabel('Precision', fontsize=20);



def plot_varimportance(model_res, cols, cols_nr):
    if hasattr(model_res['model'], 'coef_'):
        feature_importance = model_res['model'].coef_[0]
    elif hasattr(model_res['model'], 'feature_importances_'):
        feature_importance = model_res['model'].feature_importances_
    else:
        raise ValueError('The model can not show the feature importance')

    top_coefficients = pd.Series(
        feature_importance,
        cols,
    ).map(abs).sort_values(ascending=False).head(cols_nr)
    top_coefficients = pd.DataFrame(top_coefficients).reset_index()
    top_coefficients.columns=['cols', 'coef']
    plt.figure(figsize=(15,8))
    sns.barplot(y = 'cols', x = 'coef', data = top_coefficients)
    plt.title('Top {0} important variables'.format(cols_nr), fontsize=20);
    plt.xlabel('Coefficient', fontsize=15);
    plt.ylabel('Columns', fontsize=15);
    plt.show();


## Feature Selection
def feature_selection(data, k):
    X = data.drop(["TARGET"], axis=1)
    y = data["TARGET"]

    column_names = X.columns  # Here you should use your dataframe's column names

    fs = SelectKBest(f_classif, k=k)

    # Applying feature selection
    X_selected = fs.fit_transform(X, y)

    # Find top features
    # I create a list like [[ColumnName1, Score1] , [ColumnName2, Score2], ...]
    # Then I sort in descending order on the score
    #top_features = sorted(zip(column_names, fs.scores_), key=lambda x: x[1], reverse=True)
    #print(top_features[:k])
    dict_scores = dict(sorted(zip(column_names, fs.scores_) ))

    cols = heapq.nlargest(k, dict_scores, key=dict_scores.get)

    cols.append('TARGET')

    return data[cols]


def save_model(fname = 'finalized_model.sav'):
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))



def load_model(fname = 'finalized_model.sav'):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    return result
