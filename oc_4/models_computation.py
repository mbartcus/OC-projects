import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as st

# File system manangement
import os
import gc
import time
import pickle

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import decomposition
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import model_selection

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


from functions import *
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))




# Preprocess bureau.csv and bureau_balance.csv
def modeling(X_train, y_train, X_test, y_test, mlclassifyalgs = ['dummy'], beta = 2):
    results = pd.DataFrame(columns=(
        'model',
        'params',
        'score',
        'predict_time',
        'cv_results_',
        'best_index_',
        'confusion_matrix',
        'f1',
        'fbeta',
        'accuracy',
        'precision',
        'recall',
        'average_precision',
        'precision_recall_curve',
        'roc_auc_score',
        'roc_curve',
    ), index = mlclassifyalgs)

    models = {}


    for alg in mlclassifyalgs:
        if alg == 'dummy':
            print('dummy....', flush=True)
            dummy_results = get_best_classifier(X_train=X_train,
                                                y_train=y_train.values.ravel(),
                                                X_test=X_test,
                                                y_test=y_test,
                                                estimator=DummyClassifier(random_state=10),
                                                params={'strategy': ['stratified', 'most_frequent', 'prior', 'uniform']}
                                               )
            results.loc['dummy',:]=dummy_results
            models['dummy']=dummy_results
        elif alg == 'logreg':
            print('Logistic regression....', flush=True)
            logistic_results = get_best_classifier(X_train=X_train,
                                                   y_train=y_train.values.ravel(),
                                                   X_test=X_test,
                                                   y_test=y_test,
                                                   estimator=LogisticRegression(penalty='elasticnet',
                                                                                solver='saga',
                                                                                max_iter=10000,
                                                                                n_jobs=-1,
                                                                                random_state=10),
                                                   params={'C': np.logspace(-5, 1, 5), # C, which controls the amount of overfitting (a lower value should decrease overfitting).
                                                           'l1_ratio': np.linspace(0, 1, 10),
                                                           'class_weight': [None, 'balanced']
                                                          })

            results.loc['logreg',:]=logistic_results
            models['logreg']=logistic_results
        elif alg == 'knn':
            print('KNeighborsClassifier...', flush=True)
            knn_results = get_best_classifier(X_train=X_train,
                                                   y_train=y_train.values.ravel(),
                                                   X_test=X_test,
                                                   y_test=y_test,
                                                   estimator=KNeighborsClassifier(n_jobs=-1),
                                                   params={"n_neighbors": [2, 4, 8]
                                                          })
            results.loc['knn',:]=knn_results
            models['knn']=knn_results
        elif alg == 'randforest':
            print('RandomForrest Classifier...', flush=True)
            rand_forest_results = get_best_classifier(X_train=X_train,
                                                   y_train=y_train.values.ravel(),
                                                   X_test=X_test,
                                                   y_test=y_test,
                                                   estimator=RandomForestClassifier(n_jobs=-1, random_state=10),
                                                   params={
                                                       #'criterion':['gini', 'entropy'],
                                                       'n_estimators': [30, 100, 200, 500, 1000],
                                                       'max_depth': [None, 2, 5, 10, 15, 20], # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                                       'max_features': [None, 'sqrt', 'log2'],
                                                       'class_weight': [None, 'balanced_subsample', 'balanced'],
                                                   })
            results.loc['randforest',:]=rand_forest_results
            models['randforest']=rand_forest_results
        elif alg == 'svc':
            print('SVC....', flush=True)
            svc_results = get_best_classifier(X_train=X_train,
                                                   y_train=y_train.values.ravel(),
                                                   X_test=X_test,
                                                   y_test=y_test,
                                                   estimator=LinearSVC(random_state=10),
                                                   params={
                                                        'C':[0.0001, 0.001, 0.01],
                                                   })
            results.loc['svc',:]=svc_results
            models['svc']=svc_results

        elif alg == 'dctree':
            print('DecisionTreeClassifier....', flush=True)
            dctree_results = get_best_classifier(X_train=X_train,
                                                   y_train=y_train.values.ravel(),
                                                   X_test=X_test,
                                                   y_test=y_test,
                                                   estimator=DecisionTreeClassifier(random_state=10),
                                                   params={
                                                    'max_features':['auto', 'sqrt', 'log2'],
                                                    'criterion':['gini', 'entropy']
                                                   })
            results.loc['dctree',:]=dctree_results
            models['dctree']=dctree_results




        elif alg == 'xgboost':
            print('XGBClassifier...', flush=True)
            xgb_results = get_best_classifier(X_train=X_train,
                                              y_train=y_train.values.ravel(),
                                              X_test=X_test,
                                              y_test=y_test,
                                              estimator=XGBClassifier(objective='binary:logistic',
                                                                        n_jobs=-1,
                                                                        random_state=10),
                                                beta_param=beta
                                              )
            results.loc['xgboost{0}'.format(beta),:]=xgb_results
            models['xgboost{0}'.format(beta)]=xgb_results

        elif alg == 'lgbm':
            print('LGBM Classifier....', flush=True)
            lgbm_results = get_best_classifier(
                X_train=X_train, y_train=y_train.values.ravel(), X_test=X_test, y_test=y_test,
                estimator=LGBMClassifier(
                objective='binary',
                n_jobs=-1,
                random_state=10), beta_param=beta)
            results.loc['lgbm{0}'.format(beta),:]=lgbm_results
            models['lgbm{0}'.format(beta)]=lgbm_results



        elif alg == 'naivebayes':
            print('GaussianNB....', flush=True)
            nb_results = get_best_classifier(X_train=X_train,
                                              y_train=y_train.values.ravel(),
                                              X_test=X_test,
                                              y_test=y_test,
                                              estimator=GaussianNB(),
                                              params={
                                              'var_smoothing': np.logspace(-5, -1, 5)
                                              }
                                              )
            results.loc['naivebayes',:]=nb_results
            models['naivebayes']=nb_results

    with open("models_adison{0}_10.pckl".format(beta), "wb") as f:
        #for model in models:
        pickle.dump(models, f)
    return results




def main():
    X_train = pd.read_csv("data/preprocess/X_train_adisonbalanced.csv")
    y_train = pd.read_csv("data/preprocess/y_train_adisonbalanced.csv")
    X_test = pd.read_csv("data/preprocess/X_test.csv")
    y_test = pd.read_csv("data/preprocess/y_test.csv")
    alg = ['lgbm'] #['knn', 'dctree', 'naivebayes', 'svc', 'dummy', 'logreg', 'randforest', 'xgboost', 'lgbm'] #,
    with timer("Modeling all ..."):
        for b in range(10, 11):
            results = modeling(X_train, y_train, X_test, y_test, alg, beta=b)
            gc.collect()


            results.to_csv("data/preprocess/results_adisonoversampl{0}_10.csv".format(b))

if __name__ == "__main__":
    with timer("Full model run"):
        main()
