# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 09:24:39 2019

@author: bmccs
"""
from collections import defaultdict

"""
Helper function used inside of ModelSelectionStream.flow. This is used when modelSelection=True
 and multiple models are competing. The data came in as a list of dictionaries and this will convert
 it to a dictionary of lists, which can be easily converted into a pandas data frame.
"""
def listofdict2dictoflist(somelistofdicts):
    nd=defaultdict(list)
    for d in somelistofdicts:
       for key,val in d.items():
          nd[key].append(val)
    return nd


def get_classification_dataset():
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris=load_iris()
    X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y=pd.DataFrame(iris['target'], columns=['target'])
    
    return (X,y)

def get_regression_dataset():
    from sklearn.datasets import load_boston
    import pandas as pd
    
    boston=load_boston()
    X=pd.DataFrame(boston['data'], columns=boston['feature_names'])
    y=pd.DataFrame(boston['target'],columns=["target"])
    
    return (X,y)
"""
Utility for users to quickly get all of the supported regressors for model selection.
"""
def get_model_selection_regressors():
    return ['lr',
             'svr',
             'rfr',
             'abr',
             'knnr',
             'ridge',
             'lasso',
             'enet',
             'mlpr',
             'br',
             'dtr',
             'gbr',
             'gpr',
             'hr',
             'tsr',
             'par',
             'ard',
             'bays_ridge',
             'lasso_lar',
             'lar']
    
"""
Utility for users to quickly get all of the supported classifiers for model selection.
"""
def get_model_selection_classifiers():
    return ['abc',
             'dtc',
             'gbc',
             'gpc',
             'knnc',
             'logr',
             'mlpc',
             'nbc',
             'rfc',
             'sgd',
             'svc']
    
"""
Utility for users to quickly get a parameter grid to start working off of for regressors.
"""
def get_model_selection_regressors_params():
    
    import numpy as np
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    return {'abr__base_estimator':[LinearRegression(), SVR(), RandomForestRegressor()],
            'abr__n_estimators':np.arange(1,50,10),
            'abr__learning_rate':np.arange(0.1,1,0.1),
            'rfr__n_estimators':np.arange(1,50,10),
            'knnr__n_neighbors':np.arange(1,50,10),
            'svr__C':np.arange(0.1,1,0.1),
            'svr__kernel':['rbf','poly','linear'],
            'svr__degree':np.arange(1,10,1),
            'svr__gamma':np.arange(0.1,1,0.1),
            'mlpr__hidden_layer_sizes':[(100), (100,100), (100,100,100)],
            'mlpr__alpha':np.arange(0.1,1.0,0.1),
            'mlpr__activation':['identity','logistic','relu','tanh'],
            'mlpr__learning_rate':['constant','invscaling']}
    
"""
Utility for users to quickly get a parameter grid to start working off of for classifiers.
"""
def get_model_selection_classifiers_params():
    
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    return { 'abc__algorithm':['SAMME'],
             'abc__base_estimator':[LogisticRegression(), SVC(), GaussianNB(), RandomForestClassifier()],
             'abc__n_estimators':np.arange(1,50,7),
             'abc__learning_rate':np.arange(0.1,1,0.1),
             'rfc__n_estimators':np.arange(1,50,10),
             'knnc__n_neighbors':np.arange(1,50,10),
             'svc__C':np.arange(0.001,10,0.01),
             'svc__kernel':['rbf','poly','linear'],
             'svc__degree':np.arange(1,100,2),
             'svc__gamma':np.arange(0.01,10,0.001),
             'mlpc__hidden_layer_sizes':[(100), (100,100), (100,100,100)],
             'mlpc__alpha':np.arange(0.01,1.0,0.01),
             'mlpc__activation':['identity','logistic','relu','tanh'],
             'mlpc__learning_rate':['constant','invscaling']}

"""
Utility for users to quickly get all of the supported regressors for feature selection.
"""
def get_feature_selection_regressors():
    return ["mixed_selection", 
            "rfr", 
            "abr", 
            "svr", 
            "enet", 
            "lasso"]
    
"""
Utility for users to quickly get all of the supported classifiers for feature selection.
"""
def get_feature_selection_classifiers():
    return ["rfc", 
            "abc", 
            "svc"]

def get_feature_selection_params():
    import numpy as np
    
    return {**{"mixed_selection__threshold_in":0.01,
                "mixed_selection__threshold_out":0.05,
                "mixed_selection__verbose":True }, 
            **{"enet__alpha":np.arange(0.1,1,0.1),
                 "enet__l1_ratio":np.arange(0,1,0.1),
                 "enet__fit_intercept":[True, False],
                 "enet__normalize":[True, False],
                 "lasso__alpha":np.arange(0.1,1,0.1),
                 "lasso__fit_intercept":[True, False],
                 "lasso__normalize":[True, False]},
            **{ 'abc__algorithm':['SAMME'],
                 'abc__n_estimators':np.arange(1,50,7),
                 'abc__learning_rate':np.arange(0.1,1,0.1),
                 'rfc__n_estimators':np.arange(1,50,10),
                 'knnc__n_neighbors':np.arange(1,50,10),
                 'svc__C':np.arange(0.001,10,0.01),
                 'svc__kernel':['rbf','poly','linear'],
                 'svc__degree':np.arange(1,100,2),
                 'svc__gamma':np.arange(0.01,10,0.001),
                 'mlpc__hidden_layer_sizes':[(100), (100,100), (100,100,100)],
                 'mlpc__alpha':np.arange(0.01,1.0,0.01),
                 'mlpc__activation':['identity','logistic','relu','tanh'],
                 'mlpc__learning_rate':['constant','invscaling']},
            **{'abr__n_estimators':np.arange(1,50,10),
                'abr__learning_rate':np.arange(0.1,1,0.1),
                'rfr__n_estimators':np.arange(1,50,10),
                'knnr__n_neighbors':np.arange(1,50,10),
                'svr__C':np.arange(0.1,1,0.01),
                'svr__kernel':['rbf','poly','linear'],
                'svr__degree':np.arange(1,100,10),
                'svr__gamma':np.arange(0.1,1,0.1),
                'mlpr__hidden_layer_sizes':[(100), (100,100), (100,100,100)],
                'mlpr__alpha':np.arange(0.1,1.0,0.1),
                'mlpr__activation':['identity','logistic','relu','tanh'],
                'mlpr__learning_rate':['constant','invscaling']}
            }
    
    
def get_transformer_models():
    return ['scale', 
            'normalize', 
            'binarize', 
            'boxcox', 
            'pca',
            'kmeans',
            'brbm', 
            'tsne']

def get_transformer_params():
    return {"kmeans__n_clusters":5,
            "pca__percent_variance":0.9,
            "pca__n_components":2,
            "tsne__n_components":2,
            "binarize":1,
            "brbm_n_components":256}


