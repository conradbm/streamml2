
import pandas as pd
import numpy as np

# FOR ANY SYSTEM: INCLUDE STREAMML
import sys
sys.path.append('/Users/bmc/Desktop/') #I.e., make it a path variable

from streamml.streamline.transformation.flow.TransformationStream import TransformationStream
from streamml.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream
from streamml.streamline.feature_selection.flow.FeatureSelectionStream import FeatureSelectionStream

# git checkout -b modelSelectionUpdates
# git push -u origin modelSelectionUpdates

# FOR MAC:
# nano ~/.bash_profile
# export PYTHONPATH="${PYTHONPATH}:/Users/bmc/Desktop/streamml"
# source ~/.bash_profile
# python -W ignore tester.py

#X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
#y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))

X2 = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
y2 = pd.DataFrame(np.random.binomial(1,0.25,200))

"""
D = pd.read_csv("../Series3_6.15.17_padel.csv")
X = D.iloc[:,2:]
y = D.iloc[:,1]
ynakiller = y.isna()
X = X.loc[-ynakiller,:]
y = y.loc[-ynakiller]
y=pd.DataFrame(pd.factorize(y)[0].tolist())
X.replace([np.nan, np.inf, -np.inf],0, inplace=True)

"""


#
# Supported Regression Feature Selection:
# mixed_selection
# rfr
# abr
# svr
# Supported Classifier Feature Selection:
# rfc
# abc
# svc
#
# {Coming Soon}
# Regressors:
# pls
# Ensemble:
# topsis (weighted product, weighted sum, weighted average, linear, log, ect..)
feature_dict = FeatureSelectionStream(X2,y2).flow(["svc", "abc", "rfc"],
                                                params={},
                                                verbose=True,
                                                regressors=False,
                                                ensemble=True)
print(feature_dict)



#print(X.shape)
#print (y.shape)

"""
Transformation Options:
["scale","normalize","boxcox","binarize","pca","kmeans", "brbm]
kmeans: n_clusters
pca: percent_variance (only keeps # comps that capture this %)
binarize: threshold (binarizes those less than threshold as 0 and above as 1)
tsne: n_components

# sklearn.decomposition.sparse_encode
# sklearn.preprocessing.PolynomialFeatures
# sklearn.linear_model.OrthogonalMatchingPursuit

"""


#Xnew = TransformationStream(X).flow(["scale","tsne"], 
#                                    params={"tnse_n_components":4,
#                                            "pca__percent_variance":0.75, 
#                                            "kmeans__n_clusters":2},
#                                   verbose=True)
#print(Xnew)




"""
Model Selection Options (Regression):
        regression_options = {"lr" : linearRegression,
                               "svr" : supportVectorRegression,
                               "rfr":randomForestRegression,
                               "abr":adaptiveBoostingRegression,
                               "knnr":knnRegression,
                               "ridge":ridgeRegression,
                               "lasso":lassoRegression,
                               "enet":elasticNetRegression,
                               "mlpr":multilayerPerceptronRegression,
                               "br":baggingRegression,
                               "dtr":decisionTreeRegression,
                               "gbr":gradientBoostingRegression,
                               "gpr":gaussianProcessRegression,
                               "hr":huberRegression,
                               "tsr":theilSenRegression,
                               "par":passiveAggressiveRegression,
                               "ard":ardRegression,
                               "bays_ridge":bayesianRidgeRegression,
                               "lasso_lar":lassoLeastAngleRegression,
                               "lar":leastAngleRegression}

Model parameter options can all be found here in the following links for the model you wish to flow on:
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
http://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression

Metric Options:
['rmse','mse', 'r2','explained_variance','mean_absolute_error','median_absolute_error']


Model Selection Options (Classification) {Coming Soon}

classification_options = {'abc':adaptiveBoostingClassifier,
                            'dtc':decisionTreeClassifier,
                            'gbc':gradientBoostingXlassifier,
                            'gpc':guassianProcessClassifier,
                            'knnc':knnClassifier,
                            'logr':logisticRegressionClassifier,
                            'mlpc':multilayerPerceptronClassifier,
                            'nbc':naiveBayesClassifier,
                            'rfc':randomForestClassifier,
                            'sgd':stochasticGradientDescentClassifier,
                            'svc':supportVectorClassifier}


Metric Options: {Coming Soon}
["auc","prec","recall","f1","accuracy", "kappa","log_loss"]


http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
http://scikithttp://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix





Feature Selection Options: {Coming Soon}

Statistical: p-values
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression
(classif)http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
(classif)http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
(classif)http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression

Tests to work off
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
   
Trees: _feature_importance_
(classif)http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
(classif)http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

Kernels (linear): _coef_
(classif)http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

Regression: _coef_
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
(reg)http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
Preprocessing options


Transformation Options: {Coming Soon}

"""

# Complex Example 

#Regression
"""
performances = ModelSelectionStream(X,y).flow(["svr","abr", "enet", "mlpr"],
                                              params={'svr__C':[1,0.1,0.01,0.001],
                                                      'svr__gamma':[0, 0.01, 0.001, 0.0001],
                                                      'svr__kernel':['poly', 'rbf'],
                                                      'svr__epsilon':[0,0.1,0.01,0.001],
                                                      'svr__degree':[1,2,3,4,5,6,7],
                                                      
                                                      'enet__alpha':[0, 0.1, 0.01,1,10,20],
                                                      'enet__l1_ratio':[.25,.5,.75],
                                                      
                                                      'abr__n_estimators':[10,20,50],
                                                      'abr__learning_rate':[0.1,1,10, 100],
                                                      'rfr__criterion':['mse', 'mae'],
                                                      
                                                      'mlpr__hidden_layer_sizes':[(X.shape[1], 
                                                                                   int(X.shape[1]/2),
                                                                                   int(X.shape[1]/2)),
                                                                                  (100,10,2),
                                                                                  (1000,100,10,1),
                                                                                 (5,5,5,5,5,1)]},
                                                 metrics=['r2','rmse', 'mse',
                                                          'explained_variance','mean_absolute_error',
                                                         'median_absolute_error'],
                                                verbose=True,
                                                modelSelection=True,
                                                regressors=True,
                                                cut=2) # evenly splits folds with points beneath cut.
"""

"""
# Classification
performances = ModelSelectionStream(X2,y2).flow(["abc","logr","mlpc","svc"],
                                              params={'abc__n_estimators':[10,100,1000],
                                                      'abc__learning_rate':[0.001,0.01,0.1,1,10,100],
                                                      'mlpr__hidden_layer_sizes':[(X.shape[1], 
                                                                                   int(X.shape[1]/2),
                                                                                   int(X.shape[1]/2)),
                                                                                  (100,10,2),
                                                                                  (1000,100,10,1),
                                                                                 (5,5,5,5,5,1)]},
                                                 metrics=["auc",
                                                          "prec",
                                                          "recall",
                                                          "f1",
                                                          "accuracy",
                                                          "kappa",
                                                          "log_loss"],
                                                verbose=True,
                                                modelSelection=True,
                                                regressors=False
                                                )
            
print(performances)
"""