
# Data manipulation
import pandas as pd
import numpy as np
import math

# Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# System
import sys
import os

# Data containers
from collections import defaultdict

# Print Settings
import warnings
warnings.filterwarnings("ignore")

# Regressors
from ..models.regressors.PLSRegressorFeatureSelectionModel import PLSRegressorFeatureSelectionModel
from ..models.regressors.MixedSelectionFeatureSelectionModel import MixedSelectionFeatureSelectionModel
from ...model_selection.models.regressors.SupportVectorRegressorPredictiveModel import SupportVectorRegressorPredictiveModel
from ...model_selection.models.regressors.LassoRegressorPredictiveModel import LassoRegressorPredictiveModel
from ...model_selection.models.regressors.ElasticNetRegressorPredictiveModel import ElasticNetRegressorPredictiveModel
from ...model_selection.models.regressors.RandomForestRegressorPredictiveModel import RandomForestRegressorPredictiveModel
from ...model_selection.models.regressors.AdaptiveBoostingRegressorPredictiveModel import AdaptiveBoostingRegressorPredictiveModel

# Classifiers
from ...model_selection.models.classifiers.AdaptiveBoostingClassifierPredictiveModel import AdaptiveBoostingClassifierPredictiveModel
from ...model_selection.models.classifiers.RandomForestClassifierPredictiveModel import RandomForestClassifierPredictiveModel
from ...model_selection.models.classifiers.SupportVectorClassifierPredictiveModel import SupportVectorClassifierPredictiveModel
            
# MADM
from .MADMFeatureSelection import MADMFeatureSelection

class FeatureSelectionStream:
    
    #Properties
    _X=None
    _y=None
    _test_size=None
    _nfolds=None
    _n_jobs=None
    _verbose=None
    _metrics=None
    _test_size=None
    _wrapper_models=None
    _bestEstimators=None
    _regressors_results=None
    _classifiers_results=None
    _modelSelection=None
    _featurePercentage=None
    _random_grid=None
    _n_iter=None
    """
    Constructor: __init__:
    
    @param: X : pd.DataFrame, dataframe representing core dataset.
    @param: y : pd.DataFrame, dataframe representing response variable, either numeric or categorical.
    """
    def __init__(self,X,y):
        assert isinstance(X, pd.DataFrame), "X was not a pandas DataFrame"
        assert any([isinstance(y,pd.DataFrame), isinstance(y,pd.Series)]), "y was not a pandas DataFrame or Series"
        self._X = X
        self._y = y

    
    """
    Methods: flow
    
    @usage meant to make models flow
    
    @param models_to_flow, list.
    @param params, dict.
                  
    """
    def flow(self, 
             models_to_flow=[], 
             params=None, 
             test_size=0.2, 
             nfolds=3, 
             nrepeats=3,
             n_jobs=1, 
             metrics=[], 
             verbose=False, 
             regressors=True,
             ensemble=False,
             featurePercentage=0.25,
             random_grid=False,
             n_iter=10):

        assert isinstance(nfolds, int), "nfolds must be integer"
        assert isinstance(nrepeats, int), "nrepeats must be integer"
        assert isinstance(n_jobs, int), "n_jobs must be integer"
        assert isinstance(verbose, bool), "verbosem ust be bool"
        assert isinstance(params, dict), "params must be a dict"
        assert isinstance(test_size, float), "test_size must be a float"
        assert isinstance(metrics, list), "model scoring must be a list"
        assert isinstance(regressors, bool), "regressor must be bool"
        assert isinstance(ensemble, bool), "ensemble must be bool"
        assert isinstance(random_grid, bool), "random_grid must be bool"
        assert isinstance(n_iter, int), "random_grid must be int"
        
        self._nfolds=nfolds
        self._nrepeats=nrepeats
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._allParams=params
        self._metrics=metrics
        self._test_size=test_size
        self._regressors=regressors
        self._ensemble=ensemble
        self._featurePercentage=featurePercentage
        self._random_grid=random_grid
        self._n_iter=n_iter
        
        # Inform the streamline to user.
        stringbuilder=""
        for thing in models_to_flow:
            stringbuilder += thing
            stringbuilder += " --> "
            
        if self._verbose:
            if self._regressors:
                print("*************************************************************************")
                print("=> (Regressor) "+"=> Feature Selection Streamline: " + stringbuilder[:-5])
                print("*************************************************************************")
            elif self._regressors == False:
                print("*************************************************************************")
                print("=> (Classifier) "+"=> Feature Selection Streamline: " + stringbuilder[:-5])
                print("*************************************************************************")
            else:
                print("Invalid model selected. Please set regressors=True or regressors=False.")
                print
        
        def supportVectorRegression():
            self._svr_params={}
            for k,v in self._allParams.items():
                if "svr" == k.split("__")[0]:
                    self._svr_params[k]=v

            
            self._svr_params["svr__kernel"]=['linear']
            model = SupportVectorRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._svr_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return abs(model.getBestEstimator().coef_.flatten())
        
        def randomForestRegression():
            self._rfr_params={}
            for k,v in self._allParams.items():
                if "rfr" == k.split("__")[0]:
                    self._rfr_params[k]=v

                
            model = RandomForestRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._rfr_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return abs(model.getBestEstimator().feature_importances_.flatten())
            
        

        def adaptiveBoostingRegression():
            self._abr_params={}
            for k,v in self._allParams.items():
                if "abr" == k.split("__")[0]:
                    self._abr_params[k]=v

                
            model = AdaptiveBoostingRegressorPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._abr_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return abs(model.getBestEstimator().feature_importances_.flatten())
        
        def lassoRegression():
            self._lasso_params={}
            for k,v in self._allParams.items():
                if "lasso" == k.split("__")[0]:
                    self._lasso_params[k]=v

                
            model = LassoRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._lasso_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return abs(model.getBestEstimator().coef_.flatten())
            
        def elasticNetRegression():
            self._enet_params={}
            for k,v in self._allParams.items():
                if "enet" == k.split("__")[0]:
                    self._enet_params[k]=v

            model = ElasticNetRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._enet_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return abs(model.getBestEstimator().coef_.flatten())
    
        def mixed_selection():
            
            mixedSelectionModel = MixedSelectionFeatureSelectionModel(self._X, self._y, self._allParams, self._verbose)
            keptFeatures = mixedSelectionModel.execute()
            return keptFeatures

        def partialLeastSquaresRegression():

            plsrModel = PLSRegressorFeatureSelectionModel(self._X, self._y, self._allParams, self._verbose)
            abs_coefs = plsrModel.execute()
            return abs_coefs
        
        ############################################
        ########## Classifiers Start Here ##########
        ############################################
        
        def adaptiveBoostingClassifier():
            self._abc_params={}
            for k,v in self._allParams.items():
                if "abc" == k.split("__")[0]:
                    self._abc_params[k]=v

                
            model = AdaptiveBoostingClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._abc_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return model.getBestEstimator().feature_importances_.flatten()
        
        def randomForestClassifier():
            self._rfc_params={}
            for k,v in self._allParams.items():
                if "rfc" == k.split("__")[0]:
                    self._rfc_params[k]=v

                
            model = RandomForestClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._rfc_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)
            return model.getBestEstimator().feature_importances_.flatten()
        
        
        def supportVectorClassifier():
            self._svc_params={}
            for k,v in self._allParams.items():
                if "svc" == k.split("__")[0]:
                    self._svc_params[k]=v

            
            self._svc_params["svc__kernel"]=['linear']
            model = SupportVectorClassifierPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._svc_params,
                                                          nfolds=self._nfolds, 
                                                          n_jobs=self._n_jobs,
                                                           random_grid=self._random_grid,
                                                           n_iter=self._n_iter,
                                                          verbose=self._verbose)

            coefs=model.getBestEstimator().coef_
            prods=coefs[0,:]
            for i in range(1, len(coefs)):
              prods = np.multiply(prods, coefs[i,:])
            return abs(prods)
        
        # Valid regressors
        regression_options = {"mixed_selection" : mixed_selection,
                               "svr" : supportVectorRegression,
                               "rfr":randomForestRegression,
                               "abr":adaptiveBoostingRegression,
                               "lasso":lassoRegression,
                               "enet":elasticNetRegression,
                               "plsr":partialLeastSquaresRegression}



        # Valid classifiers
        classification_options = {'abc':adaptiveBoostingClassifier,
                                    'rfc':randomForestClassifier,
                                    'svc':supportVectorClassifier
                                 }
        
        # Define return dictionary
        return_dict={}
        
		# Train test split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                     self._y,
                                                                                     test_size=self._test_size)

        # Wrapper models    
        self._key_features={}

        if self._regressors:
            for key in models_to_flow:
                 self._key_features[key]=regression_options[key]()
        elif self._regressors == False:
            for key in models_to_flow:
                 self._key_features[key]=classification_options[key]()
        else:
            print("Invalid model type. Please set regressors=True or regressors=False.")
            print
        if self._verbose:
            print
        
        return_dict['feature_importances']=pd.DataFrame(self._key_features, index=self._X.columns.tolist())
        
        self._ensemble_results = None
        self._kept_features = None
        if self._ensemble:

            madmSelector=MADMFeatureSelection(self._X, self._y, self._key_features, self._featurePercentage, self._verbose)
            self._ensemble_results, self._kept_features=madmSelector.execute()
          # Print data with only those features
            return_dict['ensemble_results']=self._ensemble_results
            return_dict['kept_features']=self._kept_features

        return return_dict