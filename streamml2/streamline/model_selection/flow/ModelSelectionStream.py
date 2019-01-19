# Data manipulation
import pandas as pd
import numpy as np

# Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# System
import sys
import os

# Data containers
from collections import defaultdict

#Stats
from scipy.stats import ttest_ind

# Regressors
from ..models.regressors.LinearRegressorPredictiveModel import LinearRegressorPredictiveModel
from ..models.regressors.SupportVectorRegressorPredictiveModel import SupportVectorRegressorPredictiveModel
from ..models.regressors.RidgeRegressorPredictiveModel import RidgeRegressorPredictiveModel
from ..models.regressors.LassoRegressorPredictiveModel import LassoRegressorPredictiveModel
from ..models.regressors.ElasticNetRegressorPredictiveModel import ElasticNetRegressorPredictiveModel
from ..models.regressors.KNNRegressorPredictiveModel import KNNRegressorPredictiveModel
from ..models.regressors.RandomForestRegressorPredictiveModel import RandomForestRegressorPredictiveModel
from ..models.regressors.AdaptiveBoostingRegressorPredictiveModel import AdaptiveBoostingRegressorPredictiveModel
from ..models.regressors.MultilayerPerceptronRegressorPredictiveModel import MultilayerPerceptronRegressorPredictiveModel
from ..models.regressors.LassoLeastAngleRegressorPredictiveModel import LassoLeastAngleRegressorPredictiveModel
from ..models.regressors.LeastAngleRegressorPredictiveModel import LeastAngleRegressorPredictiveModel
from ..models.regressors.BayesianRidgeRegressorPredictiveModel import BayesianRidgeRegressorPredictiveModel
from ..models.regressors.ARDRegressorPredictiveModel import ARDRegressorPredictiveModel
from ..models.regressors.PassiveAggressiveRegressorPredictiveModel import PassiveAggressiveRegressorPredictiveModel
from ..models.regressors.TheilSenRegressorPredictiveModel import TheilSenRegressorPredictiveModel
from ..models.regressors.HuberRegressorPredictiveModel import HuberRegressorPredictiveModel
from ..models.regressors.GaussianProcessRegressorPredictiveModel import GaussianProcessRegressorPredictiveModel
from ..models.regressors.GradientBoostingRegressorPredictiveModel import GradientBoostingRegressorPredictiveModel
from ..models.regressors.BaggingRegressorPredictiveModel import BaggingRegressorPredictiveModel
from ..models.regressors.DecisionTreeRegressorPredictiveModel import DecisionTreeRegressorPredictiveModel

# Classifiers
from ..models.classifiers.AdaptiveBoostingClassifierPredictiveModel import AdaptiveBoostingClassifierPredictiveModel
from ..models.classifiers.DecisionTreeClassifierPredictiveModel import DecisionTreeClassifierPredictiveModel
from ..models.classifiers.GradientBoostingClassifierPredictiveModel import GradientBoostingClassifierPredictiveModel
from ..models.classifiers.GuassianProcessClassifierPredictiveModel import GuassianProcessClassifierPredictiveModel
from ..models.classifiers.KNNClassifierPredictiveModel import KNNClassifierPredictiveModel
from ..models.classifiers.LogisticRegressionClassifierPredictiveModel import LogisticRegressionClassifierPredictiveModel
from ..models.classifiers.MultilayerPerceptronClassifierPredictiveModel import MultilayerPerceptronClassifierPredictiveModel
from ..models.classifiers.NaiveBayesClassifierPredictiveModel import NaiveBayesClassifierPredictiveModel
from ..models.classifiers.RandomForestClassifierPredictiveModel import RandomForestClassifierPredictiveModel
from ..models.classifiers.StochasticGradientDescentClassifierPredictiveModel import StochasticGradientDescentClassifierPredictiveModel
from ..models.classifiers.SupportVectorClassifierPredictiveModel  import SupportVectorClassifierPredictiveModel 

#Utils
from ....utils.helpers import listofdict2dictoflist

# Print Settings
import warnings
warnings.filterwarnings("ignore")


"""
Class Purpose: Flow through several models, gridsearching and returning a hypertuned version of each. Enabled with competition to see quickly who performed best and on what metrics.

Core Methods:

1. flow - 
"""
class ModelSelectionStream:
    #members
    _X=None
    _y=None
    _test_size=None
    _nfolds=None
    _nrepeats=None
    _n_jobs=None
    _verbose=None
    _metrics=None
    _wrapper_models=None
    _bestEstimators=None
    _scoring_results=None
    _final_results=None
    _regressors_results_list=None
    _regressors_results=None
    _classifier_results_list=None
    _classifiers_results=None
    _modelSelection=None
    _stratified=None
    _metrics_significance_dict=None
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
	Method: getActualErrorOnTest:
    
    @usage: Called before kfold cross validation to get the actual estimate of each hyper-tuned model on the train test split.
              However, the kfold cross validation is often a better generalized error than this.
    
    @param: wrapper_models - list, list of wrapper models constructed with a getCode() method and a validate() method.
    @param: metrics - list, list of metrics used to benchmark the models.
    @return dict, final results dictionary representing the true error when trained on by the Xtrain and tested on by Xtest with a test_size percentage partition.
	"""
    def getActualErrorOnTest(self, metrics, wrapper_models):
        self._final_results={}
        for model in wrapper_models:
            model._model.fit(self._X_train, self._y_train)
            results=model.validate(self._X_test, self._y_test, metrics)
            self._final_results[model.getCode()]=results
            
        print("*************************************************************")
        print("=> (Final Results) => True Error Between Xtrain Xtest")
        print("*************************************************************")
        df_results = pd.DataFrame(self._final_results)
        print(df_results)
        
        df_results.plot(title='Errors by Model')
        plt.xticks(range(len(df_results.index.tolist())), df_results.index.tolist())
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)
        plt.show()
        
        return self._final_results
    """
	Method: getBestEstimators:
    
    @usage: Called after the models have been hypertuned, will return the dictionary containing each model object.
    
    @return dict or None
	"""
    def getBestEstimators(self):
        return self._bestEstimators

    """
	Methods: determineBestEstimators
	
    @usage updates best estiminators dictioanry based on provided models
    
    @param: models, list; each model object in competition
    @return dict
	"""
    def determineBestEstimators(self, models):
        self._bestEstimators={}
        print("**************************************************")
        print("Determining Best Estimators.")
        print("**************************************************")
        for model in models:
            self._bestEstimators[model.getCode()]=model.getBestEstimator()

        return self._bestEstimators
        
    """
	Methods: handleRegressors
	
    @usage called when regressor=True and len(metrics)>0. hence will compete regressors and let you know who did the best on your data and on what metrics.
    
    @param Xtest, pd.DataFrame
    @param ytest, pd.Dataframe
    @param metrics, list; specified metrics from sklearn
    @param wrapper_models, list; specified streamml models wrapping sklearn regressors or classifiers
    @param cut, int; point which stratified kfold should keep points beneath spread evenly

    
    @return pd.DataFrame
	"""
    def handleRegressors(self, Xtrain, ytrain, metrics, wrapper_models, cut, stratified):
        
        return_dict={}
        
        self._regressors_results_list=defaultdict(list)
        self._regressors_results=dict()
        
        # Get a StratifiedKFold Cross Validation parameters
        if stratified:
            
            assert self._cut != None , "you must select a cut point for your stratified folds to equally distribute your critical points"
            rskf = RepeatedStratifiedKFold(n_splits=self._nfolds, n_repeats=self._nrepeats,random_state=36851234)
            
            ycut = ytrain.copy()
            ycut[ycut < cut] = 1
            ycut[ycut >= cut] = 0
    
            for train_index, test_index in rskf.split(Xtrain, ycut):
                
                fold_X_train, fold_X_test = Xtrain.iloc[train_index,:], Xtrain.iloc[test_index,:]
                fold_y_train, fold_y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
                
                for model in wrapper_models:
                    model._model.fit(fold_X_train, fold_y_train)
                    results=model.validate(fold_X_test, fold_y_test, metrics)
                    self._regressors_results_list[model.getCode()].append(results)
            
            # Convert our repeated stratified K folds into an average dataframe
            for k,v in  self._regressors_results_list.items():
                example_df = pd.DataFrame(self._regressors_results_list[k])
                mean = example_df.mean()
                self._regressors_results[k]=mean

        else:
            kf = KFold(n_splits=self._nfolds)
            for train_index, test_index in kf.split(Xtrain):
                fold_X_train, fold_X_test = Xtrain.iloc[train_index,:], Xtrain.iloc[test_index,:]
                fold_y_train, fold_y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
                
                for model in wrapper_models:
                    model._model.fit(fold_X_train, fold_y_train)
                    results=model.validate(fold_X_test, fold_y_test, metrics)
                    self._regressors_results_list[model.getCode()].append(results)
            
            # Convert our repeated stratified K folds into an average dataframe
            for k,v in  self._regressors_results_list.items():
                example_df = pd.DataFrame(self._regressors_results_list[k])
                mean = example_df.mean()
                self._regressors_results[k]=mean
        return_dict["avg_kfold"]=self._regressors_results
        
        # create a pandas dataframe of each metric on each model
        print("*****************************************")
        print("=> (Regressor) => Performance Sheet")
        print("*****************************************")
        
        df_results = pd.DataFrame(self._regressors_results)
        print(df_results)
        
        df_results.plot(title='Errors by Model')
        plt.xticks(range(len(df_results.index.tolist())), df_results.index.tolist())
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)
        plt.show()
        

        
        if self._stats:
            print("*****************************************************************************************")
            print("=> (Regressor) => (Two Tailed T-test) =>(Calculating Statistical Differences In Means)")
            print("*****************************************************************************************")
            self._metrics_significance_dict={}
            for m in metrics:
                ttest_sig_mat = np.zeros((len(wrapper_models), len(wrapper_models)))
                for i, k in enumerate(self._regressors_results.keys()):
                    for j,k2 in enumerate(self._regressors_results.keys()):
                        nd=listofdict2dictoflist(self._regressors_results_list[k])
                        nd2=listofdict2dictoflist(self._regressors_results_list[k2])
                        p_int=ttest_ind(nd[m],
                                        nd2[m], 
                                        equal_var = False)
                        if p_int[1] <= 0.05:
                            ttest_sig_mat[i,j]=1
                df=pd.DataFrame(ttest_sig_mat)
                df.index=list(self._regressors_results.keys())
                df.columns=list(self._regressors_results.keys())
                self._metrics_significance_dict[m]=df
            return_dict["significance"]=self._metrics_significance_dict
        # plot models against one another in charts
        
        return return_dict
        
    """
	Methods: handleClassifiers
	
    @usage called when regressor=False and len(metrics)>0. hence will compete classifiers and let you know who did the best on your data and on which metrics.
    
    @param Xtest, pd.DataFrame
    @param ytest, pd.Dataframe
    @param metrics, list; specified metrics from sklearn
    @param wrapper_models, list; specified streamml models wrapping sklearn regressors or classifiers
    
    @return pd.DataFrame
	"""
    def handleClassifiers(self, Xtrain, ytrain, metrics, wrapper_models, stratified):
            
        return_dict={}
        
        self._classifier_results_list=defaultdict(list)
        self._classifier_results=dict()
        
        # Get a StratifiedKFold Cross Validation parameters
        if stratified:
            
            rskf = RepeatedStratifiedKFold(n_splits=self._nfolds, n_repeats=self._nrepeats,random_state=36851234)
            for train_index, test_index in rskf.split(Xtrain, ytrain):
                
                fold_X_train, fold_X_test = Xtrain.iloc[train_index,:], Xtrain.iloc[test_index,:]
                fold_y_train, fold_y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
                
                for model in wrapper_models:
                    model._model.fit(fold_X_train, fold_y_train)
                    results=model.validate(fold_X_test, fold_y_test, metrics)
                    self._classifier_results_list[model.getCode()].append(results)
    
            # Convert our repeated stratified K folds into an average dataframe
            for k,v in  self._classifier_results_list.items():
                example_df = pd.DataFrame(self._classifier_results_list[k])
                mean = example_df.mean()
                self._classifier_results[k]=mean
        else:
            kf = KFold(n_splits=self._nfolds)
            for train_index, test_index in kf.split(Xtrain):
                fold_X_train, fold_X_test = Xtrain.iloc[train_index,:], Xtrain.iloc[test_index,:]
                fold_y_train, fold_y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
                
                for model in wrapper_models:
                    model._model.fit(fold_X_train, fold_y_train)
                    results=model.validate(fold_X_test, fold_y_test, metrics)
                    self._classifier_results_list[model.getCode()].append(results)
            
            # Convert our repeated stratified K folds into an average dataframe
            for k,v in  self._classifier_results_list.items():
                example_df = pd.DataFrame(self._classifier_results_list[k])
                mean = example_df.mean()
                self._classifier_results[k]=mean
        
        return_dict["avg_kfold"]=self._classifier_results
        
        # create a pandas dataframe of each metric on each model

        print("*****************************************")
        print("=> (Classifier) => Performance Sheet")
        print("*****************************************")
        
        df_results = pd.DataFrame(self._classifier_results)
        print(df_results)
        
        df_results.plot(title='Errors by Model')
        plt.xticks(range(len(df_results.index.tolist())), df_results.index.tolist())
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)
        plt.show()
        # plot models against one another in charts
        
        if self._stats:
            
            print("*****************************************************************************************")
            print("=> (Classifier) => (Two Tailed T-test) =>(Calculating Statistical Differences In Means)")
            print("*****************************************************************************************")
            self._metrics_significance_dict={}
            for m in metrics:
                ttest_sig_mat = np.zeros((len(wrapper_models), len(wrapper_models)))
                for i, k in enumerate(self._classifier_results.keys()):
                    for j,k2 in enumerate(self._classifier_results.keys()):
                        nd=listofdict2dictoflist(self._classifier_results_list[k])
                        nd2=listofdict2dictoflist(self._classifier_results_list[k2])
                        p_int=ttest_ind(nd[m],
                                        nd2[m], 
                                        equal_var = False)
                        
                        if p_int[1] <= 0.10:
                            ttest_sig_mat[i,j]=1
                        else:
                            ttest_sig_mat[i,j]=0
                df=pd.DataFrame(ttest_sig_mat)
                df.index=list(self._classifier_results.keys())
                df.columns=list(self._classifier_results.keys())
                self._metrics_significance_dict[m]=df
            return_dict["significance"]=self._metrics_significance_dict
            # plot models against one another in charts
        
        return return_dict
        
        
    """
    Methods: flow
    
    @usage meant to make models flow
    
    @param models_to_flow, list.
    @param params, dict.
                  
    """
    def flow(self, 
             models_to_flow=[], 
             params=None, 
             test_size=0.3, 
             nfolds=3,
             nrepeats=10,
             stats=True,
             stratified=False,
             n_jobs=1, 
             metrics=[], 
             verbose=False, 
             regressors=True,
             modelSelection=False,
             cut=None):
      
        assert isinstance(nfolds, int), "nfolds must be integer"
        assert isinstance(nrepeats, int), "nrepeats must be integer"
        assert isinstance(n_jobs, int), "n_jobs must be integer"
        assert isinstance(verbose, bool), "verbosem ust be bool"
        assert isinstance(params, dict), "params must be a dict"
        assert isinstance(test_size, float), "test_size must be a float"
        assert isinstance(metrics, list), "model scoring must be a list"
        assert isinstance(regressors, bool), "regressor must be bool"
        assert isinstance(modelSelection, bool), "modelSelection must be bool"
        assert isinstance(stratified, bool), "modelSelection must be bool"
        assert isinstance(stats, bool), "stats must be bool"
        self._nfolds=nfolds
        self._nrepeats=nrepeats
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._allParams=params
        self._metrics=metrics
        self._test_size=test_size
        self._regressors=regressors
        self._modelSelection=modelSelection
        self._cut = cut
        self._stratified=stratified
        self._stats=stats
        
        # Inform the streamline to user.
        stringbuilder=""
        for thing in models_to_flow:
            stringbuilder += thing
            stringbuilder += " --> "
            
        if self._verbose:
            
            if self._regressors:
                print("*************************")
                print("=> (Regressor) "+"=> Model Selection Streamline: " + stringbuilder[:-5])
                print("*************************")
            elif self._regressors == False:
                print("*************************")
                print("=> (Classifier) "+"=> Model Selection Streamline: " + stringbuilder[:-5])
                print("*************************")
            else:
                print("Invalid model selected. Please set regressors=True or regressors=False.")
                print
        
        
        ###########################################
        ########## Regressors Start Here ##########
        ###########################################
        
        def linearRegression():
            
            self._lr_params={}
            for k,v in self._allParams.items():
                if "lr" in k:
                    self._lr_params[k]=v

                
            model = LinearRegressorPredictiveModel(self._X_train, 
                                                   self._y_train,
                                                   self._lr_params,
                                                   self._nfolds, 
                                                   self._n_jobs,
                                                   self._verbose)
            return model
            
        def supportVectorRegression():
            self._svr_params={}
            for k,v in self._allParams.items():
                if "svr" in k:
                    self._svr_params[k]=v

                
            model = SupportVectorRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._svr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        def randomForestRegression():
            self._rfr_params={}
            for k,v in self._allParams.items():
                if "rfr" in k:
                    self._rfr_params[k]=v

                
            model = RandomForestRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._rfr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        

        
        def adaptiveBoostingRegression():
            self._abr_params={}
            for k,v in self._allParams.items():
                if "abr" in k:
                    self._abr_params[k]=v

                
            model = AdaptiveBoostingRegressorPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._abr_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def knnRegression():
            self._knnr_params={}
            for k,v in self._allParams.items():
                if "knnr" in k:
                    self._knnr_params[k]=v

            
            
            model = KNNRegressorPredictiveModel(self._X_train, 
                                                self._y_train,
                                                self._knnr_params,
                                                self._nfolds, 
                                                self._n_jobs,
                                                self._verbose)
            
            return model
            
        def ridgeRegression():
            self._ridge_params={}
            for k,v in self._allParams.items():
                if "ridge" in k:
                    self._ridge_params[k]=v

                
            model = RidgeRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._ridge_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        def lassoRegression():
            self._lasso_params={}
            for k,v in self._allParams.items():
                if "lasso" in k:
                    self._lasso_params[k]=v

                
            model = LassoRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._lasso_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        def elasticNetRegression():
            self._enet_params={}
            for k,v in self._allParams.items():
                if "enet" in k:
                    self._enet_params[k]=v

            model = ElasticNetRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._enet_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        def multilayerPerceptronRegression():
            self._mlpr_params={}
            for k,v in self._allParams.items():
                if "mlpr" in k:
                    self._mlpr_params[k]=v

            model = MultilayerPerceptronRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._mlpr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        
        def leastAngleRegression():
            self._lar_params={}
            for k,v in self._allParams.items():
                if "lar" in k:
                    self._lar_params[k]=v

            model = LeastAngleRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._lar_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        
        def lassoLeastAngleRegression():
            self._lasso_lar_params={}
            for k,v in self._allParams.items():
                if "lasso_lar" in k:
                    self._lasso_lar_params[k]=v

            model = LassoLeastAngleRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._lasso_lar_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
       
        def bayesianRidgeRegression():
            self._bays_ridge={}
            for k,v in self._allParams.items():
                if "bays_ridge" in k:
                    self._bays_ridge[k]=v

            model = BayesianRidgeRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._bays_ridge,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def ardRegression():
            self._ardr_params={}
            for k,v in self._allParams.items():
                if "ardr" in k:
                    self._ardr_params[k]=v

            model = ARDRegressorPredictiveModel(self._X_train, 
                                                self._y_train,
                                                self._ardr_params,
                                                self._nfolds, 
                                                self._n_jobs,
                                                self._verbose)
            
            return model
        
        def passiveAggressiveRegression():
            self._par_params={}
            for k,v in self._allParams.items():
                if "par" in k:
                    self._par_params[k]=v

            model = PassiveAggressiveRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._par_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def theilSenRegression():
            self._tsr_params={}
            for k,v in self._allParams.items():
                if "tsr" in k:
                    self._tsr_params[k]=v

            model = TheilSenRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._tsr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def huberRegression():
            self._hr_params={}
            for k,v in self._allParams.items():
                if "hr" in k:
                    self._hr_params[k]=v

            model = HuberRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._hr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def gaussianProcessRegression():
            self._gpr_params={}
            for k,v in self._allParams.items():
                if "gpr" in k:
                    self._gpr_params[k]=v

            model = GaussianProcessRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._gpr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def gradientBoostingRegression():
            self._gbr_params={}
            for k,v in self._allParams.items():
                if "gbr" in k:
                    self._gbr_params[k]=v

            model = GradientBoostingRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._gbr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def baggingRegression():
            self._br_params={}
            for k,v in self._allParams.items():
                if "br" in k:
                    self._br_params[k]=v

            model = BaggingRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._br_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model
        
        def decisionTreeRegression():
            self._dtr_params={}
            for k,v in self._allParams.items():
                if "dtr" in k:
                    self._dtr_params[k]=v

            model = DecisionTreeRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._dtr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            
            return model

        ############################################
        ########## Classifiers Start Here ##########
        ############################################
        
        def adaptiveBoostingClassifier():
            self._abc_params={}
            for k,v in self._allParams.items():
                if "abc" in k:
                    self._abc_params[k]=v

                
            model = AdaptiveBoostingClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._abc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        

        def decisionTreeClassifier():
            self._dtc_params={}
            for k,v in self._allParams.items():
                if "dtc" in k:
                    self._dtc_params[k]=v

                
            model = DecisionTreeClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._dtc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        
        def gradientBoostingClassifier():
            self._gbc_params={}
            for k,v in self._allParams.items():
                if "gbc" in k:
                    self._gbc_params[k]=v

                
            model = GradientBoostingClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._gbc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def guassianProcessClassifier():
            self._gpc_params={}
            for k,v in self._allParams.items():
                if "gpc" in k:
                    self._gpc_params[k]=v

                
            model = GuassianProcessClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._gpc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        

        def knnClassifier():
            self._knnc_params={}
            for k,v in self._allParams.items():
                if "knnc" in k:
                    self._knnc_params[k]=v

                
            model = KNNClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._knnc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def logisticRegressionClassifier():
            self._logr_params={}
            for k,v in self._allParams.items():
                if "logr" in k:
                    self._logr_params[k]=v

                
            model = LogisticRegressionClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._logr_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def multilayerPerceptronClassifier():
            self._mlpc_params={}
            for k,v in self._allParams.items():
                if "mlpc" in k:
                    self._mlpc_params[k]=v

                
            model = MultilayerPerceptronClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._mlpc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def naiveBayesClassifier():
            self._nbc_params={}
            for k,v in self._allParams.items():
                if "nbc" in k:
                    self._nbc_params[k]=v

                
            model = NaiveBayesClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._nbc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def randomForestClassifier():
            self._rfc_params={}
            for k,v in self._allParams.items():
                if "rfc" in k:
                    self._rfc_params[k]=v

                
            model = RandomForestClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._rfc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
        
        def stochasticGradientDescentClassifier():
            self._sgdc_params={}
            for k,v in self._allParams.items():
                if "sgdc" in k:
                    self._sgdc_params[k]=v

                
            model = StochasticGradientDescentClassifierPredictiveModel(self._X_train, 
                                                                      self._y_train,
                                                                      self._sgdc_params,
                                                                      self._nfolds, 
                                                                      self._n_jobs,
                                                                      self._verbose)
            return model
        
        def supportVectorClassifier():
            self._svc_params={}
            for k,v in self._allParams.items():
                if "svc" in k:
                    self._svc_params[k]=v

                
            model = SupportVectorClassifierPredictiveModel(self._X_train, 
                                                                      self._y_train,
                                                                      self._svc_params,
                                                                      self._nfolds, 
                                                                      self._n_jobs,
                                                                      self._verbose)
            return model
        
        # Valid regressors
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



        # Valid classifiers
        classification_options = {'abc':adaptiveBoostingClassifier,
                                  'dtc':decisionTreeClassifier,
                                  'gbc':gradientBoostingClassifier,
                                    'gpc':guassianProcessClassifier,
                                    'knnc':knnClassifier,
                                    'logr':logisticRegressionClassifier,
                                    'mlpc':multilayerPerceptronClassifier,
                                    'nbc':naiveBayesClassifier,
                                    'rfc':randomForestClassifier,
                                    'sgd':stochasticGradientDescentClassifier,
                                    'svc':supportVectorClassifier}
        
		# Train test split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                     self._y,
                                                                                     test_size=self._test_size)

        
    
        # Wrapper models    
        self._wrapper_models=[]
        
        if self._regressors:
            for key in models_to_flow:
                 self._wrapper_models.append(regression_options[key]())
        elif self._regressors == False:
            for key in models_to_flow:
                 self._wrapper_models.append(classification_options[key]())
        else:
            print("Invalid model type. Please set regressors=True or regressors=False.")
            print
		
        #results dict
        return_dict={}
        
        # Hyper-tune models
        self._bestEstimators = self.determineBestEstimators(self._wrapper_models)
		
        # Do you want 1 or many models returned? If verbose, a visual appears.
        if self._modelSelection:
            assert(len(self._wrapper_models) > 1, "In order to compare models, you must have more than one. Add some in the first argument of `flow(...)` and retry.")
            if not len(self._metrics) > 0:
                if self._regressors:
                    self._metrics=['rmse',
                                    'mse',
                                    'r2',
                                    'explained_variance',
                                    'mean_absolute_error',
                                    #'mean_squared_log_error', #not general - negative values not supported
                                    'median_absolute_error'
                                    ]
                else:
                    self._metrics=[  #"auc", # not general - multi-class problems
                                     "precision",
                                     "recall",
                                     "f1",
                                     "accuracy",
                                     "kappa",
                                     #"log_loss" # not working with splits right now
                                    ]
            
            # Get final results from actual train test dataset
            self._final_results = self.getActualErrorOnTest(self._metrics,self._wrapper_models)
            
            if self._regressors:
                return_dict = self.handleRegressors(self._X_train, 
                                                      self._y_train, 
                                                      self._metrics,
                                                      self._wrapper_models,
                                                      self._cut,
                                                      self._stratified)
            elif self._regressors == False: #classifiers
                return_dict = self.handleClassifiers (self._X_train,
                                                       self._y_train,
                                                       self._metrics,
                                                       self._wrapper_models,
                                                       self._stratified)
            else:
                print("You selected an invalid type of model.")
                print
            
            return_dict["final_errors"]=self._final_results
        return_dict["models"]=self._bestEstimators
		# Return each best estimator the user is interested in
        return return_dict
    

