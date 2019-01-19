import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
import numpy as np
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/"))
from ..AbstractPredictiveModel import AbstractPredictiveModel


class AbstractRegressorPredictiveModel(AbstractPredictiveModel):

    #constructor
    _options = ['rmse',
                'mse',
                'r2',
                'explained_variance',
                'mean_absolute_error',
                'mean_squared_log_error',
                'median_absolute_error'
                ]
    
    
    _validation_results=None
    
    def __init__(self, modelType, X, y, params, nfolds, n_jobs, scoring, verbose):
        
        #if self._verbose:
            #print("Constructed AbstractRegressorPredictiveModel: "+self._code)
        
        assert modelType == "regressor", "You are creating a regressor, but have not specified it to be one."
        #assert any([isinstance(y.dtypes[0],float),isinstance(y.dtypes[0],float)]), "Your response variable y is not a float."

        self._modelType = modelType
        self._y=y
        self._scoring=scoring
        AbstractPredictiveModel.__init__(self, X, params, nfolds, n_jobs, verbose)
        
    #methods
    def validate(self, Xtest, ytest, metrics, verbose=False):
        assert any([isinstance(metrics, str), isinstance(metrics, list)]), "Your regressor error metric must be a str or list"
        assert all([i in self._options for i in metrics]) , "Your regressor error metric must be in valid: " + ' '.join([i for i in self._options])

        
        
        self._validation_results={}
        for m in metrics:
            if m == 'r2':
                ypred = self._model.predict(Xtest)
                self._validation_results["r2"]=r2_score(ytest, ypred)
            elif m == 'rmse':
                ypred = self._model.predict(Xtest)
                self._validation_results["rmse"]=np.sqrt(mean_squared_error(ytest, ypred))
            elif m == 'mse':
                ypred = self._model.predict(Xtest)
                self._validation_results["mse"]=mean_squared_error(ytest, ypred)
            elif m == 'explained_variance':
                ypred = self._model.predict(Xtest)
                self._validation_results["explained_variance"]=explained_variance_score(ytest, ypred)
            elif m == 'mean_absolute_error':
                ypred = self._model.predict(Xtest)
                self._validation_results["mean_absolute_error"]=mean_absolute_error(ytest, ypred)
            elif m == 'median_absolute_error':
                ypred = self._model.predict(Xtest)
                self._validation_results["median_absolute_error"]=median_absolute_error(ytest,ypred)
            elif m == 'mean_squared_log_error':
                ypred = self._model.predict(Xtest)
                self._validation_results["median_absolute_error"]=mean_squared_log_error(ytest,ypred)
            else:
                print(str(m)+" not a valid regressor metric, skipping.")
        
        return self._validation_results
    
    def constructRegressor(self, model):
        self._pipe          = Pipeline([(self._code, model)])

        
        self._grid          = GridSearchCV(self._pipe,
                                            param_grid=self._params, 
                                            n_jobs=self._n_jobs,
                                            cv=self._nfolds, 
                                            verbose=False)
        
        
        self._model                 = self._grid.fit(self._X,self._y).best_estimator_.named_steps[self._code]
        return self._model    
    
    def getValidationResults(self):
        return self._validation_results