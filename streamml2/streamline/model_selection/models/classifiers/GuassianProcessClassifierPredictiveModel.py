import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.gaussian_process  import GaussianProcessClassifier 

class GuassianProcessClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, gpc_params, nfolds=3, n_jobs=1, scoring=None, random_grid=False, n_iter=10, verbose=True):
        
        self._code="gpc"
        
        if verbose:
            print ("Constructed GaussianProcessClassifier PredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, gpc_params, nfolds, n_jobs, scoring, random_grid, n_iter,  verbose)
        self._model = self.constructClassifier(GaussianProcessClassifier(), self._random_grid)
        
       
