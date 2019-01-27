import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.ensemble import BaggingClassifier

class BaggingClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, bc_params, nfolds=3, n_jobs=1, scoring=None,random_grid=False, n_iter=10,  verbose=True):
        
        self._code="bc"
        
        if verbose:
            print ("Constructed BaggingClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, bc_params, nfolds, n_jobs, scoring,random_grid, n_iter,  verbose)
        self._model = self.constructClassifier(BaggingClassifier(), self._random_grid)
        
       
