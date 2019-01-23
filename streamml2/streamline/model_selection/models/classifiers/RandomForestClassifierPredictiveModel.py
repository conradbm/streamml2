import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, rfc_params, nfolds=3, n_jobs=1, scoring=None, random_grid=False, n_iter=10,  verbose=True):
        
        self._code="rfc"
        
        if verbose:
            print ("Constructed RandomForestClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, rfc_params, nfolds, n_jobs, scoring, random_grid, n_iter, verbose)
        self._model = self.constructClassifier(RandomForestClassifier(), self._random_grid)
        
       
