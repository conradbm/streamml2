import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.linear_model import SGDClassifier

class StochasticGradientDescentClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, sgdc_params, nfolds=3, n_jobs=1, scoring=None, random_grid=False, n_iter=10, verbose=True):
        
        self._code="sgdc"
        
        if verbose:
            print ("Constructed StochasticGradientDescentClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, sgdc_params, nfolds, n_jobs, scoring, random_grid, n_iter, verbose)
        self._model = self.constructClassifier(SGDClassifier(), self._random_grid)
        
       
