import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.ensemble import AdaBoostClassifier

class AdaptiveBoostingClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, abc_params, nfolds=3, n_jobs=1, scoring=None,random_grid=False, n_iter=10,  verbose=True):
        
        self._code="abc"
        
        if verbose:
            print ("Constructed AdaptiveBoostingClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, abc_params, nfolds, n_jobs, scoring,random_grid, n_iter,  verbose)
        self._model = self.constructClassifier(AdaBoostClassifier(), self._random_grid)
        
       
