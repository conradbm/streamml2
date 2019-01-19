import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.ensemble import GradientBoostingClassifier 

class GradientBoostingClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, gbc_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="gbc"
        
        if verbose:
            print ("Constructed GradientBoostingClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, gbc_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructClassifier(GradientBoostingClassifier ())
        
       
