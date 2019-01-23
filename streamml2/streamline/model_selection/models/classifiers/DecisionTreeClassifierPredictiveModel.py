import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, dtc_params, nfolds=3, n_jobs=1, scoring=None, random_grid=False, n_iter=10, verbose=True):
        
        self._code="dtc"
        
        if verbose:
            print ("Constructed DecisionTreeClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, dtc_params, nfolds, n_jobs, scoring,random_grid, n_iter,  verbose)
        self._model = self.constructClassifier(DecisionTreeClassifier(), self._random_grid)
        
       
