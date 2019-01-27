import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.linear_model import Perceptron

class PerceptronClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, pc_params, nfolds=3, n_jobs=1, scoring=None,random_grid=False, n_iter=10,  verbose=True):
        
        self._code="pc"
        
        if verbose:
            print ("Constructed PerceptronClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, pc_params, nfolds, n_jobs, scoring,random_grid, n_iter,  verbose)
        self._model = self.constructClassifier(Perceptron(), self._random_grid)
        
       
