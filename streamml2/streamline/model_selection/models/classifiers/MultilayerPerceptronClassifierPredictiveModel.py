import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.neural_network import MLPClassifier

class MultilayerPerceptronClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, mlpc_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="mlpc"
        
        if verbose:
            print ("Constructed MultilayerPerceptronClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, mlpc_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructClassifier(MLPClassifier())
        
       
