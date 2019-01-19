import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.svm import SVC

class SupportVectorClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, svc_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="svc"
        
        if verbose:
            print ("Constructed SupportVectorClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, svc_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructClassifier(SVC())
        
       
