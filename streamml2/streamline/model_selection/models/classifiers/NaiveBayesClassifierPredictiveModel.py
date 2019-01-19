import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.naive_bayes import GaussianNB 

class NaiveBayesClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, nbc_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="nbc"
        
        if verbose:
            print ("Constructed NaiveBayesClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, nbc_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructClassifier(GaussianNB())
        
       
