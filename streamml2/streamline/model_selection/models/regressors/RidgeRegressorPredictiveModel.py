import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import Ridge

class RidgeRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, ridge_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):

        self._code="ridge"
        
        if verbose:
            print ("Constructed RidgeRegressorPredictiveModel: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, ridge_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(Ridge())
        
    
    #methods
    def execute(self):
        pass
    
