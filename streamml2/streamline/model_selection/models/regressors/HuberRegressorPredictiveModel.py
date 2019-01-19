import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import HuberRegressor

class HuberRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, hr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="hr"
        
        if verbose:
            print ("Constructed HuberRegressor: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, hr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(HuberRegressor())
        
    
    #methods
    def execute(self):
        pass
       
