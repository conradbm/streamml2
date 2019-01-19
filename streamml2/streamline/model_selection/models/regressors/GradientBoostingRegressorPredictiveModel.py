import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.ensemble import GradientBoostingRegressor

class GradientBoostingRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, gbr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="gbr"
        
        if verbose:
            print ("Constructed GradientBoostingRegressor: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, gbr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(GradientBoostingRegressor())
        
    
    #methods
    def execute(self):
        pass
       
