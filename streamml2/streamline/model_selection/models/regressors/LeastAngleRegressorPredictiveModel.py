import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import Lars

class LeastAngleRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, lar_params, nfolds=3, n_jobs=1, scoring=None, random_grid=False, n_iter=10,verbose=True):
        
        self._code="lar"
        
        if verbose:
            print ("Constructed Lars: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, lar_params, nfolds, n_jobs, scoring, random_grid, n_iter,verbose)
        self._model = self.constructRegressor(Lars(), self._random_grid)
        
    
    #methods
    def execute(self):
        pass
       
