import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import LinearRegression

class LinearRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, lr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):

        self._code="lr"
        
        if verbose:
            print ("Constructed LinearRegressorPredictiveModel: "+self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, lr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(LinearRegression())
        
    
    #methods
    def execute(self):
        pass
