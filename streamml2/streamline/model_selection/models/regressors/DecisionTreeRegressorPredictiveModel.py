import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.tree import DecisionTreeRegressor

class DecisionTreeRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, dtr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="dtr"
        
        if verbose:
            print ("Constructed DecisionTreeRegressor: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, dtr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(DecisionTreeRegressor())
        
    
    #methods
    def execute(self):
        pass
       
