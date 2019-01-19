import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.neighbors import KNeighborsRegressor

class KNNRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, knnr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="knnr"
        
        if verbose:
            print ("Constructed KNeighborsRegressorRegressorPredictiveModel: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, knnr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(KNeighborsRegressor())
        
    
    #methods
    def execute(self):
        pass
