import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.svm import SVR

class SupportVectorRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, svr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        

        self._code="svr"
        
        if verbose:
            print ("Constructed SupportVectorRegressorPredictiveModel: "+self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, svr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(SVR())
        
    
    #methods
    def execute(self):
        pass
