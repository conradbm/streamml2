import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.neural_network import MLPRegressor

class MultilayerPerceptronRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, mlpr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="mlpr"
        
        if verbose:
            print ("Constructed MultilayerPerceptronRegressor: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, mlpr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(MLPRegressor())
        
    
    #methods
    def execute(self):
        pass