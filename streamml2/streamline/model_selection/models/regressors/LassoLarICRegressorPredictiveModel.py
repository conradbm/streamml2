from ..AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import LassoLarsIC

class LassoLarICRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, lasso_lar_ic_params, nfolds=3, n_jobs=1, scoring=None, random_grid=False, n_iter=10, verbose=True):
        
        self._code="lasso_lar_ic"
        
        if verbose:
            print ("Constructed LassoLarICRegressorPredictiveModel: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, lasso_lar_ic_params, nfolds, n_jobs, scoring, random_grid, n_iter,verbose)
        self._model = self.constructRegressor(LassoLarsIC(), self._random_grid)
        
    
    #methods
    def execute(self):
        pass
       
