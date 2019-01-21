from sklearn.cross_decomposition import PLSRegression
from ...AbstractFeatureSelectionModel import AbstractFeatureSelectionModel
class PLSRegressorFeatureSelectionModel(AbstractFeatureSelectionModel):

    
    def __init__(self, X, y, params, verbose):
        AbstractFeatureSelectionModel.__init__(self,"plsr", X, y, params, verbose)

    def execute(self):

        super(PLSRegressorFeatureSelectionModel, self).execute()

        # The components are not helpful for this context. They might be for transformation, however.
        #if "plsr__n_components" in self._allParams.keys():
        #  n_components = self._allParams["plsr__n_components"]
        #else:
        #  n_components = 2
        pls_model = PLSRegression()
        pls_out = pls_model.fit(self._X, self._y)

        # The coefficients are used to show direction of the relationship
        return abs(pls_out.coef_.flatten())