from ..AbstractTransformer import *
from sklearn.preprocessing import Binarizer

class BinarizeTransformer(AbstractTransformer):
    
    def __init__(self, threshold):
        self._threshold=threshold
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        assert(isinstance(X, pd.DataFrame), "please ensure X is of type pd.DataFrame")
        columns=list(X.columns)
        return pd.DataFrame(Binarizer(threshold=self._threshold).fit(X).transform(X), columns=columns)