from ..AbstractTransformer import *
from sklearn.preprocessing import normalize

class NormalizeTransformer(AbstractTransformer):
    
    def __init__(self):
        AbstractTransformer.__init__(self, "normalize")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        assert(isinstance(X, pd.DataFrame), "please ensure X is of type pd.DataFrame")
        columns=list(X.columns)
        return pd.DataFrame(normalize(X, norm='l2'), columns=columns)