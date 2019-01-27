from ..AbstractTransformer import *
from sklearn.manifold import TSNE

class TSNETransformer(AbstractTransformer):
    
    def __init__(self, ncomps):
        self._tsne_n_components = ncomps
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        assert(isinstance(X, pd.DataFrame), "please ensure X is of type pd.DataFrame")
        X_embedded = TSNE(n_components=self._tsne_n_components).fit_transform(X)
        cols=X_embedded.shape[1]
        columns=["embedding_"+str(i) for i in range(cols)]
        return pd.DataFrame(X_embedded, columns=columns)