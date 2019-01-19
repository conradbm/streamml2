from ..AbstractTransformer import *
from sklearn.cluster import KMeans

class KMeansTransformer(AbstractTransformer):
    
    def __init__(self, nclusters):
        self._n_clusters = nclusters
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        kmeans = KMeans(n_clusters=self._n_clusters).fit(X)
        X['cluster'] = pd.DataFrame(kmeans.labels_, columns=['cluster'], dtype='category')
        return X