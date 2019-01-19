from ..AbstractTransformer import *
from sklearn.neural_network import BernoulliRBM


class BernoulliRBMTransformer(AbstractTransformer):
    
    def __init__(self):
        AbstractTransformer.__init__(self, "brbm")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        brbm = BernoulliRBM(n_components=256, 
                            learning_rate=0.1,
                            batch_size=10,
                            n_iter=10, 
                            verbose=0, 
                            random_state=None)
           
        return pd.DataFrame(brbm.fit_transform(X))