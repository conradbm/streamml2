import pandas as pd
from ..AbstractTransformer import *
from sklearn.preprocessing import PolynomialFeatures

class PolynomialTransformer(AbstractTransformer):
    
    def __init__(self, degree, interaction_only):
        self._degree = degree
        self.interaction_only=interaction_only
        AbstractTransformer.__init__(self, "poly")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        assert(isinstance(X, pd.DataFrame), "please ensure X is of type pd.DataFrame")
        poly = PolynomialFeatures(degree=self._degree, 
                                  interaction_only=self.interaction_only)
        self._powers=poly.fit(X).powers_
        
        names=[]
        for row in range(self._powers.shape[0]):
            names.append("_".join([str(X.columns[i])+"^"+str(self._powers[row,i]) 
                                    for i in range(len(self._powers[row,:]))]))
        X_tranformed=poly.fit_transform(X)
        
        return pd.DataFrame(X_tranformed, columns=names)