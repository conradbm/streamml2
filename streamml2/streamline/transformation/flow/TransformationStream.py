from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


from ..transformers.ScaleTransformer import ScaleTransformer
from ..transformers.BernoulliRBMTransformer import BernoulliRBMTransformer
from ..transformers.BinarizeTransformer import BinarizeTransformer
from ..transformers.KMeansTransformer import KMeansTransformer
from ..transformers.TSNETransformer import TSNETransformer
from ..transformers.NormalizeTransformer import NormalizeTransformer
from ..transformers.PCATransformer import PCATransformer
from ..transformers.BoxcoxTransformer import BoxcoxTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
"""
Example Usage:

import pandas as pd
import numpy as np
from streamml.streamline.transformation.TransformationStream import TransformationStream

test = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(20)]))

t = TransformationStream(test)
formed = t.flow(["scale","normalize","pca","binarize","kmeans"], params={"pca__percent_variance":0.75,
"kmeans__n_clusters":3})
print formed

formed = t.flow(["scale","normalize", "pca"], params={"pca__percent_variance":0.75,
"kmeans__n_clusters":3})
print formed

formed = t.flow(["boxcox","scale","normalize"], verbose=True)
print formed
"""

class TransformationStream:
    
    _vectorizer=None
    """
    Constructor:
    1. Default
        Paramters: df : pd.DataFrame, dataframe must be accepted to use this class
    """
    def __init__(self, df=None, corpus=False, method='tfidf',min_df=0.01, max_df=0.99, n_features=10000):
        
        self._corpus_options=['tfidf','count','hash']
        if corpus == False:
            assert isinstance(df, pd.DataFrame), "data must be a pandas DataFrame"
            self._X = df
        else:
            assert isinstance(df, list), "data must be a list of strings when corpus is true"
            assert method in self._corpus_options, "method must be in corpus_options: " + " ".join(self._corpus_options)
            
            if method == 'tfidf':
                self._vectorizer=TfidfVectorizer(min_df=min_df, max_df=max_df).fit(df)
                self._vocabulary=self._vectorizer.vocabulary_
                tmp=self._vectorizer.fit_transform(df)
                self._X=pd.DataFrame(tmp.todense(), columns=self._vocabulary)
            elif method == 'count':
                self._vectorizer=CountVectorizer(min_df=min_df, max_df=max_df).fit(df)
                self._vocabulary=self._vectorizer.vocabulary_
                tmp=self._vectorizer.fit_transform(df)
                self._X=pd.DataFrame(tmp.todense(), columns=self._vocabulary)
            elif method == 'hash':

                tmp=HashingVectorizer(n_features=n_features).fit_transform(df)
                self._X=pd.DataFrame(tmp.todense())
            else:
                print("Error: method specified not in list of vectorizers\n")
                sys.exit()

    """
    Methods:
    1. flow
        Parameters: preproc_args : list(), User specified preprocessing techniques. 
                    These can include any of the following:
                    ["scale","normalize","boxcox","binarize","pca","tnse","kmeans"]. 
                    Restrictions: 1. All dtyles must be float. 
                                  2. boxcox demands a positive matrix. 
                                  3. kmeans must be executed last.
                    MetaParamters:
                        - scale -> None
                        - normalize -> None
                        - boxcox -> None
                        - binarize -> threshold : float, will binarize 1 if >= threshold, 0 else
                        - pca -> percent_variance : float, will only return the number of components needed for this.
                        - kmeans -> n_clusters : integer, will cluster with this number.
                        - brbm -> None
                        - tnse -> n_components : integer, how many dimensions to keep
                    
                    
    """
    def flow(self, preproc_args=[], params=None, verbose=False):
        
        # Assert correct formatting
        assert isinstance(self._X, pd.DataFrame), "data must be a pandas DataFrame."
        
        # Set verbosity.
        self._verbose=verbose
        
        # Enforce PCA parameters
        if "pca" in preproc_args:
            if "pca__percent_variance" in params.keys():
                assert isinstance(params["pca__percent_variance"], float), "percent variance must be float."
                self._percent_variance=params["pca__percent_variance"]
                print ("custom: pca__percent_variance="+str(self._percent_variance) )
            else:
                self._percent_variance=0.90
                print ("default: pca__percent_variance="+str(self._percent_variance) )

            if "pca__n_components" in params.keys():
                assert isinstance(params["pca__n_components"], int), "number of components must be int."
                self._pca_n_components=params["pca__n_components"]
                print ("custom: pca__n_components="+str(self._pca_n_components) )
            else:
                self._pca_n_components=len(self._X.columns.tolist())
                print ("default: pca__n_components="+str(self._pca_n_components) )

        # Enforce TSNE parameters
        if "tsne" in preproc_args:
            if "tsne__n_components" in params.keys():
                assert isinstance(params["tsne__n_components"], int), "n_components must be integer."
                self._tsne_n_components=params["tsne__n_components"]
                print ("custom: tsne__n_components"+str(self._tsne_n_components) )
            else:
                self._tsne_n_components=3
                print ("default: _tsne_n_components= "+str(self._tsne_n_components) )
            
        # Enforce Kmeans parameters
        if "kmeans" in preproc_args:
            if "kmeans__n_clusters" in params.keys():
                assert isinstance(params["kmeans__n_clusters"], int), "clusters must be integer."
                self._n_clusters=params["kmeans__n_clusters"]
                print ("custom: kmeans__n_clusters="+str(self._n_clusters))
            else:
                self._n_clusters=2
                print ("default: kmeans__n_clusters="+str(self._n_clusters))
        
        # Enforce Binarize parameters
        if "binarize" in preproc_args:
            if "binarize__threshold" in params.keys():
                assert isinstance(params["binarize__threshold"], float), "threshold must be float."
                self._threshold=params["binarize__threshold"]
                print ("default: binarize__threshold="+str(self._threshold))
            else:
                self._threshold=0.0
                print ("default: binarize__threshold="+str(self._threshold))

        if "brbm" in preproc_args:
            if "brbm__n_components" in params.keys():
                assert isinstance(params["brbm__n_components"], int), "n_components must be integer."
                self._n_components = params["brbm__n_components"]
            elif "brbm__learning_rate" in params.keys():
                assert isinstance(params["brbm__learning_rate"], float), "learning_rate must be a float"
                self._learning_rate = params["brbm__learning_rate"]
        


        # Inform the streamline to user.
        stringbuilder=""
        for thing in preproc_args:
            stringbuilder += thing
            stringbuilder += "--> "

        print("**************************************************")
        print("Transformation Streamline: " + stringbuilder[:-4])
        print("**************************************************")
        
        
        # Define helper functions to execute our transformation streamline
        
        # Implemented
        def runScale(X, verbose=False):
            if verbose:
                print ("Executing Scaling")
                       
            return ScaleTransformer().transform(X)

        # Implemented
        def runNormalize(X, verbose=False):
            if verbose:
                print ("Executing Normalize")
            # More parameters can be found here: 
            # http://scikit-learn.org/stable/modules/preprocessing.html
            
            
            
            #X_normalized = preprocessing.normalize(X, norm='l2')
            #return pd.DataFrame(X_normalized)
            return NormalizeTransformer().transform(X)
        
        # Implemented
        def runBinarize(X, verbose=False):
            if verbose:
                print ("Executing Binarization")
            # More parameters can be found here: 
            # http://scikit-learn.org/stable/modules/preprocessing.html
            
            #X_binarized = preprocessing.Binarizer(threshold=self._threshold).fit(X).transform(X)
            #return pd.DataFrame(X_binarized)
            return BinarizeTransformer(self._threshold).transform(X)
        
        # Implemented | NOTE: Only works on positive data
        def runBoxcox(X, verbose=False):
            if verbose:
                print ("Executing Boxcox")
            # More parameters can be found here: 
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html

            bct = BoxcoxTransformer()
            X_boxcoxed = bct.transform(X)
            self._lambdas = bct._lambdas
            return X_boxcoxed
        
        # Implemented
        def runPCA(X, verbose=False):
            if verbose:
                print ("Executing PCA")
               
			# More parameters can be found here: 
            # http://scikit-learn.org/stable/modules/preprocessing.html
            """
            pca = PCA()
            
            pca_output = pca.fit(X)
            components_nums = [i for i in range(len(pca_output.components_))]
            self._percentVarianceExplained = pca_output.explained_variance_/sum(pca_output.explained_variance_)
            
            tot=0.0
            idx=0
            for i in range(len(self._percentVarianceExplained)):
                tot+= self._percentVarianceExplained[i]
                if tot >= self._percent_variance:
                    idx=i+1
                    break
            if verbose:
                print ("Percent of Variance Explained By Components:\n")
                print (str(self._percentVarianceExplained), "\n")
                print (str(self._percent_variance*100), "% variance is explained in the first ", str(idx), " components\n")
                pca_df=pd.DataFrame({'explained_variance':pca_output.explained_variance_}, index=components_nums)
                pca_df.plot(title='Components vs. Variance')
                plt.show()
            
            pca_output = pca.fit_transform(X)
            pca_df = pd.DataFrame(pca_output, columns=["PC_"+str(i) for i in components_nums])
            
            return pca_df.iloc[:, :idx]
            """
            return PCATransformer(self._percent_variance, self._pca_n_components, self._verbose).transform(X)
        
        # Implemented
        def runKmeans(X, verbose=False):
            if verbose:
                print ("Executing Kmeans with " + str(self._n_clusters) + " clusters\n")
            
			# More parameters can be found here: 
            # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            #kmeans = KMeans(n_clusters=self._n_clusters).fit(X)
            #X['cluster'] = pd.DataFrame(kmeans.labels_, columns=['cluster'], dtype='category')
            #return X
            return KMeansTransformer(self._n_clusters).transform(X)
        
        def runBRBM(X, verbose=False):
            if verbose:
                print ("Executing Bernoulli Restricted Boltzman Machine\n")
            
            
            #brbm = BernoulliRBM(n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None)
            #Xnew = pd.DataFrame(brbm.fit_transform(X))
            
            #return Xnew
            return BernoulliRBMTransformer().transform(X)

        def runTSNE(X, verbose=False):
            if verbose:
                print("Executing TNSE with " + str(self._tsne_n_components) + " components\n")
            
            #X_embedded = TSNE(n_components=self._tsne_n_components).fit_transform(X)
            #return pd.DataFrame(X_embedded)
            return TSNETransformer(self._tsne_n_components).transform(X)
        
        # Unimplemented
        def runItemset(X, verbose=False):
            if verbose:
                print ("Itemset mining unimplemented\n")
            return X

        # map the inputs to the function blocks
        options = {"scale" : runScale,
                   "normalize" : runNormalize,
                   "binarize" :runBinarize,
                   "itemset": runItemset,
                   "boxcox" : runBoxcox,
                   "pca" : runPCA,
                   "kmeans" : runKmeans,
                  "brbm": runBRBM,
                  "tsne":runTSNE}
        
        # Execute commands as provided in the preproc_args list
        self._df_transformed = self._X
        for key in preproc_args:
            self._df_transformed = options[key](self._df_transformed, verbose=self._verbose)

        # Return transformed data
        return self._df_transformed

