
import pandas as pd
import numpy as np

# FOR ANY SYSTEM: INCLUDE STREAMML
import sys
sys.path.append('/Users/bmc/Desktop/') #I.e., make it a path variable

from streamml.streamline.transformation.flow.TransformationStream import TransformationStream
from streamml.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream
from streamml.streamline.feature_selection.flow.FeatureSelectionStream import FeatureSelectionStream

# git checkout -b modelSelectionUpdates
# git push -u origin modelSelectionUpdates

# FOR MAC:
# nano ~/.bash_profile
# export PYTHONPATH="${PYTHONPATH}:/Users/bmc/Desktop/streamml"
# source ~/.bash_profile
# python -W ignore tester.py

X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))




#print(X.shape)
#print (y.shape)

"""
Transformation Options:
["scale","normalize","boxcox","binarize", "brbm", "pca", "tsne", "kmeans"]
kmeans: n_clusters
pca: percent_variance (only keeps # comps that capture this %)
binarize: threshold (binarizes those less than threshold as 0 and above as 1)
tsne: n_components
brbm: standard sklearn parameters
# sklearn.decomposition.sparse_encode
# sklearn.preprocessing.PolynomialFeatures
"""


Xnew = TransformationStream(X).flow(["pca"], 
                                    params={"pca__percent_variance":0.1},
                                   verbose=True)
print(Xnew)

Xnew = TransformationStream(X).flow(["tsne"], 
                                    params={"tsne__n_components":2},
                                   verbose=True)
print(Xnew)

Xnew = TransformationStream(X).flow(["pca","tsne"], 
                                    params={"pca__percent_variance":.5,
                                            "tsne__n_components":2},
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["pca","tsne"], 
                                    params={"pca__percent_variance":.5,
                                            "tsne__n_components":3},
                                   verbose=True)
print(Xnew)

Xnew = TransformationStream(X).flow(["scale"],
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["normalize"],
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["boxcox"],
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["binarize"],
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["brbm"],
                                    params={"brbm__learning_rate":0.01},
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["kmeans"],
                                    params={"kmeans__n_clusters":5},
                                   verbose=False)
print(Xnew)

Xnew = TransformationStream(X).flow(["boxcox", "scale", "normalize","binarize","brbm"],
                                    params={"brbm__learning_rate":0.01},
                                   verbose=False)
print(Xnew)