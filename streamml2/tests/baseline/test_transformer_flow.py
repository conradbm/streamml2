#
#
#
#
# Transformation Flow Example
# test_transformation_flow.py
#
#
#
#
#
#
"""
Transformation Selection Params:
    def flow(self, preproc_args=[], params=None, verbose=False):

Transformation Selection Options:
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

kmeans, pca, and tsne have standard sklearn parameters available.
pca has the additional  pca__percent_variance option which keeps only the components representing the specified percentage of variance given.
"""

import pandas as pd
import numpy as np
import os
import sys
#sys.path.append("C:\\Users\\bmccs\\Desktop\\streamml2")
# must change the package name or it will reference your installed PyPi version of streamml2
from streamml2_test.streamml2.streamline.transformation.flow.TransformationStream import TransformationStream
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)

X2 = TransformationStream(newsgroups_train.data,corpus=True, method='tfidf').flow(["pca","normalize","binarize","kmeans"],
                                                              params={"pca__percent_variance":0.95,
                                                              "kmeans__n_clusters":len(categories)},
                                                              verbose=False)
print(X2)
print("~~~~~~~~~")
from sklearn.datasets import load_iris
iris=load_iris()
X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
y=pd.DataFrame(iris['target'], columns=['target'])

X2 = TransformationStream(X).flow(["pca","normalize","binarize","kmeans"],
                                  params={"pca__n_components":2,
                                          "binarize__threshold":0.0,
                                          "kmeans__n_clusters":len(set(y['target']))
                                          },
                                          verbose=True)
print(X2)