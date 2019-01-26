"""
#
#
#
#
# Model Selection Example (Classification)
# test_model_selection_flow.py
#
#
#
#
#
#

Model Selection Params:
    def flow(self, 
             models_to_flow=[], 
             params=None, 
             test_size=0.2, 
             nfolds=3, 
             nrepeats=3,
             pos_split=1,
             n_jobs=1, 
             metrics=[], 
             verbose=False, 
             regressors=True,
             modelSelection=False,
             cut=None):

"""

import pandas as pd
import numpy as np
from streamml2_test.streams import ModelSelectionStream
from streamml2_test.streams import TransformationStream
from streamml2_test.streamml2.utils.helpers import *
from sklearn.datasets import load_iris

iris=load_iris()
X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
y=pd.DataFrame(iris['target'], columns=['target'])
X2=TransformationStream(X).flow(["scale","normalize"])
models=get_model_selection_classifiers()
params=get_model_selection_classifiers_params()
results_dict = ModelSelectionStream(X2,y).flow(["abc","gbc"],
                                                params=params,
                                                regressors=False,
                                                verbose=True)

for k in results_dict.keys():
    print(k)
    print(results_dict[k])
