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
from streamml2_test.streamml2.utils.helpers import *

from sklearn.datasets import load_iris

iris=load_iris()
X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
y=pd.DataFrame(iris['target'], columns=['target'])

models=get_model_selection_classifiers()
params=get_model_selection_classifiers_params()
results_dict = ModelSelectionStream(X,y).flow(models,
                                                    params=params,
                                                    metrics=[],
                                                    test_size=0.33,
                                                    nfolds=10,
                                                    nrepeats=3,
                                                    verbose=True, 
                                                    regressors=False,
                                                    stratified=True,
                                                    stats=False,
                                                    modelSelection=False,
                                                    random_grid=True,
                                                    n_jobs=3)

for k in results_dict.keys():
    print(k)
    print(results_dict[k])
