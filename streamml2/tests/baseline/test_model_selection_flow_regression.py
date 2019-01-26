"""
#
#
#
#
# Model Selection Example (Regression)
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
             cut=None,
             random_grid=False,
             n_iter=10):

"""

import pandas as pd
import numpy as np
from streamml2_test.streams import ModelSelectionStream
from streamml2_test.streamml2.utils.helpers import *
from sklearn.datasets import load_boston

boston=load_boston()
X=pd.DataFrame(boston['data'], columns=boston['feature_names'])
y=pd.DataFrame(boston['target'],columns=["target"])

models=get_model_selection_regressors()
params=get_model_selection_regressors_params()
results_dict = ModelSelectionStream(X,y).flow(models,
                                                params=params,
                                                regressors=True,
                                                nfolds=3,
                                                nrepeats=3,
                                                n_jobs=3,
                                                random_grid=True,
                                                n_iter=2,
                                                model_selection=True)

for k in results_dict.keys():
    print(k)
    print(results_dict[k])