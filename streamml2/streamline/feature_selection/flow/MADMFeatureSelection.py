from skcriteria import Data, MAX
from skcriteria.madm import closeness, simple

import pandas as pd
import math
import numpy as np

class MADMFeatureSelection():

    def __init__(self, X, y, keyFeatures, featurePercentage, verbose):
        self._X=X
        self._y=y
        self._key_features=keyFeatures
        self._featurePercentage=featurePercentage
        self._verbose=verbose

    def execute(self):

        alternative_names = self._X.columns.tolist()
        criterion_names = list(self._key_features.keys())
        criteria = [MAX for i in criterion_names]
        weights = [i/len(criterion_names) for i in range(len(criterion_names))]

        df = pd.DataFrame(self._key_features,
                          index=alternative_names)
 
        data = Data(df.as_matrix(),
                    criteria,
                    weights,
                    anames=df.index.tolist(),
                    cnames=df.columns
                    )
        #if self._verbose:
          #data.plot("radar");

        dm1 = simple.WeightedSum()
        dm2 = simple.WeightedProduct()
        dm3 = closeness.TOPSIS()
        dec1 = dm1.decide(data)
        dec2 = dm2.decide(data)
        dec3 = dm3.decide(data)

        ranks=[dec1.rank_, dec2.rank_,dec3.rank_]
        self._ensemble_results = pd.DataFrame({"TOPSIS":dec3.rank_,
                                              "WeightedSum":dec1.rank_,
                                              "WeightedProduct":dec2.rank_},
                                              index=df.index.tolist())
        
        # Only keep features that our decision makers deemed in the top % specified
        num_features_requested=math.ceil(len(alternative_names)*self._featurePercentage)
        sum_ranks=sum(ranks)
        argmin_sorted=np.argpartition(sum_ranks, num_features_requested)
        self._kept_features=[]
        
        count=0
        for i in argmin_sorted:
            self._kept_features.append(alternative_names[i])
            count+=1
            if count >= num_features_requested:
                break
          
        if self._verbose:
            print("",self._featurePercentage*100, " % -> ("+str(num_features_requested)+") features kept.")
            print(self._kept_features)
        
        return self._ensemble_results, self._kept_features