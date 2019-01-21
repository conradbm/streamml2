import pandas as pd
from statsmodels.regression import linear_model
import statsmodels.api as sm
from ...AbstractFeatureSelectionModel import AbstractFeatureSelectionModel
class MixedSelectionFeatureSelectionModel(AbstractFeatureSelectionModel):

    def __init__(self, X, y, params, verbose):
        AbstractFeatureSelectionModel.__init__(self,"plsr", X, y, params, verbose)

    def execute(self):

        super(MixedSelectionFeatureSelectionModel, self).execute()
            
        X = self._X
        y = self._y
        
        initial_list=[]
        threshold_in_specified = False
        threshold_out_specified = False

        if "mixed_selection__threshold_in" in self._params.keys():
          assert isinstance(self._params["mixed_selection__threshold_in"], float), "threshold_in must be a float"
          threshold_in = self._params["mixed_selection__threshold_in"]
          threshold_in_specified=True
        else:
          threshold_in=0.01

        if "mixed_selection__threshold_out" in self._params.keys():
          assert isinstance(self._params["mixed_selection__threshold_out"], float), "threshold_out must be a float"
          threshold_out = self._params["mixed_selection__threshold_out"]
          threshold_out_specified=True
        else:
          threshold_out = 0.05

        if "mixed_selection__verbose" in self._params.keys():
          assert isinstance(self._params["mixed_selection__verbose"], bool), "verbose must be a bool"
          verbose = self._params["mixed_selection__verbose"]
        else:
          verbose = False
        
        if threshold_in_specified and threshold_out_specified:
          assert threshold_in < threshold_out, "threshold in must be strictly less than the threshold out to avoid infinite looping."


        #initial_list = self._initial_list
        #threshold_in = self._threshold_in
        #threshold_out = self._threshold_out
        #verbse = self._verbose
        
        """ Perform a forward-backward feature selection 
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features 
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        """
                  
        included = list(initial_list)
        while True:
            changed=False
            
            # forward step
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded)

            for new_column in excluded:

                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]

            best_pval = new_pval.min()

            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                #best_feature = new_pval.argmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Adding  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
            if worst_pval > threshold_out:
                changed=True
                worst_feature = pvalues.idxmax()
                #worst_feature = pvalues.argmax()
                included.remove(worst_feature)
                if verbose:
                    print('Dropping {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

            if not changed:
                break

        new_included = []
        for col in X.columns:
            if col in included:
              new_included.append(1)
            else:
              new_included.append(0)
        
        return new_included