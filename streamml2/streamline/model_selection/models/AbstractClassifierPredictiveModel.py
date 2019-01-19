
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from ..AbstractPredictiveModel import *

class AbstractClassifierPredictiveModel(AbstractPredictiveModel):

    #constructor
    _options = ["auc",
                 "precision",
                 "recall",
                 "f1",
                 "accuracy",
                 "kappa",
                 "log_loss"
                ]
    
    
    _validation_results=None
    
    def __init__(self, modelType, X, y, params, nfolds, n_jobs, scoring, verbose):
        
        #if self._verbose:
        #    print("Constructed AbstractClassifierPredictiveModel: "+self._code)
        
        assert modelType == "classifier", "You are creating a classifier, but have not specified it to be one."
        #assert any([isinstance(y.dtypes[0],float),isinstance(y.dtypes[0],float)]), "Your response variable y is not a float."

        self._modelType = modelType
        self._y=y
        self._scoring=scoring
        AbstractPredictiveModel.__init__(self, X, params, nfolds, n_jobs, verbose)
        
    #methods
    def validate(self, Xtest, ytest, metrics, verbose=False):
        assert any([isinstance(metrics, str), isinstance(metrics, list)]), "Your classifier error metric must be a str or list"
        assert all([i in self._options for i in metrics]) , "Your clasifier error metric must be in valid: " + ' '.join([i for i in self._options])

        
        self._validation_results={}
        for m in metrics:
            if m == "auc":
                ypred = self._model.predict(Xtest)
                self._validation_results["auc"]=roc_auc_score(ytest, ypred, average="macro")
            elif m == "precision":
                ypred = self._model.predict(Xtest)
                self._validation_results["precision"]=precision_score(ytest, ypred, average="macro")
            elif m == "recall":
                ypred = self._model.predict(Xtest)
                self._validation_results["recall"]=recall_score(ytest, ypred, average="macro")
            elif m == "f1":
                ypred = self._model.predict(Xtest)
                self._validation_results["f1"]=f1_score(ytest, ypred, average="macro")
            elif m == "accuracy":
                ypred = self._model.predict(Xtest)
                self._validation_results["accuracy"]=accuracy_score(ytest, ypred)
            elif m == "kappa":
                ypred = self._model.predict(Xtest)
                self._validation_results["kappa"]=cohen_kappa_score(ytest,ypred)
            # Not working for multi-label classifiers.
            elif m == 'log_loss':
                print("Currently not supported: log_loss")
                raise(Exception)
            #     ypred = self._model.predict(Xtest)
            #     
            #     self._validation_results["log_loss"]=log_loss(ytest,ypred, labels=list(set(ypred).union(set(ytest['target']))))
            else:
                print(str(m)+" not a valid classifier metric, skipping.")
        
        return self._validation_results
    
    def constructClassifier(self, model):
        self._pipe          = Pipeline([(self._code, model)])

        self._grid          = GridSearchCV(self._pipe,
                                            param_grid=self._params, 
                                            n_jobs=self._n_jobs,
                                            cv=self._nfolds, 
                                            verbose=False)
        
        
        self._model                 = self._grid.fit(self._X,self._y).best_estimator_.named_steps[self._code]
        return self._model   

    def getValidationResults(self):
        return self._validation_results