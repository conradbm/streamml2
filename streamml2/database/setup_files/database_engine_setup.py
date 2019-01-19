"""
********************************************************
*** Set up and import the actual data for our tables ***
********************************************************

************************
*** Table Structures ***
************************

class FeatureSelector(Base):
    __tablename__ = 'T_FeatureSelector'
    F_FeatureSelector_ID = Column(Integer, primary_key=True)
    F_Estimator_ID = Column(Integer, ForeignKey('T_Estimator.F_Estimator_ID'), nullable=True)
    F_FeatureSelector_Name = Column(String(100), nullable=False)
    F_FeatureSelector_HasCoef = Column(Integer, nullable=False) # 1,0 if the model has coef attribute
    F_FeatureSelector_HasFeatureImportance = Column(Integer, nullable=False) # 1,0 if the model has feature_importance attribute
    F_FeatureSelector_PredictionClass = Column(String(20), nullable=False) # regressor or classifier
    
    #Relationship From
    F_Estimator = relationship(Estimator)

class FeatureSelectorParameter(Base):
    __tablename__ = 'T_FeatureSelectorParameter'
    F_FeatureSelectorParameter_ID = Column(Integer, primary_key=True)
    F_FeatureSelector_ID = Column(Integer, ForeignKey('T_FeatureSelector.F_FeatureSelector_ID'))
    F_FeatureSelectorParameter_Open= Column(Integer, nullable=True)
    F_FeatureSelectorParameter_Name = Column(String(20), nullable=True)
    F_FeatureSelectorParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_FeatureSelector = relationship(FeatureSelector)
    


# When user selects transformer then chooses values for each parameter to go with it
class FeatureSelectorParameterValue(Base):
    #F_ParameterValue_ID | PK
    #F_ParameterValue_Realization | Char(20); Actual value the user selected for the parameter
    
    __tablename__ = 'T_FeatureSelectorParameterValue'
    F_FeatureSelectorParameterValue_ID = Column(Integer, primary_key=True)
    F_FeatureSelectorParameter_ID = Column(Integer , ForeignKey('T_FeatureSelectorParameter.F_FeatureSelectorParameter_ID'))
    F_FeatureSelectorParameterValue_Realization = Column(String(10), nullable=False)
    
     #Relationship From
    F_EstimatorParameter = relationship(FeatureSelectorParameter)
    
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base
from database_setup import Estimator, EstimatorParameter, EstimatorParameterValue
from database_setup import Transformer, TransformerParameter, TransformerParameterValue
from database_setup import NonEstimatorFeatureSelector, NonEstimatorFeatureSelectorParameter, NonEstimatorFeatureSelectorParameter


# Re-create the database
engine = create_engine('sqlite:///streamml.db')


# Relate Tables to DB
Base.metadata.bind = engine

# SQL Session Wrapper
DBSession = sessionmaker(bind=engine)
session = DBSession()


"""
# Set up transformers
options = {"scale" : runScale,
           "normalize" : runNormalize,
           "binarize" :runBinarize,
           "itemset": runItemset,
           "boxcox" : runBoxcox,
           "pca" : runPCA,
           "kmeans" : runKmeans,
          "brbm": runBRBM,
          "tsne":runTSNE}
regression_options = {"mixed_selection" : mixed_selection,
                               "svr" : supportVectorRegression,
                               "rfr":randomForestRegression,
                               "abr":adaptiveBoostingRegression,
                               "lasso":lassoRegression,
                               "enet":elasticNetRegression,
                               "plsr":partialLeastSquaresRegression}



        # Valid classifiers
        classification_options = {'abc':adaptiveBoostingClassifier,
                                    'rfc':randomForestClassifier,
                                    'svc':supportVectorClassifier
                                 }

class FeatureSelector(Base):
    __tablename__ = 'T_FeatureSelector'
    F_FeatureSelector_ID = Column(Integer, primary_key=True)
    F_FeatureSelector_Name = Column(String(100), nullable=False)
    F_FeatureSelector_HasCoef = Column(Integer, nullable=False) # 1,0 if the model has coef attribute
    F_FeatureSelector_HasFeatureImportance = Column(Integer, nullable=False) # 1,0 if the model has feature_importance attribute
    F_FeatureSelector_PredictionClass = Column(String(20), nullable=False) # regressor or classifier
    
class FeatureSelectorParameter(Base):
    __tablename__ = 'T_FeatureSelectorParameter'
    F_FeatureSelectorParameter_ID = Column(Integer, primary_key=True)
    F_FeatureSelector_ID = Column(Integer, ForeignKey('T_Transformer.F_Transformer_ID'))
    F_FeatureSelectorParameter_Value = Column(Integer, nullable=False)
    F_FeatureSelectorParameter_Name = Column(String(20), nullable=False)
    F_FeatureSelectorParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_FeatureSelector = relationship(FeatureSelector)


class Transformer(Base):
    __tablename__ = 'T_Transformer'
    F_Transformer_ID = Column(Integer, primary_key=True)
    F_Transformer_Name = Column(String(100), nullable=False)
    F_Transformer_CanAugment = Column(Integer, nullable=False) # 1,0 if it can append feautres
    F_Transformer_CanDimDrop = Column(Integer, nullable=False) # 1,0 if it can drop dimensions

class TransformerParameter
    __tablename__ = 'T_TransformerParameter'
    F_TransformerParameter_ID = Column(Integer, primary_key=True)
    F_Transformer_ID = Column(Integer, ForeignKey('T_Transformer.F_Transformer_ID'))
    F_TransformerParameter_Value = Column(Integer, nullable=False)
    F_TransformerParameter_Name = Column(String(20), nullable=False)
    F_TransformerParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_Transformer = relationship(Transformer)
"""
# Add feature selectors ...

transformers = []
scale = Transformer(F_Transformer_Name="scale",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 0)

# Scale parameters
transformers.append(scale)

normalize = Transformer(F_Transformer_Name="normalize",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 0)
# Normalize parameters
transformers.append(normalize)

boxcox = Transformer(F_Transformer_Name="boxcox",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 0)
# Boxcox parameters
transformers.append(boxcox)

binarize = Transformer(F_Transformer_Name="binarize",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 0)

binarize_param = TransformerParameter(F_Transformer = binarize,
                             F_TransformerParameter_Type = 'float',
                               F_TransformerParameter_Name = 'threshold',
                               F_TransformerParameter_Description='Sets all values greater than this value to 1 and less than it to 0.')
transformers.append(binarize)
transformers.append(binarize_param)

brbm = Transformer(F_Transformer_Name="brbm",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 0)
brbm_param1 = TransformerParameter(F_Transformer = brbm,
                                   F_TransformerParameter_Type = 'integer',
                                   F_TransformerParameter_Name = 'n_components',
                                   F_TransformerParameter_Description='Number of binary hidden units.')
brbm_param2 = TransformerParameter(F_Transformer = brbm,
                                   F_TransformerParameter_Type = 'integer',
                                   F_TransformerParameter_Name = 'learning_rate',
                                   F_TransformerParameter_Description='The learning rate for weight updates. It is highly recommended to tune this hyper-parameter. Reasonable values are in the 10**[0., -3.] range.')
transformers.append(brbm)
transformers.append(brbm_param1)
transformers.append(brbm_param2)

# PCA parameters
pca = Transformer(F_Transformer_Name="pca",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 1)
pca_param1 = TransformerParameter(F_Transformer = pca,
                                   F_TransformerParameter_Type = 'float',
                                   F_TransformerParameter_Name = 'percent_variance',
                                   F_TransformerParameter_Description='The first K number of components that cumulatively capture this perentage.')

# PCA parameters
transformers.append(pca)
transformers.append(pca_param1)

tsne = Transformer(F_Transformer_Name="tsne",
                    F_Transformer_CanAugment = 0,
                    F_Transformer_CanDimDrop = 1)
tsne_param1 = TransformerParameter(F_Transformer = tsne,
                                   F_TransformerParameter_Type = 'integer',
                                   F_TransformerParameter_Name = 'n_components',
                                   F_TransformerParameter_Description='Number dimensions to capture.')
# tsne parameters
transformers.append(tsne)
transformers.append(tsne_param1)

kmeans = Transformer(F_Transformer_Name="kmeans",
                    F_Transformer_CanAugment = 1,
                    F_Transformer_CanDimDrop = 0)

kmeans_param1 = TransformerParameter(F_Transformer = kmeans,
                                   F_TransformerParameter_Type = 'integer',
                                   F_TransformerParameter_Name = 'n_clusters',
                                   F_TransformerParameter_Description='Number of clusters to calculate.')
# Kmeans parameters
transformers.append(kmeans)
transformers.append(kmeans_param1)

# Commit all transformers
for t in transformers:
  session.add(t)



models=[]
# Insert Regression Estimators
lr = Estimator(F_Estimator_Name = "Linear Regressor",
                   F_Estimator_Symbol = 'lr',
                   F_Estimator_PredictionClass = 'regressor',
               F_Estimator_CanFeatureSelect=0,
              F_Estimator_Description="Ordinary least squares Linear Regression.")


# Add lr
models.append(lr)

svr = Estimator(F_Estimator_Name = "Support Vector Regressor",
                   F_Estimator_Symbol = 'svr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description="Epsilon-Support Vector Regression. The free parameters in the model are C and epsilon. The implementation is based on libsvm.")

# Add svr
models.append(svr)

rfr = Estimator(F_Estimator_Name = "Random Forest Regressor",
                   F_Estimator_Symbol = 'rfr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                F_Estimator_Description = "A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).")


# Add rfr
models.append(rfr)

abr = Estimator(F_Estimator_Name = "Adaptive Boosting Regressor",
                   F_Estimator_Symbol = 'abr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "An AdaBoost regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases. This class implements the algorithm known as AdaBoost.R2.")

# Add abr
models.append(abr)

knnr = Estimator(F_Estimator_Name = "K-Nearest Neighbors Regressor",
                   F_Estimator_Symbol = 'knnr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")

# add knnr
models.append(knnr)

ridge = Estimator(F_Estimator_Name = "Ridge Regressor",
                   F_Estimator_Symbol = 'ridge',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                 F_Estimator_Description = "")

# add ridge
models.append(ridge)


lasso = Estimator(F_Estimator_Name = "Lasso Regressor",
                   F_Estimator_Symbol = 'lasso',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                 F_Estimator_Description = "Linear Model trained with L1 prior as regularizer (aka the Lasso). The optimization objective for Lasso is: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1")
                 

# add lasso
models.append(lasso)

enet = Estimator(F_Estimator_Name = "ElasticNet Regressor",
                   F_Estimator_Symbol = 'enet',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                F_Estimator_Description = "")

models.append(enet)

mlpr = Estimator(F_Estimator_Name = "Multi-Layer Perceptron Regressor",
                   F_Estimator_Symbol = 'mlpr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")

models.append(mlpr)

br = Estimator(F_Estimator_Name = "Bagging Regressor",
                   F_Estimator_Symbol = 'br',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
              F_Estimator_Description = "")

models.append(br)

dtr = Estimator(F_Estimator_Name = "Decision Tree Regressor",
                   F_Estimator_Symbol = 'dtr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(dtr)

gbr = Estimator(F_Estimator_Name = "Gradient Boosting Regressor",
                   F_Estimator_Symbol = 'gbr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gbr)

gpr = Estimator(F_Estimator_Name = "Gaussian Process Regressor",
                   F_Estimator_Symbol = 'gpr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gpr)


hr = Estimator(F_Estimator_Name = "Huber Regressor",
                   F_Estimator_Symbol = 'hr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
              F_Estimator_Description = "")


models.append(hr)

tsr = Estimator(F_Estimator_Name = "Theil-Sen Regressor",
                   F_Estimator_Symbol = 'tsr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(tsr)

par = Estimator(F_Estimator_Name = "Passive Aggressive Regressor",
                   F_Estimator_Symbol = 'par',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(par)

ard = Estimator(F_Estimator_Name = "ARD Regressor",
                   F_Estimator_Symbol = 'ard',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(ard)

bays_ridge = Estimator(F_Estimator_Name = "Baysian Ridge Regressor",
                   F_Estimator_Symbol = 'bays_ridge',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                      F_Estimator_Description = "")


models.append(bays_ridge)

lasso_lar = Estimator(F_Estimator_Name = "Lasso Least Angle Regressor",
                   F_Estimator_Symbol = 'lasso_lar',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                     F_Estimator_Description = "")


models.append(lasso_lar)

lar = Estimator(F_Estimator_Name = "Least Angle Regressor",
                   F_Estimator_Symbol = 'lar',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(lar)

# Insert Classification Estimators
logr = Estimator(F_Estimator_Name = "Logistic Regression Classifier",
                   F_Estimator_Symbol = 'logr',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")


models.append(logr)

svc = Estimator(F_Estimator_Name = "Support Vector Classifier",
                   F_Estimator_Symbol = 'svc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(svc)

rfc = Estimator(F_Estimator_Name = "Random Forest Classifier",
                   F_Estimator_Symbol = 'rfc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(rfc)

abc = Estimator(F_Estimator_Name = "Adaptive Boosting Classifier",
                   F_Estimator_Symbol = 'abc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(abc)

dtc = Estimator(F_Estimator_Name = "Decision Tree Classifier",
                   F_Estimator_Symbol = 'dtc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(dtc)

gbc = Estimator(F_Estimator_Name = "Gradient Boosting Classifier",
                   F_Estimator_Symbol = 'gbc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gbc)

sgd = Estimator(F_Estimator_Name = "Stochastic Gradient Descent Classifier",
                   F_Estimator_Symbol = 'sgd',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(sgd)

gpc = Estimator(F_Estimator_Name = "Gaussian Process Classifier",
                   F_Estimator_Symbol = 'gpc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gpc)

knnc = Estimator(F_Estimator_Name = "K-Nearest Neighbors Classifier",
                   F_Estimator_Symbol = 'knnc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")


models.append(knnc)

mlpc = Estimator(F_Estimator_Name = "Multi-Layer Perceptron Classifier",
                   F_Estimator_Symbol = 'mlpc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")


models.append(mlpc)

nbc = Estimator(F_Estimator_Name = "Naive Bayes Classifier",
                   F_Estimator_Symbol = 'nbc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "Gaussian Naive Bayes (GaussianNB). Can perform online updates to model parameters via partial_fit method. For details on algorithm used to update feature means and variance online, see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque: http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf")


models.append(nbc)

"""
class FeatureSelectorParameter(Base):
    __tablename__ = 'T_FeatureSelectorParameter'
    F_FeatureSelectorParameter_ID = Column(Integer, primary_key=True)
    F_FeatureSelector_ID = Column(Integer, ForeignKey('T_FeatureSelector.F_FeatureSelector_ID'))
    F_FeatureSelectorParameter_Open= Column(Integer, nullable=True)
    F_FeatureSelectorParameter_Name = Column(String(20), nullable=True)
    F_FeatureSelectorParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_FeatureSelector = relationship(FeatureSelector)
"""



mixed_seleciton_fs = NonEstimatorFeatureSelector(F_NonEstimatorFeatureSelector_Name="Mixed Selection Regressor",
                          F_NonEstimatorFeatureSelector_HasCoef=0,
                          F_NonEstimatorFeatureSelector_HasFeatureImportance=0,
                          F_NonEstimatorFeatureSelector_PredictionClass="regressor"
                          )
session.add(mixed_seleciton_fs)
mixed_selection_fs_param1 = NonEstimatorFeatureSelectorParameter(F_NonEstimatorFeatureSelector=mixed_seleciton_fs,
                                                     F_NonEstimatorFeatureSelectorParameter_Open=0,
                                                     F_NonEstimatorFeatureSelectorParameter_Name="threshold_in",
                                                     F_NonEstimatorFeatureSelectorParameter_Description="Keep variables with corresponding p-values from OLS that are less than or equal to this amount.")
session.add(mixed_selection_fs_param1)
mixed_selection_fs_param2 = NonEstimatorFeatureSelectorParameter(F_NonEstimatorFeatureSelector=mixed_seleciton_fs,
                                                     F_NonEstimatorFeatureSelectorParameter_Open=0,
                                                     F_NonEstimatorFeatureSelectorParameter_Name="threshold_out",
                                                     F_NonEstimatorFeatureSelectorParameter_Description="Kick out variables with corresponding p-values from OLS that are greater than or equal to this amount after we accumulate all of our variables for an iteration.")
session.add(mixed_selection_fs_param2)
plsr_fs = NonEstimatorFeatureSelector(F_NonEstimatorFeatureSelector_Name="Partial Least Squares Regressor",
                          F_NonEstimatorFeatureSelector_HasCoef=1,
                          F_NonEstimatorFeatureSelector_HasFeatureImportance=0,
                          F_NonEstimatorFeatureSelector_PredictionClass="regressor"
                          )

session.add(plsr_fs)
# Begin adding parameters for models


links = ["http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
         "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor",
                 "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression",

        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge",
"http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html",
         "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html",
         
"http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
         "http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html"

]

import requests
import re
import string
printable = set(string.printable)
regex_estimators1 = r'<dd><p>(.*)<\/p>'
regex_estimators2 = r'<p>(.*)<\/p>'
regex_estimators_flag = False
regex_parameters = r'<p.*><strong>(.*)<\/strong> : (.*)<\/p>'

# Make sure the links line up with the models
#for i,j in zip(models, links):
#    print(i.F_Estimator_Name,j)
#input("...")

link_contents = []
for i,link in enumerate(links):
    link_contents.append(str(requests.get(link).content).split("\\n"))
    #print(link_contents[-1])
    #print(link_contents[-1].replace("\\n","")[:100])
    #print("****%s****" %(link[-30:]))
    
    # Find the description of the model, its the first <dd> tag folowed by the next <p>, so it looks funny.
    estimator_description = ""
    for thing in link_contents[-1]:
        #print(thing)
        if regex_estimators_flag == False:
            results_est = re.findall(regex_estimators1,thing)
        else:
            results_est = re.findall(regex_estimators2,thing)
        
        if len(results_est) > 0:
            
            if regex_estimators_flag == False:
                hits = list(map(lambda x: x, results_est))[0]
                
                estimator_description += hits
                #print("a")
                #input(estimator_description)
                regex_estimators_flag = True
                break
            else:
                
                # Process
                hits = list(map(lambda x: x, results_est))[0]
                estimator_description += hits
                regex_estimators_flag=False
                #print("b")
                #input(estimator_description)
                break
            

    # Update description of models by specific regex foolishness
    estimator_description = param_descr = re.sub(r'\\x\d*|e2|<.*>|<|>', '', estimator_description)
    models[i].F_Estimator_Description=estimator_description
    session.add(models[i])
    
    for thing in link_contents[-1]:
        
        results_para = re.findall(regex_parameters, thing)
        if len(results_para) > 0:
            #print(thing)
            #print(list(map(lambda x: (x[0],x[1]), results)))
            #input("...")
            hits = list(map(lambda x: (x[0],x[1]), results_para))[0]
            param_name = hits[0]
            param_descr = re.sub(r'\\x\d*|e2|<.*>|<|>', '', hits[1])
            
            if any([param_name in i for i in ["n_jobs", "random_state", "verbose", "copy_X", "copy_X_train", "cache_size"]]):
                continue
            if 'Attributes:' in thing:
                break
            #if param_name == "X":
            #    break
            param_open = 1
            if any([i in param_descr for i in ['str', 'string', 'bool', 'boolean']]):
                param_open = 0
                
            param = EstimatorParameter(F_Estimator = models[i],
				                       F_EstimatorParameter_Open = param_open,
				                       F_EstimatorParameter_Name = param_name,
				                       F_EstimatorParameter_Description=param_descr)
            session.add(param)
            
    #input("...")



# ... Create Parameters for every estimator

# .. Add Parameters for every estimator

session.commit()

"""
everything = session.query(Estimator).all()

for e in everything:
    print("%s\n\t%s\n\t%s\n\t%s\n\t%s" %(e.F_Estimator_ID,
                                           e.F_Estimator_Name, 
                                           e.F_Estimator_Symbol, 
                                           e.F_Estimator_PredictionClass, 
                                           e.F_Estimator_CanFeatureSelect) )



everything = session.query(Parameter).all()

print("\n")

for e in everything:
    print("%s\n\t%s\n\t%s\n\t%s\n\t%s" %(e.F_Parameter_ID, 
                                           e.F_Estimator_ID, 
                                           e.F_Parameter_Name, 
                                           e.F_Parameter_Open,
                                           e.F_Parameter_Description) )


# Query just the match ups
query = session.query(Estimator, Parameter).filter(Estimator.F_Estimator_ID == Parameter.F_Estimator_ID).all()
for e,p in query:
    print("%s (aka) %s\n\t%s(%s)\t%s" %(e.F_Estimator_Name,e.F_Estimator_Symbol, p.F_Parameter_Name, p.F_Parameter_Open, p.F_Parameter_Description))
"""

