# Initialize database with all of our tables and fields

import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


# Estimators the user can select from
class Estimator(Base):
    #F_Estimator_ID | PK
    #F_Estimator_Name | char(200)
    
    __tablename__ = 'T_Estimator'
    F_Estimator_ID = Column(Integer, primary_key=True)
    F_Estimator_Name = Column(String(250), nullable=False)
    F_Estimator_Symbol = Column(String(20), nullable=False)
    F_Estimator_PredictionClass = Column(String(20), nullable=False) # regressor or classifier
    F_Estimator_CanFeatureSelect = Column(Integer, nullable=False) # 1,0 if it can feature select
    F_Estimator_Description = Column(String(1000), nullable=True)
    
# All possible parameters
class EstimatorParameter(Base):
    
    __tablename__ = 'T_EstimatorParameter'
    F_EstimatorParameter_ID = Column(Integer, primary_key=True)
    F_Estimator_ID = Column(Integer, ForeignKey('T_Estimator.F_Estimator_ID'))
    F_EstimatorParameter_Open = Column(Integer, nullable=False)
    F_EstimatorParameter_Name = Column(String(20), nullable=False)
    F_EstimatorParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_Estimator = relationship(Estimator)


# When user selects estimator then chooses values for each parameter to go with it
class EstimatorParameterValue(Base):
    #F_ParameterValue_ID | PK
    #F_ParameterValue_Realization | Char(20); Actual value the user selected for the parameter
    
    __tablename__ = 'T_EstimatorParameterValue'
    F_EstimatorParameterValue_ID = Column(Integer, primary_key=True)
    F_EstimatorParameter_ID = Column(Integer , ForeignKey('T_EstimatorParameter.F_EstimatorParameter_ID'))
    F_EstimatorParameterValue_Realization = Column(String(10), nullable=False)

    
     #Relationship From
    F_EstimatorParameter = relationship(EstimatorParameter)

class Transformer(Base):
    __tablename__ = 'T_Transformer'
    F_Transformer_ID = Column(Integer, primary_key=True)
    F_Transformer_Name = Column(String(100), nullable=False)
    F_Transformer_CanAugment = Column(Integer, nullable=False) # 1,0 if it can append feautres
    F_Transformer_CanDimDrop = Column(Integer, nullable=False) # 1,0 if it can drop dimensions
    F_Transformer_Description = Column(String(200), nullable=True)

class TransformerParameter(Base):
    __tablename__ = 'T_TransformerParameter'
    F_TransformerParameter_ID = Column(Integer, primary_key=True)
    F_Transformer_ID = Column(Integer, ForeignKey('T_Transformer.F_Transformer_ID'))
    F_TransformerParameter_Type = Column(String(10), nullable=False)
    F_TransformerParameter_Name = Column(String(20), nullable=False)
    F_TransformerParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_Transformer = relationship(Transformer)

# When user selects transformer then chooses values for each parameter to go with it
class TransformerParameterValue(Base):
    #F_ParameterValue_ID | PK
    #F_ParameterValue_Realization | Char(20); Actual value the user selected for the parameter
    
    __tablename__ = 'T_TransformerParameterValue'
    F_TransformerParameterValue_ID = Column(Integer, primary_key=True)
    F_TransformerParameter_ID = Column(Integer , ForeignKey('T_TransformerParameter.F_TransformerParameter_ID'))
    F_TransformerParameterValue_Realization = Column(String(10), nullable=False)
    
     #Relationship From
    F_EstimatorParameter = relationship(TransformerParameter)

class NonEstimatorFeatureSelector(Base):
    __tablename__ = 'T_NonEstimatorFeatureSelector'
    F_NonEstimatorFeatureSelector_ID = Column(Integer, primary_key=True)
    F_NonEstimatorFeatureSelector_Name = Column(String(100), nullable=False)
    F_NonEstimatorFeatureSelector_HasCoef = Column(Integer, nullable=False) # 1,0 if the model has coef attribute
    F_NonEstimatorFeatureSelector_HasFeatureImportance = Column(Integer, nullable=False) # 1,0 if the model has feature_importance attribute
    F_NonEstimatorFeatureSelector_PredictionClass = Column(String(20), nullable=False) # regressor or classifier

class NonEstimatorFeatureSelectorParameter(Base):
    __tablename__ = 'T_NonEstimatorFeatureSelectorParameter'
    F_NonEstimatorFeatureSelectorParameter_ID = Column(Integer, primary_key=True)
    F_NonEstimatorFeatureSelector_ID = Column(Integer, ForeignKey('T_NonEstimatorFeatureSelector.F_NonEstimatorFeatureSelector_ID'))
    F_NonEstimatorFeatureSelectorParameter_Open= Column(Integer, nullable=True)
    F_NonEstimatorFeatureSelectorParameter_Name = Column(String(20), nullable=True)
    F_NonEstimatorFeatureSelectorParameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_NonEstimatorFeatureSelector = relationship(NonEstimatorFeatureSelector)
    
# When user selects transformer then chooses values for each parameter to go with it
class NonEstimatorFeatureSelectorParameterValue(Base):
    #F_ParameterValue_ID | PK
    #F_ParameterValue_Realization | Char(20); Actual value the user selected for the parameter
    
    __tablename__ = 'T_NonEstimatorFeatureSelectorParameterValue'
    F_NonEstimatorFeatureSelectorParameterValue_ID = Column(Integer, primary_key=True)
    F_NonEstimatorFeatureSelectorParameter_ID = Column(Integer , ForeignKey('T_NonEstimatorFeatureSelectorParameter.F_NonEstimatorFeatureSelectorParameter_ID'))
    F_NonEstimatorFeatureSelectorParameterValue_Realization = Column(String(10), nullable=False)
    
     #Relationship From
    F_NonEstimatorFeatureSelectorParameter = relationship(NonEstimatorFeatureSelectorParameter)

"""

class T_ErrorMetric(Base):
    #F_ErrorMetric_ID PK | PK
    #F_ErrorMetric_Name | Char(100)
    #F_ErrorMetric_Purpose | Char(20)
    pass

class T_History(Base):
    #F_History_ID | PK
    #F_History_Time | Date
    #F_ErrorMetric_ID | FK
    #F_History_ErrorValue | Char(100)
    #F_History_ErrorValue | Integer
    #F_Batch_ID | FK
    #F_Estimator_ID | FK
    #F_Parameter_ID | FK
    #F_ParameterValue_ID |FK 
    #F_Parameter_Value_Actual | Char(100)
    #F_Preprocessor_ID | FK
    #F_PreprocesorType_ID | FK
    pass

"""

# Create the db file
engine = create_engine('sqlite:///streamml.db')


# Link Tables to DB via Base
Base.metadata.create_all(engine)