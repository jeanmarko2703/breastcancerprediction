from http.client import HTTPException
from telnetlib import STATUS
from fastapi import APIRouter
from schemas.prediction import Prediction
from schemas.treatment import Treatment
#import ML
import pandas as pd
import numpy as np  
import seaborn as sns
import sklearn.metrics 
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pygeohash as gh
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import joblib

predictionRouter = APIRouter()

@predictionRouter.post("/diagnosis")
def create_prediction(prediction: Prediction):
    mnb = joblib.load('routes/breast_cancer_diagnosis_random_forest_model.joblib')


    data = {
        'texture_mean' : [prediction.texture_mean],
        'perimeter_mean' : [prediction.perimeter_mean],
        'smoothness_mean': [prediction.smoothness_mean],
        'concave points_mean': [prediction.concave_points_mean],
        'symmetry_mean' : [prediction.symmetry_mean],
        'fractal_dimension_mean' : [prediction.fractal_dimension_mean],
    }
    
    

    data = pd.DataFrame(data)

    #------ datos a predecir
    y_pred=mnb.predict(data)

    if y_pred[0] == 0:
        result = 'BENIGNO'
    elif y_pred[0] == 1:
        result = 'MALIGNO'

    else:
        result = 'ERROR'



    print(result)
    return result


@predictionRouter.post("/treatment")
def create_prediction(PredictionTreatment: Treatment):
    mnb = joblib.load('routes/breast_cancer_treatment_random_forest_model.joblib')


    # data = {
    #     'texture_mean' : [PredictionTreatment.AgeAtDiagnosis],
    #     'perimeter_mean' : [prediction.perimeter_mean],
    #     'smoothness_mean': [prediction.smoothness_mean],
    #     'concave points_mean': [prediction.concave_points_mean],
    #     'symmetry_mean' : [prediction.symmetry_mean],
    #     'fractal_dimension_mean' : [prediction.fractal_dimension_mean],
    # }
    
    data = {
        'AgeAtDiagnosis' : [PredictionTreatment.AgeAtDiagnosis],
        'Chemotherapy' : [PredictionTreatment.Chemotherapy],
        'Cohort' : [PredictionTreatment.Cohort],
        'ERStatus' : [PredictionTreatment.ERStatus],
        'NeoplasmHistologicGrade' : [PredictionTreatment.NeoplasmHistologicGrade],
        'HormoneTherapy' : [PredictionTreatment.HormoneTherapy],
        'LymphNodesExaminedPositive' : [PredictionTreatment.LymphNodesExaminedPositive],
        'MutationCount' : [PredictionTreatment.MutationCount],
        'NottinghamPrognosticIndex' : [PredictionTreatment.NottinghamPrognosticIndex],
        'OncotreeCode' : [PredictionTreatment.OncotreeCode],
        'OverallSurvival(Months)' : [PredictionTreatment.OverallSurvival_Months],
        'PRStatus' : [PredictionTreatment.PRStatus],
        'RadioTherapy' : [PredictionTreatment.RadioTherapy],
        'GeneClassifierSubtype' : [PredictionTreatment.GeneClassifierSubtype],
        'TumorStage' : [PredictionTreatment.TumorStage],
    }

    data = pd.DataFrame(data)

    #------ datos a predecir
    y_pred=mnb.predict(data)

    if y_pred[0] == 0:
        result = 'MUERE'
    elif y_pred[0] == 1:
        result = 'SOBREVIVE'

    else:
        result = 'ERROR'



    print(result)
    return result
