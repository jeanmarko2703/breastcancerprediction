from pydantic import BaseModel
from typing import Optional

class Prediction(BaseModel):
    texture_mean : float
    perimeter_mean : float
    smoothness_mean : float
    concave_points_mean : float
    symmetry_mean : float
    fractal_dimension_mean : float


class Treatment(BaseModel):
    AgeAtDiagnosis : float
    Chemotherapy : float
    Cohort : float
    ERStatus : float
    NeoplasmHistologicGrade : float
    HormoneTherapy : float
    LymphNodesExaminedPositive : float
    MutationCount : float
    NottinghamPrognosticIndex : float
    OncotreeCode : float
    OverallSurvival_Months : float
    PRStatus : float
    RadioTherapy : float
    GeneClassifierSubtype : float
    TumorStage : float




