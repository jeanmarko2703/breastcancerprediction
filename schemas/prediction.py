from pydantic import BaseModel
from typing import Optional

class Prediction(BaseModel):
    texture_mean : float
    perimeter_mean : float
    smoothness_mean : float
    concave_points_mean : float
    symmetry_mean : float
    fractal_dimension_mean : float


