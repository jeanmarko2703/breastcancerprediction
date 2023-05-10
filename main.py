from fastapi import FastAPI
from routes.prediction import predictionRouter

app = FastAPI()

app.include_router(predictionRouter)
