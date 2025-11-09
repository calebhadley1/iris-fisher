from pathlib import Path
import pickle
from sys import prefix
from fastapi import APIRouter
from iris_fisher.api.schemas.iris import IrisIn
from iris_fisher.config import MODELS_DIR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from loguru import logger

router = APIRouter(prefix="/v1/iris")


@router.post("/predict")
def predict(iris_in: IrisIn):
    logger.info(f"Received Request with params: {iris_in}")

    logger.info("Create DataFrame from Input")
    df = pd.DataFrame([iris_in.model_dump()])
    logger.info(f"Input DF: {df}")

    logger.info("Loading Model")
    model_path: Path = MODELS_DIR / "model.pkl"
    with open(model_path, 'rb') as f:
        model: Pipeline = pickle.load(f)
    
    logger.info("Getting Prediction")
    pred_encoded = model.predict(df)
    logger.info(f"Encoded Prediction: {pred_encoded}")

    logger.info("Loading LabelEncoder")
    encoder_path: Path = MODELS_DIR / "encoder.pkl"
    with open(encoder_path, 'rb') as f:
        encoder: LabelEncoder = pickle.load(f)
    
    logger.info("Transforming Encoded Prediction into readable Species")
    pred = encoder.inverse_transform(pred_encoded)
    logger.info(f"Predicted Species: {pred}")
    return {"pred": pred[0]}