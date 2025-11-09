from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from iris_fisher.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def train_log_reg(X, y, model_path):
    logger.info("Training Logistic Regression...")

    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    results = cross_validate(pipeline, X, y, cv=5, scoring=['accuracy', 'f1_macro'], return_estimator=False)
    logger.info(f"Cross Validation Results {results}")
    logger.info("%0.2f accuracy with a standard deviation of %0.2f" % (results['test_accuracy'].mean(), results['test_accuracy'].std()))
    logger.info("%0.2f macro f1 with a standard deviation of %0.2f" % (results['test_f1_macro'].mean(), results['test_f1_macro'].std()))

    # Retrain model on entire dataset
    pipeline.fit(X, y) 

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    logger.info(f"Logistic Regression Training Complete. Pickle can be found at: {model_path}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # Note: Labels are already includeed in features.csv
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training models...")

    df = pd.read_csv(features_path)

    X = df.drop(["species", "species_encoded"], axis=1)
    y = df["species_encoded"]

    """
    TODO: Try the following models:
    Linear Discriminant Analysis (LDA): This is the method Ronald Fisher originally used in his 1936 paper to develop a linear discriminant model.
    K-Nearest Neighbors (KNN): Often recommended for beginners due to its simplicity and high effectiveness on this dataset.
    Support Vector Machines (SVM): A popular and very accurate model for this dataset, with the potential to achieve high accuracy scores through hyperparameter tuning.
    Decision Trees and Ensemble Methods: Decision trees, random forests, and gradient boosting models are effective and widely used.
    Logistic Regression: Another standard method used for classification tasks on this dataset.
    Neural Networks: Both basic and more complex deep learning models, such as Multi-Layer Perceptrons (MLPs), have been successfully applied. 
    """
    train_log_reg(X, y, model_path)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
