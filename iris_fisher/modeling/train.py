from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from iris_fisher.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


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
    logger.info("Training some model...")

    df = pd.read_csv(features_path)

    X = df.drop(["species", "species_encoded"], axis=1)
    y = df["species_encoded"]

    # TODO: explore cross validation compared to this approach. Right now we ensure deterministic training set with `random_state`
    """
    TODO: explore cross validation compared to this approach. Right now we ensure deterministic training set with `random_state`
    TODO: Try the following models:
    Linear Discriminant Analysis (LDA): This is the method Ronald Fisher originally used in his 1936 paper to develop a linear discriminant model.
    K-Nearest Neighbors (KNN): Often recommended for beginners due to its simplicity and high effectiveness on this dataset.
    Support Vector Machines (SVM): A popular and very accurate model for this dataset, with the potential to achieve high accuracy scores through hyperparameter tuning.
    Decision Trees and Ensemble Methods: Decision trees, random forests, and gradient boosting models are effective and widely used.
    Logistic Regression: Another standard method used for classification tasks on this dataset.
    Neural Networks: Both basic and more complex deep learning models, such as Multi-Layer Perceptrons (MLPs), have been successfully applied. 
    """
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    scores = cross_val_score(pipeline, X, y, cv=5)
    logger.info(f"Cross Validation Scores: {scores}")

    # TODO: Can I get outputted model from cross val instead of having these extra lines with manual train test split?
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipeline.fit(X_train, y_train)  # apply scaling on training data
    score = pipeline.score(X_test, y_test)
    logger.info(
        f"Model Score {score}"
    )  # apply scaling on testing data, without leaking training data.

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
