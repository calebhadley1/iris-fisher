from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

    X = df.drop(['species', 'species_encoded'], axis=1)
    y = df['species_encoded']

    # TODO: explore cross validation compared to this approach. Right now we ensure deterministic training set with `random_state`
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    
    pipe.fit(X_train, y_train)  # apply scaling on training data
    score = pipe.score(X_test, y_test)
    logger.info(f"Model Score {score}")  # apply scaling on testing data, without leaking training data.

    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
