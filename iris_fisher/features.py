from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import typer

from iris_fisher.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    encoder_path: Path = MODELS_DIR / "encoder.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    df = pd.read_csv(input_path)

    # Species has the following unique vals. We want to encode them as numerical values
    # array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
    encoder = LabelEncoder()
    df["species_encoded"] = encoder.fit_transform(df["species"])

    df.to_csv(output_path, index=False)

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
