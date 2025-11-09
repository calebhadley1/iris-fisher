from pathlib import Path

from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import typer
import pandas as pd
import pickle
from iris_fisher.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "iris/iris.data",
    encoder_path: Path = MODELS_DIR / "encoder.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "iris.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(
        input_path, 
        names=column_names
    )

    # Species has the following unique vals. We want to encode them as numerical values
    # array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
    encoder = LabelEncoder()
    df['species_encoded'] = encoder.fit_transform(df['species'])

    df.to_csv(output_path, index=False)

    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
