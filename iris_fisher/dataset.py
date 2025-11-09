from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from iris_fisher.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "iris/iris.data",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    df = pd.read_csv(input_path, names=column_names)

    df.to_csv(output_path, index=False)

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
