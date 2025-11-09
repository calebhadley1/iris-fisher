from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import typer

from iris_fisher.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def train_linear_model(X, y, model, model_path):
    pipeline = make_pipeline(StandardScaler(), model)
    results = cross_validate(
        pipeline, X, y, cv=5, scoring=["accuracy", "f1_macro"], return_estimator=False
    )
    logger.info(f"Cross Validation Results {results}")
    logger.info(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (results["test_accuracy"].mean(), results["test_accuracy"].std())
    )
    logger.info(
        "%0.2f macro f1 with a standard deviation of %0.2f"
        % (results["test_f1_macro"].mean(), results["test_f1_macro"].std())
    )

    # Retrain model on entire dataset
    pipeline.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)


def train_lin_disc(X, y, model_path):
    logger.info("Training Linear Discriminant, just like Fisher did in 1936...")
    train_linear_model(X, y, LinearDiscriminantAnalysis(), model_path)
    logger.info(f"Linear Discriminant Training Complete. Pickle can be found at: {model_path}\n")


def train_knn(X, y, model_path):
    logger.info("Training KNN")
    train_linear_model(X, y, KNeighborsClassifier(), model_path)
    logger.info(f"KNN Training Complete. Pickle can be found at: {model_path}\n")


def train_svm(X, y, model_path):
    logger.info("Training SVM")
    train_linear_model(X, y, SVC(), model_path)
    logger.info(f"SVM Training Complete. Pickle can be found at: {model_path}\n")


def train_dec_tree(X, y, model_path):
    logger.info("Training DecisionTreeClassifier...")
    train_linear_model(X, y, DecisionTreeClassifier(), model_path)
    logger.info(
        f"DecisionTreeClassifier Training Complete. Pickle can be found at: {model_path}\n"
    )


def train_rfc(X, y, model_path):
    logger.info("Training RandomForestClassifier...")
    train_linear_model(X, y, RandomForestClassifier(), model_path)
    logger.info(
        f"RandomForestClassifier Training Complete. Pickle can be found at: {model_path}\n"
    )


def train_gbc(X, y, model_path):
    logger.info("Training GradientBoostingClassifier...")
    train_linear_model(X, y, GradientBoostingClassifier(), model_path)
    logger.info(
        f"GradientBoostingClassifier Training Complete. Pickle can be found at: {model_path}\n"
    )


def train_mlp(X, y, model_path):
    logger.info("Training MLPClassifier...")
    train_linear_model(X, y, MLPClassifier(), model_path)
    logger.info(f"MLPClassifier Training Complete. Pickle can be found at: {model_path}\n")


def train_log_reg(X, y, model_path):
    logger.info("Training Logistic Regression...")
    train_linear_model(X, y, LogisticRegression(), model_path)
    logger.info(f"Logistic Regression Training Complete. Pickle can be found at: {model_path}\n")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # Note: Labels are already includeed in features.csv
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    lin_disc_model_path: Path = MODELS_DIR / "lin_disc.pkl",
    knn_model_path: Path = MODELS_DIR / "knn.pkl",
    svm_model_path: Path = MODELS_DIR / "svm.pkl",
    dec_tree_model_path: Path = MODELS_DIR / "dec_tree.pkl",
    rfc_model_path: Path = MODELS_DIR / "rfc.pkl",
    gbc_model_path: Path = MODELS_DIR / "gbc.pkl",
    mlp_model_path: Path = MODELS_DIR / "mlp.pkl",
    log_reg_model_path: Path = MODELS_DIR / "log_reg.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training models...")

    df = pd.read_csv(features_path)

    X = df.drop(["species", "species_encoded"], axis=1)
    y = df["species_encoded"]

    train_lin_disc(X, y, lin_disc_model_path)
    train_knn(X, y, knn_model_path)
    train_svm(X, y, svm_model_path)
    train_dec_tree(X, y, dec_tree_model_path)
    train_rfc(X, y, rfc_model_path)
    train_gbc(X, y, gbc_model_path)
    train_mlp(X, y, mlp_model_path)
    train_log_reg(X, y, log_reg_model_path)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
