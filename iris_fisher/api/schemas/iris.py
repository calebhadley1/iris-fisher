from enum import Enum

from pydantic import BaseModel


class IrisIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class SpeciesEnum(str, Enum):
    iris_setosa = "Iris-setosa"
    iris_versicolor = "Iris-versicolor"
    iris_virginica = "Iris-virginica"


class IrisOut(BaseModel):
    species: SpeciesEnum
