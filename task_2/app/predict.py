"""Prediction logic for the Iris app.

This module loads the classifier that train_model.py saved (Delivery 1)
and turns four flower measurements into a species name.
It does not train anything: the fitted state is already inside model.joblib.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris

# Find model.joblib relative to THIS file, not to the folder the user runs
# the command from. app/predict.py -> parent is app/ -> parent is the project root.
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

# Load the trained model once, when the module is first imported.
model = joblib.load(MODEL_PATH)

# The model returns numeric labels 0, 1, 2.
# The Iris dataset gives the name for each label in the same order:
# 0 -> setosa, 1 -> versicolor, 2 -> virginica.
SPECIES_NAMES = load_iris().target_names

# Training order of the features (iris_data.feature_names in train_model.py).
FEATURE_ORDER = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def predict(features):
    """Predict the Iris species for one flower.

    features: four numbers in training order:
        [sepal length, sepal width, petal length, petal width], in cm.
    Returns the species name, for example "setosa".
    """
    if len(features) != len(FEATURE_ORDER):
        raise ValueError(
            f"Expected {len(FEATURE_ORDER)} features in this order: "
            f"{FEATURE_ORDER}. Got {len(features)}."
        )

    # Outer list = a collection of samples, inner list = one flower.
    # scikit-learn's predict() always expects a 2D input.
    sample = [[float(value) for value in features]]

    label = model.predict(sample)[0]  # numeric label, e.g. 0
    return str(SPECIES_NAMES[label]).capitalize()


if __name__ == "__main__":
    # Quick check without Streamlit: python app/predict.py
    print(predict([5.1, 3.5, 1.4, 0.2]))  # expected: Setosa
    print(predict([7.0, 3.2, 4.7, 1.4]))  # expected: Versicolor
    print(predict([6.3, 3.3, 6.0, 2.5]))  # expected: Virginica
