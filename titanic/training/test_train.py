import numpy as np
import pandas as pd
from titanic.training.train import get_model_metrics
import joblib

def init():
    # load the model from file into a global object
    global model
    global cols
    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])

    model = joblib.load(model_path)

    cols = ['Pclass', 'Sex', 'Age', 'SibSp',
               'Parch', 'Fare', 'Cabin', 'Embarked']

def test_directional_model(): # Females survive more than males

    X_1 = pd.DataFrame(
        np.array([[1, 'male', 32.0, 1, 1, 53.5, 3.0, 'S']]), columns = cols
    )

    X_2 = pd.DataFrame(
        np.array([[1, 'female', 32.0, 1, 1, 53.5, 3.0, 'S']]), columns
    )

    y_1 = model.predict_proba(X_1)
    y_2 = model.predict_proba(X_2)

    assert y_1[0][0] > y_2[0][0]

def test_get_model_metrics():

    class MockModel:

        @staticmethod
        def predict(data):
            return (np.array([0,1]))

    X_test = pd.DataFrame(
        np.array([[1, 'male', 32.0, 1, 1, 53.5, 3.0, 'S'],
                  [1, 'female', 32.0, 1, 1, 53.5, 3.0, 'S']]), columns = cols
    )

    y_test = np.array([0, 1])
    data = {"test": {"X": X_test, "y": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'precision' in metrics
    pre = metrics['precision']
    np.testing.assert_equal(pre, 1)