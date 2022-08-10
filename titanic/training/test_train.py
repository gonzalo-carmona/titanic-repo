import numpy as np
import pandas as pd
from titanic.training.train import get_model_metrics, train_model, split_data
import os


def test_directional_model():  # Females survive more than males

    # Define training parameters
    SVC_args = {"C": 112.884}

    global cols

    cols = ['Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Fare', 'Cabin', 'Embarked']
    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'train.csv')
    data = pd.read_csv(data_file)

    data = split_data(data)

    model = train_model(data, SVC_args)

    X_1 = pd.DataFrame(
        np.array([[1, 'male', 32.0, 1, 1, 53.5, 3.0, 'S']]), columns=cols
    )

    X_2 = pd.DataFrame(
        np.array([[1, 'female', 32.0, 1, 1, 53.5, 3.0, 'S']]), columns=cols
    )

    y_1 = model.predict_proba(X_1)
    y_2 = model.predict_proba(X_2)

    assert y_1[0][0] > y_2[0][0]


def test_get_model_metrics():

    class MockModel:

        @staticmethod
        def predict(data):
            return (np.array([0, 1]))

    X_test = pd.DataFrame(
        np.array([[1, 'male', 32.0, 1, 1, 53.5, 3.0, 'S'],
                  [1, 'female', 32.0, 1, 1, 53.5, 3.0, 'S']]), columns=cols
    )

    y_test = np.array([0, 1])
    data = {"test": {"X": X_test, "y": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'precision' in metrics
    pre = metrics['precision']
    np.testing.assert_equal(pre, 1)
