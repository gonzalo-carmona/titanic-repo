"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd


silencer = enable_iterative_imputer


def cabin_transformer(x):  # Función que transforma los
    # datos de la columna cabin
    if pd.isnull(x):
        return
    else:
        z = str(x)
        d = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
        return d[z[0]]


def cabin_transform_impute(data):  # Efectúa la transformación de Cabin...
    y_data = pd.DataFrame(data['Survived'])
    X_data = data.drop('Survived', inplace=False, axis=1)

    X_data['Cabin'] = X_data['Cabin'].apply(cabin_transformer)

    data_cabin = pd.DataFrame(
        {'Pclass': X_data['Pclass'],
         'Fare': X_data['Fare'],
         'Cabin': X_data['Cabin']}
         )

    imputer = KNNImputer(n_neighbors=5)  # ... e imputa los datos que
    # faltan con Knn.
    X_data['Cabin'] = imputer.fit_transform(data_cabin)[:, 2]
    return {'X_data': X_data, 'y_data': y_data}


def remove_columns(X_data):  # Elimina columnas que no usaremos en el modelo
    X_data = X_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1,
                         inplace=False)
    return X_data


# Split the dataframe into test and train data
def split_data(data):

    data = cabin_transform_impute(data)

    X_data = remove_columns(data['X_data'])
    y_data = data['y_data']

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                        random_state=42)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data, SVC_args):

    # Pipelines de transformación previa
    age_pipeline = make_pipeline(
        IterativeImputer(estimator=RandomForestClassifier(), max_iter=10),
        MinMaxScaler(feature_range=(0, 1))
        )
    fare_pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        MinMaxScaler(feature_range=(0, 1))
        )
    cabin_pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1)))
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(dtype=int, sparse=False,
                      handle_unknown='ignore')
        )
    preprocessor = make_column_transformer(
        (categorical_pipeline, ['Pclass', 'Sex', 'SibSp',
                                'Parch', 'Embarked']),
        (age_pipeline, ['Age']), (fare_pipeline, ['Fare']),
        (cabin_pipeline, ['Cabin']), remainder='drop'
        )
    svc = SVC(C=SVC_args["C"], probability=True)
    final_pipeline = make_pipeline(preprocessor, svc)
    pipe = final_pipeline.fit(
        data["train"]["X"], data["train"]["y"].to_numpy().ravel()
    )

    return pipe


# Evaluate the metrics for the model
def get_model_metrics(model, data):

    preds = model.predict(data["test"]["X"])

    # Precision
    pre = precision_score(data["test"]["y"], preds)
    metrics = {"precision": pre}
    return metrics


def main():
    print("Running train.py")

    # Define training parameters
    SVC_args = {"C": 112.884}

    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'train.csv')
    data = pd.read_csv(data_file)

    data = split_data(data)

    model = train_model(data, SVC_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")

    model_name = "svc_classifier_pipeline.pkl"

    joblib.dump(value=model, filename=model_name)


if __name__ == '__main__':
    main()
