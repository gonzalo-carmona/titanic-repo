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
import numpy
import pandas
import joblib
import os
from azureml.core.model import Model
from inference_schema.schema_decorators \
    import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
from titanic.training.train import cabin_transformer, remove_columns


def init():
    # load the model from file into a global object
    global model

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])

    model = joblib.load(model_path)


input_sample = numpy.array([
    [1, 3, 'Braund, Mr. Owen Harris', 'male',
     22.0, 1, 0, 'A/5 21171', 7.2500, 'C85', 'S'],
    [3, 3, 'Heikkinen, Miss. Laina', 'female', 26.0, 0, 0,
     'STON/O2. 3101282', 7.9250, 'C123', 'S']])

output_sample = numpy.array([
    0,
    1])


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, request_headers):
    data = pandas.DataFrame(data, columns=[
        'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
    ])
    data['Cabin'] = data['Cabin'].map(cabin_transformer)
    data = remove_columns(data)
    result = model.predict(data)

    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}


if __name__ == "__main__":
    # Test scoring
    init()
    test_row_1 = "{'data':[[1, 3, 'Braund, Mr. Owen Harris', 'male', 22.0,"
    test_row_2 = " 1, 0, 'A/5 21171', 7.2500, 'C85', 'S'], [3, 3, 'Heikkin"
    test_row_3 = "en, Miss. Laina', 'female', 26.0, 0, 0, 'STON/O2. 3101282'"
    test_row_4 = ", 7.9250, 'C123', 'S']]}"
    test_row = test_row_1 + test_row_2 + test_row_3 + test_row_4
    prediction = run(test_row, {})
    print("Test result: ", prediction)
