import pickle

import mlflow
from mlflow.tracking import MlflowClient

from flask import Flask, request, jsonify

RUN_ID = "baed3f4f08d64a59b58c51c884a22067"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("green-taxi-duration")

client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)

path = client.download_artifacts(run_id = RUN_ID, path='dict_vectorizer.bin')
print(f'downloading the dict vectorizer to {path}')

with open(path, 'rb') as f_in:
    dv = pickle.load(f_in)

logged_model = f'runs:/{RUN_ID}/model'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)
print(type(model))

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(ride)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)