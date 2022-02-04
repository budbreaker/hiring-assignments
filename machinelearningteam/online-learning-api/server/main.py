import http
import json
import multiprocessing as mp
from flask import Flask, request
from sklearn.metrics import precision_recall_fscore_support
from model import predict, update, metrics


class API:
    def __init__(self, queue):
        self.queue = queue
        self.predictions = []

    def start(self, host='127.0.0.1', port=5000):
        app = Flask(__name__)

        @app.route("/sample", methods=['POST'])
        def _sample():
            # Accepts a list of samples from the dataset. The format is a list of JSON dictionaries using the keys from
            # the CSV file - and strings for all values.

            # Runs a prediction against the most recently trained model - if a model exists yet. Don't output to the user.
            # You only need this to compute model metrics later.

            # The samples should be added to the set of previously received samples

            # Store the sample as received + the predicted  AccountNumber - you don't have to overthink this - opening a
            # file, adding the samples and storing it again is fine

            # Train a new version of the machine learning model using all data received so far as the training set
            content = json.loads(request.get_json())
            for sample in content:
                prediction_features = {key: sample[key] for key in ['CompanyId', 'BankEntryText', 'BankEntryAmount']}
                prediction = predict(prediction_features)[0]
                self.predictions.append((int(sample['AccountNumber']), prediction))
                self.queue.put(content)
            return '', http.HTTPStatus.NO_CONTENT

        @app.route("/predict", methods=['POST'])
        def _predict():
            content = json.loads(request.get_json())
            prediction = predict(content[0])
            return json.dumps([{'AccountNumber': str(prediction[0])}])

        @app.route("/metrics/<int:n>", methods=['GET'])
        def _metrics(n=None):
            true_vals = [x[0] for x in self.predictions]
            predicted = [x[1] for x in self.predictions]

            precision, recall, _, _ = precision_recall_fscore_support(true_vals, predicted, average='weighted', zero_division=0)
            return json.dumps([{'precision': str(precision), 'recall': str(recall)}])

        app.run(host=host, port=port, debug=True)

        # api.add_resource(Sample, "/sample")
        # api.add_resource(Predict, "/predict")
        # api.add_resource(Monitoring, "/monitoring/<int:n>")


# class Sample(Resource):
#     def __init__(self, queue):
#         self.queue = queue
#
#     def post(self):
#         # Accepts a list of samples from the dataset. The format is a list of JSON dictionaries using the keys from
#         # the CSV file - and strings for all values.
#
#         # Runs a prediction against the most recently trained model - if a model exists yet. Don't output to the user.
#         # You only need this to compute model metrics later.
#
#         # The samples should be added to the set of previously received samples
#
#         # Store the sample as received + the predicted  AccountNumber - you don't have to overthink this - opening a
#         # file, adding the samples and storing it again is fine
#
#         # Train a new version of the machine learning model using all data received so far as the training set
#         content = request.get_json()
#         q.put(content)
#
#         return {"cmd": "post"}
#
#
# class Predict(Resource):
#     def post(self):
#         # Accepts a single sample from the dataset. The format is a JSON dictionary using the keys from the CSV file
#         # - except the AccountName,AccountNumber,AccountTypeName fields Predict the expected AccountNumber for a
#         # sample from the dataset Return value is a JSON list containing one string element
#         return {"cmd": "post"}
#
#
# class Monitoring(Resource):
#     def get(self, n):
#         return {"cmd": "monitoring", "data": n}
#
# sample = Sample(q)
# api.add_resource(Sample, "/sample")
# api.add_resource(Predict, "/predict")
# api.add_resource(Monitoring, "/monitoring/<int:n>")


def update_model(q):
    while True:
        data = q.get()
        update(data)


if __name__ == "__main__":
    q = mp.Queue()
    p = mp.Process(target=update_model, args=(q,))
    p.start()
    a = API(q)
    a.start()
