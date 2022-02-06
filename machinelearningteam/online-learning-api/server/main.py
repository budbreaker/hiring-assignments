import http
import json
import multiprocessing as mp
import re
import pandas as pd

from flask import Flask, request
from model import predict, update, metrics


class API:
    def __init__(self, queue, lock):
        self.queue = queue
        self.lock = lock
        self.app = None

    def start(self, host='127.0.0.1', port=5000):
        self.app = Flask(__name__)

        @self.app.route("/sample", methods=['POST'])
        def _sample():
            """
            Method processes incoming sample data.
            Validates the incoming data and creates Pandas DataFrame from it.
            It passes the received data to a multiprocessing queue to free up the API endpoint ASAP.
            responses:
                204:
                    Success. Does not return any data.
                400:
                    Bad request.
            """
            json_data = request.get_json()
            input_validation = re.match(r'\[(\{(\"[a-zA-Z]*\":\s+\"[0-9a-zA-Z\s:<>-]+\",?\s?)+\},?\s*)+\]', json_data)
            if input_validation is None:
                return '', http.HTTPStatus.BAD_REQUEST
            content = json.loads(json_data)
            df = pd.DataFrame(content)
            self.queue.put(df)
            return '', http.HTTPStatus.NO_CONTENT

        @self.app.route("/predict", methods=['POST'])
        def _predict():
            """
            Passes inputted label to model and returns predicted label.
            responses:
                200:
                    Success. Returns predicted label.
                    Sample response: [{'AccountNumber':'2200'}]
                400:
                    Bad request.
            """
            json_data = request.get_json()
            input_validation = re.match(r'\[(\{(\"[a-zA-Z]*\":\s+\"[0-9a-zA-Z\s:<>-]+\",?\s?)+\},?\s*)+\]', json_data)
            if input_validation is None:
                return '', http.HTTPStatus.BAD_REQUEST
            content = json.loads(json_data)
            df = pd.DataFrame(content)
            lock.acquire()
            prediction = predict(df)
            lock.release()
            return json.dumps([{'AccountNumber': str(prediction[0])}])

        @self.app.route("/metrics/<int:n>", methods=['GET'])
        def _metrics(n=None):
            """
            Route for displaying metrics.
            responses:
                200:
                    Success. Returns precision and recall on n last predictions.
                    Sample response: [{'precision': '0.047', 'recall': '0.001'}]
                500:
                    internal error, probably bad input.
            """
            lock.acquire()
            precision, recall = metrics(n)
            lock.release()
            return json.dumps([{'precision': str(precision), 'recall': str(recall)}])

        self.app.run(host=host, port=port, debug=True)


def predict_and_update_model(queue, lock):
    """
    Waits for data to be present in queue. Then predicts labels and trains the model when the data arrive.
    :param queue: A multiprocessing queue
    :type queue: mp.Queue()
    """
    while True:
        data = queue.get()
        prediction = predict(data)
        data['AccountNumberPredicted'] = prediction
        lock.acquire()
        update(data)
        lock.release()
        print(f'Processed {data.shape[0]} samples')


if __name__ == "__main__":
    #  Initialize process that will process incoming data to not hold up API
    queue = mp.Queue()
    lock = mp.Lock()
    p = mp.Process(target=predict_and_update_model, args=(queue, lock,))
    p.start()

    #  Initialize and start the server
    a = API(queue, lock)
    a.start()
