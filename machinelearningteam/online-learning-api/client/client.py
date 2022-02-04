import csv
import json
import random
import requests

base_url = "http://127.0.0.1:5000"


def stream_requests():
    with open('../bank_expenses_obfuscated.csv') as file:
        reader = csv.DictReader(file)
        total_requests = 0
        while total_requests < 15000:
            requests_count = random.randint(1, 1000)
            total_requests += requests_count
            print(total_requests)
            request_array = []
            for _ in range(requests_count):
                request_array.append(reader.__next__())
            requests.post(base_url + '/sample', json=json.dumps(request_array))
    response = requests.get(base_url + '/metrics/1000')
    print(response.json())


def request_prediction():
    response = requests.post(base_url + '/predict', json=json.dumps(
        [{'CompanyId': 'foo', 'BankEntryText': 'bar baz', 'BankEntryAmount': '> 10'}]))
    print(response.json())


if __name__ == "__main__":
    # loaded = dill.load(open('naive_bayes_classifier.pkl', 'rb'))

    stream_requests()
    # request_prediction()
