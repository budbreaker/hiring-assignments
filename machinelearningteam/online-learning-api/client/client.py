import csv
import json
import random
import requests
import time

base_url = "http://127.0.0.1:5000"


def stream_requests():
    """
    Function reads samples from provided csv file and sends them to the API server in random chunks to simulate load.
    After sending the samples it waits a bit and sends requests to /predict and /metrics endpoints with the sample inputs.
    """
    with open('../bank_expenses_obfuscated.csv') as file:
        reader = csv.DictReader(file)  # Read input rows as a dictionary for easy passing to JSON
        total_requests = 0
        while total_requests < 15000:
            request_count = random.randint(1, 1000)
            total_requests += request_count
            print(f'Sending {request_count} samples. Subtotal:{total_requests}')  # Reports number of samples to send
            request_array = []
            for _ in range(request_count):  # gather this many samples
                request_array.append(reader.__next__())
            requests.post(base_url + '/sample', json=json.dumps(request_array))  # encode the array of dicts and send away

    for _ in range(10):
        response = requests.get(base_url + '/metrics/1000')  # request model metrics
        print(response.json())
        time.sleep(1)

    response = requests.post(base_url + '/predict', json=json.dumps(
        [{'CompanyId': 'foo', 'BankEntryText': 'bar baz', 'BankEntryAmount': '> 10'}]))  # predict sample
    print(response.json())


if __name__ == "__main__":
    stream_requests()
