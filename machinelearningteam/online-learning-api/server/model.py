import csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing as prep
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
import dill

import numpy as np

np.random.seed(2302)


def column_selector(column_name):
    return prep.FunctionTransformer(
        lambda X: X[column_name], validate=False)


def predict(data):
    try:
        loaded = dill.load(open('naive_bayes_classifier.pkl', 'rb'))
        prediction = loaded.predict(pd.DataFrame(data,index=[0]))
        return prediction

    except FileNotFoundError:
        return None


def update(data):
    fieldnames = ["", "CompanyId", "BankEntryDate", "BankEntryText", "BankEntryAmount", "AccountName",
                  "AccountNumber", "AccountTypeName"]
    with open('samples.csv', 'a+') as file:
        writer = csv.DictWriter(file, fieldnames, lineterminator='\n')
        for row in data:
            writer.writerow(row)

    try:
        loaded = dill.load(open('naive_bayes_classifier.pkl', 'rb'))
        train = pd.read_csv('samples.csv', names=fieldnames)
        Y_train = train.AccountNumber
        loaded.fit(train, Y_train)
        return True

    except FileNotFoundError:
        data = pd.read_csv('samples.csv', names=fieldnames)
        data['split'] = np.random.random(data.shape[0])
        test = data[data.split > 0.5]
        train = data[data.split <= 0.5]
        Y_test = test.AccountNumber
        Y_train = train.AccountNumber

        vectorizer = CountVectorizer(max_features=10000)
        amount_encoder = CountVectorizer(max_features=50)
        companyId_encoder = CountVectorizer(max_features=500)
        all_features = FeatureUnion(
            [
                ['company', make_pipeline(column_selector('CompanyId'), companyId_encoder)],
                ['text', make_pipeline(column_selector('BankEntryText'), vectorizer)],
                ['amount', make_pipeline(column_selector('BankEntryAmount'), amount_encoder)],
            ])
        classifier = MultinomialNB()
        model = Pipeline([('features', all_features), ('nb', classifier)])
        model.fit(train, Y_train)
        dill.dump(model, open('naive_bayes_classifier.pkl', 'wb'))

        return False


def metrics():
    try:
        data = pd.read_csv('samples.csv')

    except FileNotFoundError:
        return
