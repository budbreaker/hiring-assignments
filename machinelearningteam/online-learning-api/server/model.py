import dill
import numpy as np
import pandas as pd

from sklearn import preprocessing as prep
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

np.random.seed(2302)
#  column names for dataframe
fieldnames = ["", "CompanyId", "BankEntryDate", "BankEntryText", "BankEntryAmount", "AccountName",
              "AccountNumber", "AccountTypeName", "AccountNumberPredicted"]


def column_selector(column_name):
    return prep.FunctionTransformer(
        lambda X: X[column_name], validate=False)


def predict(data: pd.DataFrame):
    """
    Loads a model from savefile or creates a new one if file doesn't exist. Runs prediction on incoming data and returns
    predicted labels.
    :param data: Incoming Pandas Dataframe
    :type data: pd.DataFrame
    :return: Predicted labels
    :rtype: List
    """
    try:
        #  load model if it exists
        loaded = dill.load(open('naive_bayes_classifier.pkl', 'rb'))
        prediction = loaded.predict(data)
        return prediction

    except FileNotFoundError:
        #  create model if it doesn't
        train = data
        y_train = train.AccountNumber

        vectorizer = CountVectorizer(max_features=10000)
        amount_encoder = CountVectorizer(max_features=50)
        company_id_encoder = CountVectorizer(max_features=500)
        all_features = FeatureUnion(
            [
                ['company', make_pipeline(column_selector('CompanyId'), company_id_encoder)],
                ['text', make_pipeline(column_selector('BankEntryText'), vectorizer)],
                ['amount', make_pipeline(column_selector('BankEntryAmount'), amount_encoder)],
            ])
        classifier = MultinomialNB()
        model = Pipeline([('features', all_features), ('nb', classifier)])
        model.fit(train, y_train)
        dill.dump(model, open('naive_bayes_classifier.pkl', 'wb'))
        prediction = model.predict(data)
        return prediction


def update(data: pd.DataFrame):
    """
    Loads classifier from the savefile and trains using new data. Saves the model after training.
    :param data: Incoming Pandas DataFrame
    :type data: pd.DataFrame
    """
    with open('samples.csv', 'a+') as file:
        data.to_csv(file, header=False, index=False, line_terminator='\n')
        loaded = dill.load(open('naive_bayes_classifier.pkl', 'rb'))
        train = pd.read_csv('samples.csv', names=fieldnames)
        y_train = train.AccountNumber
        loaded.fit(train, y_train)
        dill.dump(loaded, open('naive_bayes_classifier.pkl', 'wb'))


def metrics(n: int):
    """
    Reads last n records in samples file, extracts true and predicted labels and computes precision and recall
    :param n: Number of last predictions to evaluate
    :type n: int
    :return: Precision and recall values of last n predictions
    :rtype: (float, float)
    """
    data = pd.read_csv('samples.csv', names=fieldnames)
    data = data.iloc[-n:]
    precision, recall, _, _ = precision_recall_fscore_support(
        data['AccountNumber'], data['AccountNumberPredicted'], average='weighted', zero_division=0)
    return precision, recall
