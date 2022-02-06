# Solution

The flask server is awaiting API calls. The client, when run, will first send around 15000 samples from `bank_expenses_obfuscated.csv` in chunks of random size using the `/sample` endpoint. After it is done it sends the `/metrics/1000` request every second for 10 seconds. Since the server is processing the samples in the meantime, we can observe changing performance of the model. Finally, it sends single request for prediction taken from model/model.py using the `/predict` endpoint. 

The server takes the input data from `/sample` calls and sends them to a multiprocessing queue where a subprocess is waiting to process the data (predict labels and train the model). This is to make API available for the next request (at least for the same endpoint, when I was sending the samples in rapid succession it would accept the next sample only after it was done processing the current one). While it's processing the samples the `/metrics` calls start arriving, and the server reports the current state of the model for the requested number of past predictions. We can see the metrics (precision, recall) change in time. When the `/predict` call arrives, the server uses the current model to predict the value.

`/server/model.py` contains functions for using and updating the model and calculating its metrics.

1. Run `/server/main.py` to start API server.
2. Run `/client/client.py` to start sending requests to the server.

### Martin Medvec

# Assignment used for Data Engineering Position

In this assignment you will build a simple online learning API that is also able to report the health of the API.

You may use any technology you find convenient to work on this. 

We supply a few tips for how to work on this in Python.

## The problem
Companies often have a lot of expenses. Each payment of these expenses is in accounting terms called a financial transaction which should be entered into the financial books. But where? This is the job of the accountant to figure out. An expense for paint might go to the maintenance account, and an expense for a taxi ride might go to the account for travel expenses.

To help the accountant we want to add machine learning. There's just one problem. We don't have a machine learning model. Yet. This is where you come in - in this assignment you will build a self-training machine learning API for bookkeeping. 

## The dataset

Included in this assignment is a dataset of examples of the kinds of transaction we want to process. Your goal is to learn how to predict the `AccountNumber` from these transactions. If you're interested in learning more about the dataset see the [appendix](#Appendix) below

## Part 1 - Online learning

Build a server that continuously trains a machine learning model. 

You can use the model contained in `model/model.py`. The quality of the model is not important to the test. 

Use any micro web framework you like to build the server - for Python Flask is an easy option.

The server has a JSON API with the following endpoints.

### `/sample`
Accepts POST requests

* Accepts a list of samples from the dataset.  The format is a list of JSON dictionaries using the keys from the CSV file - and strings for all values.
* Runs a prediction against the most recently trained model - if a model exists yet. Don't output to the user. You only need this to compute model metrics later.
* The samples should be added to the set of previously received samples
* Store the sample as received + the predicted `AccountNumber` - you don't have to overthink this - opening a file, adding the samples and storing it again is fine
* Train a new version of the machine learning model using all data received so far as the training set

### `/predict`
Accepts POST requests

* Accepts a single sample from the dataset. The format is a JSON dictionary using the keys from the CSV file - _except_ the `AccountName`,`AccountNumber`,`AccountTypeName` fields
* Predict the expected `AccountNumber` for a sample from the dataset
* Return value is a JSON list containing one string element

##  Part 2 - Client

Prepare a client to call the API. Start streaming in data from the dataset. 

##  Part 3 - Monitoring

In this part you will add the ability to report the learning progress of the system. 

You will add the following endpoint

### `/metrics/<n>`
Accepts GET requests

* Compare the submitted and predicted `AccountNumber`from the last `n` samples submitted.
* Report the precision and recall. In python you can use the `precision_score` and `recall_score` functions from 
[scikit learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) to compute this. 
* Return the values as a JSON dictionary with the keys `precision` and `recall`

Use your client to report the results of `/metrics/1000` after streaming in at least 10K samples.
 
## Guidelines

## Got stuck?
You can always email us and ask for advice or just ask question to ensure you correctly understood the task. This will not be seen as a sign of weakness, to the contrary it shows that fully understanding the problem is important to you.

## Suggestions for improvements?

Please help us improve this assignment by suggesting changes, or making a pull request.

## Appendix

The quality of the model is not evaluated as part of this exercise - but in case you're curious here's a little background on the data

The dataset consist of expenses from 100 random companies. For each company we provide all expenses that was booked in e-conomic.

Description of each column in the dataset:
- __CompanyId:__ The identifyer of the company to help you slice and dice the data in the right way.
- __BankEntryDate *(feature)*__: The date of the financial transaction.
- __BankEntryText *(feature)*__: The text following along with the financial transaction. This is typically machine generated, but in case of manual transactions they may be manually written by a human. _Please note that the text has been split into words before they have been hashed._
- __BankEntryAmount *(feature)*__: The amount of the financial transaction. Expenses are negative, earnings are positive.
- __AccountNumber *(target)*__: The account number. The uniquely identifies an account, and can therefore be used as the target variable / the class that we want to predict.
- __AccountName__: The name of the account.
- __AccountTypeName__: The type of the account.

Columns marked by _(feature)_ Can optionally be used as a feature in your predictive model. All of these features are typically what you see when you look at your bank statement. The _AccountNumber_ is your target variable. The _AccountName_ and the _AccountTypeName_ are properties of the account, and hence not of direct interest to the prediction problem, but if you can come up with creative ways of using it, then feel free to do so.

The rows are sorted first by _BankEntryDate_, then by _CompanyId_.

Due to privacy reasons the amounts has been bucketed and the texts has been obfuscated using the following function:

    data = query(limit = 100) # Pandas DataFrame

    def short_hash(word):
        try:
            int(word)
            typ = 'int'
        except:
            typ = 'str'
        bytes_ = word.encode() + secret_salt
        sha_word = hashlib.sha512(bytes_).hexdigest()
        return '{}:{}'.format(typ, sha_word[:7])

    def obfuscate_text(string_):
        return " ".join([short_hash(w) for w in string_.split()])

    def modify_row(row):
        # Translate AccountTypeName to english
        row['AccountTypeName'] = 'Profit and Loss' \
            if row['AccountTypeName'] == 'Drift' else 'Balance'
        # Obfuscate AccountName
        row['AccountName'] = short_hash(row['AccountName'])
        # Obfuscate BankEntryText
        row['BankEntryText'] = obfuscate_text(row['BankEntryText'])
        # Obfuscate CompanyId
        row['CompanyId'] = short_hash(row['CompanyId'])

        p_bar.update()
        return row

    data = data.apply(modify_row, axis=1)

    # Bin BankEntryAmount
    data['BankEntryAmount'] = pd.cut(
        data['BankEntryAmount'],
        bins=[float('-inf'), -10000, -1000, -100, -10, 0, 10, 100, 1000,
              10000, float('inf')],
        labels=['big negative', '> -10000', '> -1000', '> -100', '> -10',
                '< 10', '< 100', '< 1000', '< 10000', 'big positive']
    )

    data.to_csv(output_filename)

The data is a zipped `.csv` file called `bank_expenses_obfuscated.csv.zip`.

## Data example

Here's the top three rows from the data set. The Account* fields should be considered output variables, not input features.

|   | CompanyId   | BankEntryDate | BankEntryText           | BankEntryAmount | AccountName | AccountNumber | AccountTypeName |
|---|-------------|---------------|-------------------------|-----------------|-------------|---------------|-----------------|
| 0 | int:a055470 | 2016-02-29    | str:6cd08e4 int:49fed34 | > -1000         | str:1e82557 | 9900          | Balance         |
| 1 | int:a055470 | 2016-02-29    | str:6cd08e4 int:49fed34 | > -1000         | str:9ce853c | 3115          | Profit and Loss |
| 2 | int:a055470 | 2016-02-29    | str:38248d2             | > -100          | str:a9f0788 | 2240          | Profit and Loss |

