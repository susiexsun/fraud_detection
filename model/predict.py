# python predict.py example.json
import sys
import pandas as pd
import data_processing as dp
import cPickle as pickle
from pymongo import MongoClient


def predict(model, data):
    #df = pd.read_json(data)
    df = pd.DataFrame([data])
    X_top = dp.process_data(df)
    # with open('data/random_forest.pkl') as f:
    #     rf_top = pickle.load(f)

    # Using unpickled model to predict on new new data
    y_pred = model.predict_proba(X_top)

    # Adds new column in df with predicted probability of fraud
    df['fraud_prob'] = y_pred[:,1]
    return df

def store_in_db(df):
    client = MongoClient()
    db = client['fraud_db']
    tab = db['fraud']
    tab.insert_many(df.to_dict('records'))


if __name__ == '__main__':
    df = predict(sys.argv[1])
    store_in_db(df)