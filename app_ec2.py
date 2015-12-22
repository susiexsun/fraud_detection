from flask import Flask, request, render_template
import cPickle as pickle
from pymongo import MongoClient
from predict import predict
import json
import requests
import socket
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
	high = tab.find({'fraud_prob': {'$gt': 0.5}}).count()
	medium = tab.find({'fraud_prob': {'$lte': 0.5, '$gt': 0.0}}).count()
	low = tab.find({'fraud_prob': 0.0}).count()
	return render_template('index.html', high_var = high, med_var = medium, low_var = low)

@app.route('/high_risk')
def high_risk():
	high = tab.find({'fraud_prob': {'$gt': 0.5}})
	df = pd.DataFrame(list(high))
	return df.to_html()

@app.route('/med_risk')
def med_risk():
	medium = tab.find({'fraud_prob': {'$lte': 0.5, '$gt': 0.0}})
	df = pd.DataFrame(list(medium))
	return df.to_html()

@app.route('/low_risk')
def low_risk():
	low = tab.find({'fraud_prob': 0.0})
	df = pd.DataFrame(list(low))
	return df.to_html()



# Hello World
@app.route('/stuff')
def hello_world():
    return '''
    Hello, World!
    '''

@app.route('/score')
def score():
	fraud_data = request.get_json(force=True)
	df = predict(rf_top, fraud_data)
	tab.insert_many(df.to_dict('records'))
	return '''
	SCORE!
	'''


if __name__ == '__main__':
    with open('data/random_forest.pkl') as f:
        rf_top = pickle.load(f)
    df = pd.read_json('example.json')
    app.run(host='52.23.249.11', port=7777, debug=True)