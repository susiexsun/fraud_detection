from flask import Flask, request, render_template
import cPickle as pickle
from pymongo import MongoClient
from predict import predict
import json
import requests
import socket
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

app = Flask(__name__)

@app.route('/')
def index():
	high = tab.find({'fraud_prob': {'$gt': 0.5}}).count()
	medium = tab.find({'fraud_prob': {'$lte': 0.5, '$gt': 0.0}}).count()
	low = tab.find({'fraud_prob': 0.0}).count()
	img = graph(high, medium, low)
	return render_template('index.html', high_var = high, med_var = medium, low_var = low, img_1 = img)

def graph(high, med, low): 
    labels = 'high risk', 'medium risk', 'low risk'
    sizes = [high, med, low]
    colors = ['indianred', 'palegoldenrod', 'lightgreen']
    explode = (0.1, 0, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True)


    f = tempfile.NamedTemporaryFile(dir='static', suffix='.png',delete=False)
    # save the figure to the temporary file
    plt.savefig(f)
    f.close() # close the file
    # get the file's name (rather than the whole path)
    # (the template will need that)
    plotPng = f.name.split('/')[-1]
    return plotPng


@app.route('/high_risk')
def high_risk():
	# Look into div='accordian' (Bootstrap) for foldout tables
	high = tab.find({'fraud_prob': {'$gt': 0.5}})
	df = pd.DataFrame(list(high))
	return '<link href="static/css/css_table.css" rel="stylesheet">' + df.to_html(classes="CSSTableGenerator")
	# with "high_risk.html" open as f:
	# 	f.write(html_df)
	#return '<link href="static/css/css_table.css" rel="stylesheet"><div class="CSSTableGenerator">' + df.to_html(classes="") + '</div>'

@app.route('/med_risk')
def med_risk():
	medium = tab.find({'fraud_prob': {'$lte': 0.5, '$gt': 0.0}})
	df = pd.DataFrame(list(medium))
	return '<link href="static/css/css_table.css" rel="stylesheet">' + df.to_html(classes="CSSTableGenerator")


@app.route('/low_risk')
def low_risk():
	low = tab.find({'fraud_prob': 0.0})
	df = pd.DataFrame(list(low))
	return '<link href="static/css/css_table.css" rel="stylesheet">' + df.to_html(classes="CSSTableGenerator")



# Hello World
@app.route('/stuff')
def hello_world():
    return '''
    Hello, World!
    '''

@app.route('/score', methods=['POST'])
def score():
	fraud_data = request.get_json(force=True)
	df = predict(rf_top, fraud_data)
	tab.insert_many(df.to_dict('records'))
	return '''
	SCORE!
	'''

def register():
    my_ip = socket.gethostbyname(socket.gethostname())
    my_port = 7777
    reg_url = 'http://10.3.34.86:5000/register'
    requests.post(reg_url, data={'ip': my_ip, 'port': my_port})
    return ''
    #r1 = requests.post("http://0.0.0.0:8080/score", json=r)


if __name__ == '__main__':
    with open('data/random_forest.pkl') as f:
        rf_top = pickle.load(f)
    client = MongoClient()
    db = client['fraud_db']
    tab = db['fraud']
    register()
    app.run(host='0.0.0.0', port=7777, debug=True)