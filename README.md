
# Goal

Build a fraud detection model for a hypothetical ecommerce company.

#### The Data: 
- ~15k rows of static user attributes (org name, org description, etc) in JSON
- Log of user activity (account creation timestamps, sale timestamps, etc) nested within the same file

#### The Deliverables: 
- Model that minimizes false negatives
- Flask app and interactive front-end dashboard

This project was built over 48 hours as a collaboration between myself and Kevin Lee (kmlee17). 

# The Features

After extensive EDA, we carved out a subset of features.  
EDA was conducted in iPython Notebook (aka Jupyter Notebook) and can be found in data/pair.ipynb

Two key tools <b>feature selection</b>: 
- Histograms of fraud and non-fraud accounts
- Feature importance graph

![Not Fraud Hist](http://s23.postimg.org/sagaarre3/hist_not_fraud.jpg)
![Fraud Hist](http://s28.postimg.org/512xaub99/hist_fraud.jpg)

<b>Feature Engineering</b> is very important due to: 
- Free Text
- Timestamps


# The Dashboard


