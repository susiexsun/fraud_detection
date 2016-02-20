
# Goal

Build a fraud detection model for a hypothetical ecommerce company.

#### The Data (JSON files): 
- ~15k rows of static user attributes (org name, org description, etc)
- Log of user activity (account creation timestamps, sale timestamps, etc) 

#### The Deliverables: 
- Model that minimizes false negatives
- Flask app and interactive front-end dashboard

This project was built over 48 hours as a collaboration between myself and Kevin Lee (kmlee17). 

# The Features

After extensive EDA, we carved out a subset of features.  
EDA was conducted in iPython Notebook (aka Jupyter Notebook) and can be found in data/pair.ipynb

Two key charts for feature selection: 
- Histograms of fraud and non-fraud accounts
- Feature importance graph

Feature Engineering is very important to this model, since there was a lot of free text and timestamps.


# The Dashboard


