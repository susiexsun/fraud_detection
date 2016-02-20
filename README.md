
# Goal

Build a fraud detection model for a hypothetical ecommerce company.

#### The Data: 
- ~15k rows of static user attributes (org name, org description, etc) in JSON
- Log of user activity (account creation timestamps, sale timestamps, etc) nested within the same file

#### The Deliverables: 
- Model that minimizes false negatives
- Interactive front-end dashboard

This project was built over 48 hours as a collaboration between myself and Kevin Lee (kmlee17). 

# The Features

After extensive EDA, we carved out a subset of features.  
EDA was conducted in iPython Notebook (aka Jupyter Notebook) and can be found in data/pair.ipynb

Two key tools <b>feature selection</b>: 
- Histograms of fraud (left) and non-fraud accounts (right)

![Not Fraud Hist](http://s10.postimg.org/gvo4mu94p/hist_not_fraud.jpg  = 150 x 150)
![Fraud Hist](http://s16.postimg.org/aatvh098l/hist_fraud.jpg)
- Feature importance graph

<b>Feature Engineering</b> is very important due to data with timestamps and free text. Below is a chart of the features engineered.

![](http://s29.postimg.org/63gx2h7hj/feature_engineering.jpg)

# The Model

<b>Class Imbalance</b>. There were far more 0 (non-frauds) than 1 (frauds). We used the smoting method to rebalance the classes. 

We tested multiple models, and Gradient Boosted Classifier worked the best. 

![](http://s8.postimg.org/ja8zbqq2d/models.jpg)

# The Dashboard


