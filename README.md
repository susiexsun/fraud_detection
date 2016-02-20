
# Goal

Build a fraud detection model for a hypothetical ecommerce company.

#### The Data: 
- ~15k rows of static user attributes (org name, org description, etc) in JSON
- Log of user activity (account creation timestamps, sale timestamps, etc) nested within the same file

#### The Deliverables: 
- Model that minimizes false negatives
- Interactive front-end dashboard that receives and classifies a live stream of data

This project was built over 48 hours as a collaboration between myself and Kevin Lee (kmlee17). 

# The Features

After extensive EDA, we carved out a subset of features.  

Two key tools <b>feature selection</b>: 
- Histograms of fraud (left) and non-fraud accounts (right)

![Not Fraud Hist](http://s10.postimg.org/gvo4mu94p/hist_not_fraud.jpg  = 150 x 150)
![Fraud Hist](http://s16.postimg.org/aatvh098l/hist_fraud.jpg)
- Feature importance graph

<b>Class Imbalance</b>. There were far more 0 (non-frauds) than 1 (frauds). We used the smoting method to rebalance the classes. 

<b>Feature Engineering</b> is very important due to data with timestamps and free text. Below is a chart of the features engineered.

![](http://s29.postimg.org/63gx2h7hj/feature_engineering.jpg)

# The Model

We tested multiple models, and <b>Gradient Boosted Classifier</b> worked the best based on 5 fold cross validation.

![](http://s8.postimg.org/ja8zbqq2d/models.jpg)

# The Dashboard

We created an interactive dashboard of the accounts / transactions. We graded those as High, Medium, and Low (or No) Risk. 

![](http://s14.postimg.org/n182vt9kh/dashboard.jpg)

Our model was great - it had a high recall and did not overload the system with false positives. 

# Repo Structure
![](http://s24.postimg.org/ou6xuioxh/repo_structure.jpg)
