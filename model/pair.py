import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher as SM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

def convert_unix_timestamp(df, column_name):
	df[column_name] = df[column_name].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

def add_dummies(df, column_name, baseline):
	dummies = pd.get_dummies(df[column_name])
	dummies.drop(baseline, axis=1, inplace=True)
	df.drop(column_name, axis=1, inplace=True)
	return pd.concat((df, dummies), axis=1)

def length_of_feature(df, column_name, length_col_name, drop_orig=True):
	df[length_col_name] = map(lambda x: len(x), df[column_name])
	if drop_orig == True:
		df.drop(column_name, axis=1, inplace=True)

def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class 
             observations in the output
    k - k in k nearest neighbors
    OUTPUT:
    X_oversampled, y_oversampled - oversampled data
    `smote` generates new observations from the positive (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    if target <= sum(y)/float(len(y)):
        return X, y
    if k is None:
        k = len(X)**.5
    # fit kNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[y==1], y[y==1])
    neighbors = knn.kneighbors()[0]
    positive_observations = X[y==1]
    # determine how many new positive observations to generate
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    target_positive_count = target*negative_count / (1. - target)
    target_positive_count = int(round(target_positive_count))
    number_of_new_observations = target_positive_count - positive_count
    # generate synthetic observations
    synthetic_observations = np.empty((0, X.shape[1]))
    while len(synthetic_observations) < number_of_new_observations:
        obs_index = np.random.randint(len(positive_observations))
        observation = positive_observations[obs_index]
        neighbor_index = np.random.choice(neighbors[obs_index])
        neighbor = X[neighbor_index]
        obs_weights = np.random.random(len(neighbor))
        neighbor_weights = 1 - obs_weights
        new_observation = obs_weights*observation + neighbor_weights*neighbor
        synthetic_observations = np.vstack((synthetic_observations, new_observation))

    X_smoted = np.vstack((X, synthetic_observations))
    y_smoted = np.concatenate((y, [1]*len(synthetic_observations)))

    return X_smoted, y_smoted

df = pd.read_json('data/train_new.json')

# Find rows that are fraudulent
frauds = df['acct_type'].str.contains('fraudster')

df['fraud'] = frauds

convert_unix_timestamp(df, 'event_created')
convert_unix_timestamp(df, 'event_start')
convert_unix_timestamp(df, 'event_end')
# Filling empty values
df['event_published'] = df['event_published'].fillna(1352831618)
convert_unix_timestamp(df, 'event_published')

convert_unix_timestamp(df, 'user_created')

# Countries dummies ('US' as baseline)
df = add_dummies(df, 'country', 'US')
# Currency dummies ('USD' as baseline)
df = add_dummies(df, 'currency', 'USD')
# Payouts dummies ('')

length_of_feature(df, 'description', 'len_desc')
length_of_feature(df, 'previous_payouts', 'len_pp')
length_of_feature(df, 'org_desc', 'len_org_desc')