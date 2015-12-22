import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher as SM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import datetime

# Taken from the pickled random forest model feature importance
# top_features = ['object_id', 'CHECK', 'body_length', 'event_published', 'has_logo',
#                 'user_created', 'venue_longitude', 'gts', 'GBP', 'org_twitter',
#                 'org_facebook', 'num_order', 'sus_domain', 'user_age', 'sale_duration',
#                 'sale_duration2', 'BLANK', 'delivery_method','user_type', 'len_pp']


top_features = ['event_published', 'len_tt','len_desc', 'delivery_method', 'GBP',
               'venue_latitude', 'venue_longitude', 'num_payouts','user_created',
               'has_logo', 'org_facebook', 'gts', 'num_order', 'sale_duration',
               'user_age', 'sale_duration2', 'BLANK', 'sus_domain', 'user_type', 'len_pp']

def convert_unix_timestamp(df, column_name):
    df[column_name] = df[column_name].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))


def add_dummies(df, column_name, baseline):
    dummies = pd.get_dummies(df[column_name])
    if column_name == 'payout_type':
        dummies.rename(columns={'': 'BLANK'}, inplace=True)
    dummies.drop(baseline, axis=1, inplace=True)
    df.drop(column_name, axis=1, inplace=True)
    return pd.concat((df, dummies), axis=1)


def length_of_feature(df, column_name, length_col_name, drop_orig=True):
    df[length_col_name] = map(lambda x: len(x), df[column_name])
    if drop_orig == True:
        df.drop(column_name, axis=1, inplace=True)


def fuzzy(x):
    return SM(None, x['org_name'], x['payee_name']).ratio()

def smote(X, y, minority_weight=.5):
    '''
    generates new observations in minority class
    so that output X, y have specified percentage of majority observations
    '''
    # compute number of new examples required
    class_ratio = y.sum()/float(len(y))
    majority_class_label = round(class_ratio)
    X_minority = X[y!=majority_class_label]
    y_minority = y[y!=majority_class_label]
    min_count = len(X_minority)
    maj_count = len(X) - min_count
    scaling_factor = (maj_count/float(min_count))*(minority_weight/(1-minority_weight))
    new_observations_target = round(scaling_factor*min_count) - min_count

    # train KNN
    knn_model = KNeighborsClassifier(n_neighbors=int(round(len(X_minority)**.5)))
    knn_model.fit(X_minority, y_minority)
    if new_observations_target < len(X_minority):
        sample_indices = np.random.choice(xrange(X_minority), 
                                          size=new_observations_target,
                                          replace=False)
        smote_samples = X_minority[sample_indices]
    else:
        smote_samples = X_minority
    neighbors = knn_model.kneighbors(smote_samples)[1]
    
    # generate new samples
    new_observations = np.empty((0,X.shape[1]))
    while len(new_observations) < new_observations_target:
        index = len(new_observations) % len(smote_samples)
        neighbor_index = np.random.choice(neighbors[index])
        neighbor = smote_samples[neighbor_index]
        x = X_minority[index]
        new_x = x + (neighbor - x)*np.random.random(size=X_minority.shape[1])
        new_observations = np.vstack((new_observations, new_x))
    minority_class_label = (majority_class_label + 1) % 2
    X = np.vstack((X, new_observations))
    y = np.hstack((y, np.array([minority_class_label]*len(new_observations))))
    
    return X, y


def feature_importance(model, dfX, dfy):
    fi = model.feature_importances_
    yticks = X.columns[np.argsort(fi)].values
    fi = sorted(fi)
    x = xrange(len(yticks))
    plt.figure(figsize=(15, 15))
    plt.barh(x, fi)
    plt.yticks(x, yticks)
    plt.tight_layout()


# ## Read in data
def process_data(df): 
    #convert_unix_timestamp(df, 'event_created')
    #convert_unix_timestamp(df, 'event_start')
    #convert_unix_timestamp(df, 'event_end')
    # Filling empty values
    #df['event_published'] = df['event_published'].fillna(1352831618)
    #convert_unix_timestamp(df, 'event_published')

    #convert_unix_timestamp(df, 'user_created')
        
    # Currency dummies ('GBP' is the only feature used in the model)
    try: 
        if df['currency'] == 'GBP': 
            df['GBP'] = 1
        else: 
            df['GBP'] = 0
    except: 
        df['GBP'] = 0

    # Payouts dummies ('ACH' as baseline)
    try: 
        if df['payout_type'] == '':
            df['BLANK'] = 1
        else: 
            df['BLANK'] = 0
    except: 
        df['BLANK'] = 0

    #featurizing strings as 'length'
    length_of_feature(df, 'description', 'len_desc')
    length_of_feature(df, 'previous_payouts', 'len_pp')
    length_of_feature(df, 'org_desc', 'len_org_desc')
    length_of_feature(df, 'ticket_types', 'len_tt')

    # imputting missing values
    df['has_header'] = df['has_header'].fillna(2)
    df['venue_latitude'] = df['venue_latitude'].fillna(0.)
    df['venue_longitude'] = df['venue_longitude'].fillna(0.)
    df['sale_duration'] = df['sale_duration'].fillna(df['sale_duration'].mean())
    df['org_facebook'] = df['org_facebook'].fillna(0)
    df['org_twitter'] = df['org_twitter'].fillna(0)
    df['event_published'] = df['event_published'].fillna(0)
    df['delivery_method'] = df['delivery_method'].fillna(0)

    # making 'listed' legible
    df['listed'] = df['listed'].map({'y': 1, 'n': 0})

    # dropping less useful columns
    df.drop(['venue_state', 'venue_name', 'venue_address', 'venue_country'], axis=1, inplace=True)

    # feature engineering suspicious emails
    sus_domains = ["gmail.com", "yahoo.com", "hotmail.com", "ymail.com","aol.com", \
                   "lidf.co.uk"," live.com", "live.fr", "yahoo.co.uk", "rocketmail.com"]
    df['sus_domain'] = map(lambda x: True if x in sus_domains else False, df['email_domain'])
    df.drop(['email_domain'], axis=1, inplace=True)

    # creating similiarity score for payee vs org name features
    df['fuzzy_sim'] = df.apply(fuzzy, axis=1)

    df.drop(['name', 'org_name', 'payee_name'], axis=1, inplace=True)

    # Creating the top20 feature matrix
    df_top = df[top_features]
    return df_top

# In[7]:
def process_model(data): 
    df = pd.read_json(data)

    # Find rows that are fraudulent
    frauds = df['acct_type'].str.contains('fraudster')
    df['fraud'] = frauds

    #convert_unix_timestamp(df, 'event_created')
    #convert_unix_timestamp(df, 'event_start')
    #convert_unix_timestamp(df, 'event_end')
    # Filling empty values
    #df['event_published'] = df['event_published'].fillna(1352831618)
    #convert_unix_timestamp(df, 'event_published')

    #convert_unix_timestamp(df, 'user_created')


    # Countries dummies ('US' as baseline)
    df = add_dummies(df, 'country', 'US')
    # Currency dummies ('USD' as baseline)
    df = add_dummies(df, 'currency', 'USD')
    # Adds 'GBP' column if it wasn't created in add_dummies
    
    # Payouts dummies ('ACH' as baseline)
    df = add_dummies(df, 'payout_type', 'ACH')

    # df_payout.rename(columns={'': 'BLANK'}, inplace=True)

    #featurizing strings as 'length'
    length_of_feature(df, 'description', 'len_desc')
    length_of_feature(df, 'previous_payouts', 'len_pp')
    length_of_feature(df, 'org_desc', 'len_org_desc')
    length_of_feature(df, 'ticket_types', 'len_tt')

    # imputing missing values
    df['has_header'] = df['has_header'].fillna(2)
    df['venue_latitude'] = df['venue_latitude'].fillna(0.)
    df['venue_longitude'] = df['venue_longitude'].fillna(0.)
    df['sale_duration'] = df['sale_duration'].fillna(df['sale_duration'].mean())
    df['org_facebook'] = df['org_facebook'].fillna(0)
    df['org_twitter'] = df['org_twitter'].fillna(0)
    df['event_published'] = df['event_published'].fillna(0)
    df['delivery_method'] = df['delivery_method'].fillna(0)

    # making 'listed' legible
    df['listed'] = df['listed'].map({'y': 1, 'n': 0})

    # dropping less useful columns
    df.drop(['venue_state', 'venue_name', 'venue_address'], axis=1, inplace=True)
    df.drop(['acct_type', 'venue_country'], axis=1, inplace=True)

    # feature engineering suspicious emails
    sus_domains = ["gmail.com", "yahoo.com", "hotmail.com", "ymail.com","aol.com", \
                   "lidf.co.uk"," live.com", "live.fr", "yahoo.co.uk", "rocketmail.com"]
    df['sus_domain'] = map(lambda x: True if x in sus_domains else False, df['email_domain'])
    df.drop(['email_domain'], axis=1, inplace=True)

    # creating similiarity score for payee vs org name features
    df['fuzzy_sim'] = df.apply(fuzzy, axis=1)

    df.drop(['name', 'org_name', 'payee_name'], axis=1, inplace=True)


    y = df['fraud']
    X = df.drop('fraud', axis=1)


    X_smoted, y_smoted = smote(X.values, y.values, minority_weight=.3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X, y, X_smoted, y_smoted, X_train, X_test, y_train, y_test





# ## Fitting Random Forest

def modeling(X): 

    rf = RandomForestClassifier()
    rf.fit(X_smoted, y_smoted)


    # In[112]:

    feature_importance(rf, X_smoted, y_smoted)


    # In[116]:

    rf.score(X_smoted, y_smoted)


    # In[26]:

    y_pred = rf.predict(X_test)


    # In[27]:

    confusion_matrix(y_test, y_pred)


    # In[28]:

    tpr = 2614. / (2614 + 1)


    # In[29]:

    print "Accuracy Score: ", rf.score(X, y)
    print "True Positive Rate: ", tpr


    # ### Random Forest on Top 20 Features

    # In[33]:

    yticks = X.columns[np.argsort(rf.feature_importances_)].values
    features = yticks[-20:]


    # In[39]:

    X_top = X[features]


    # In[44]:

    X_top_smoted, y_smoted = smote(X_top.values, y.values, minority_weight=.3)


    # In[46]:

    X_top_train, X_top_test, y_train, y_test = train_test_split(X_top, y, test_size=0.20, random_state=42)


    # In[45]:

    rf_top = RandomForestClassifier()
    rf_top.fit(X_top_smoted, y_smoted)


    # In[47]:

    rf_top.score(X_top_test, y_test)


    # In[48]:

    y_pred = rf_top.predict(X_top_test)


    # In[53]:

    confusion_matrix(y_test, y_pred)


    # ## Trying Other Models

    # In[54]:

    # Grid Search on SVM
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 0.1]}
    svm = SVC()
    gs = GridSearchCV(svm, parameters, scoring='recall')


    # In[ ]:

    gs.fit(X_top_train, y_train)


    # In[ ]:

    # Grid Search on GBR
    gradient_boosting_grid = {'learning_rate':[0.01, 0.05, 0.1, 0.5], 
                              'max_depth': [1, 3, 5, 7],
                              'min_samples_leaf': [1, 2, 4, 6],
                              'n_estimators': [20, 100, 200]}
    gbr = GradientBoostingClassifier()
    gs_gbr = GridSearchCV(gbr, gradient_boosting_grid, scoring='recall')

    pass


# In[ ]:



