# python predict.py test_script_examples.csv
import sys
import pandas as pd
#from data_processing import add_dummies, length_of_feature, fuzzy
import data_processing as dp

def process_data(data): 
    df = pd.read_json(data)

    #convert_unix_timestamp(df, 'event_created')
    #convert_unix_timestamp(df, 'event_start')
    #convert_unix_timestamp(df, 'event_end')
    # Filling empty values
    #df['event_published'] = df['event_published'].fillna(1352831618)
    #convert_unix_timestamp(df, 'event_published')

    #convert_unix_timestamp(df, 'user_created')


    # Countries dummies ('US' as baseline)
    df = dp.add_dummies(df, 'country', 'US')
    # Currency dummies ('USD' as baseline)
    df = dp.add_dummies(df, 'currency', 'USD')
    # Payouts dummies ('ACH' as baseline)
    df = dp.add_dummies(df, 'payout_type', 'ACH')

    #featurizing strings as 'length'
    dp.length_of_feature(df, 'description', 'len_desc')
    dp.length_of_feature(df, 'previous_payouts', 'len_pp')
    dp.length_of_feature(df, 'org_desc', 'len_org_desc')
    dp.length_of_feature(df, 'ticket_types', 'len_tt')

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
    df.drop(['venue_state', 'venue_name', 'venue_address', 'venue_country'], axis=1, inplace=True)

    # feature engineering suspicious emails
    sus_domains = ["gmail.com", "yahoo.com", "hotmail.com", "ymail.com","aol.com", \
                   "lidf.co.uk"," live.com", "live.fr", "yahoo.co.uk", "rocketmail.com"]
    df['sus_domain'] = map(lambda x: True if x in sus_domains else False, df['email_domain'])
    df.drop(['email_domain'], axis=1, inplace=True)

    # creating similiarity score for payee vs org name features
    df['fuzzy_sim'] = df.apply(dp.fuzzy, axis=1)

    df.drop(['name', 'org_name', 'payee_name'], axis=1, inplace=True)


    return df


def predict(X):
	X_processed = process_data(X)
	return X_processed
	# unpack pickled model
	# predict X_processed on pickled model
	# return prediction

if __name__ == '__main__':
	test = predict(sys.argv[1])