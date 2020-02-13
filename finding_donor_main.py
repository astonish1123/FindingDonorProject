import numpy as np
import pandas as pd

if __name__=='__main__':
    data = pd.read_csv('census.csv', sep=',')
    # print(data.head())

    n_records = data.shape[0]
    n_greater_50k = len(data[data['income']=='>50K'])
    n_at_most_50k = len(data[data['income']=='<=50K'])
    greater_percent = n_greater_50k/n_records
    print('Total number of records: {}'.format(n_records))
    print('Individuals making more than $50,000: {}'.format(n_greater_50k))
    print('Individuals making at most $50,000: {}'.format(n_at_most_50k))
    print('Percentage of individuals making more than $50,000: {}'.format(greater_percent))

    income_raw = data['income']
    features_raw = data.drop('income', axis=1)
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    print(features_log_transformed.head())
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x+1))
    print(features_log_transformed.head())