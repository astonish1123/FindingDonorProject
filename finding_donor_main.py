import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    data = pd.read_csv('census.csv', sep=',')
    # print(data.head())

    n_records = data.shape[0]
    n_greater_50k = len(data[data['income']=='>50K'])
    n_at_most_50k = len(data[data['income']=='<=50K'])
    greater_percent = float(n_greater_50k)/n_records
    print('Total number of records: {}'.format(n_records))
    print('Individuals making more than $50,000: {}'.format(n_greater_50k))
    print('Individuals making at most $50,000: {}'.format(n_at_most_50k))
    print('Percentage of individuals making more than $50,000: {}'.format(greater_percent))

    income_raw = data['income']
    features_raw = data.drop('income', axis=1)
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x+1))

    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    features_final = pd.get_dummies(data=features_log_minmax_transform, columns=['workclass', 'education_level',
                                                                                'marital-status', 'occupation',
                                                                                'relationship', 'race', 'sex', 
                                                                                'native-country'])
    income = income_raw.map({'<=50K': 0, '>50K': 1})
    encoded = list(features_final.columns)
    print('{} total features after one-hot encoding.'.format(len(encoded)))

    X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size=0.2, random_state=0)
    print('Training set has {} samples.'.format(X_train.shape[0]))
    print('Testing set has {} samples.'.format(X_test.shape[0]))

    TP = np.sum(income)
    FP = income.count() - TP
    TN = 0
    FN = 0
    accuracy = float(TP) / (TP+FP)
    recall = float(TP) / (TP+FN)
    precision = accuracy
    fscore = (1 + 0.5**2)*(precision*recall) / (0.5**2*precision+recall)
    print('Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}'.format(accuracy, fscore))